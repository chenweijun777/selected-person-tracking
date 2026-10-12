"""The single public evaluation entry point."""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import frame as frame_scoring
from ._adapters import target_adapter, tracker_adapter
from ._common import (
    EVENT_FIELDS,
    GLOBAL_FIELDS,
    ROOT,
    canonical_split,
    csv_rows,
    read_json,
    record,
    portable_record,
    project_path,
    resolve_asset,
    sha,
    write_csv,
)
from ._scoring import (
    event_details,
    global_mot,
    recovery_summary,
    target_assessment,
    target_summary,
    tracker_matches,
)


def evaluate_run(
    run_name,
    prediction_path,
    run_type,
    config_path=None,
    output_dir=None,
    split="development",
    *,
    strict=True,
):
    """Score a saved run, write reports, and return the metrics dict.

    Rates are percentages; delays are seconds. strict=False labels recovery results
    provisional. Config's optional evaluation mapping selects existing manifests.
    Relative paths resolve from the project root. Reports never overwrite.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run_name must be a safe directory name")
    split = canonical_split(split)
    if run_type not in ("tracker", "target_manager") or split not in (
        "development",
        "validation",
        "test",
    ):
        raise ValueError(
            "run_type must be tracker/target_manager; split development/validation/test"
        )
    if not isinstance(strict, bool):
        raise TypeError("strict must be bool")
    path = project_path(prediction_path)
    config = project_path(config_path) if config_path else None
    if config is None:
        base = path if path.is_dir() else path.parent
        config = next(
            (p for p in (base / "config.yaml", base / "metadata.json") if p.is_file()), None
        )
    if config is None:
        raise ValueError(
            "Provide saved run config or adjacent config.yaml/metadata.json; parameters cannot be inferred"
        )
    settings = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise TypeError("Run config must be a mapping")
    options = settings.get("evaluation", {})
    if not isinstance(options, dict):
        raise TypeError("evaluation config must be a mapping")
    unknown = set(options) - {"protocol", "events", "visibility", "episode_manifest"}
    if unknown:
        raise ValueError(f"Unknown evaluation options (thresholds are frozen): {unknown}")
    scope = ROOT / "protocols/current" / ("test" if split == "test" else "devval")
    options = {"episode_manifest": str(scope / "episode_manifest.json"), **options}
    protocol_path = project_path(
        options.get("protocol", ROOT / "protocols/current" / f"{split}.json")
    )
    protocol = read_json(protocol_path)
    event_path = project_path(options.get("events", scope / "gt_events.json"))
    visibility_path = project_path(
        options.get("visibility", event_path.with_name("visibility_intervals.csv"))
    )
    event_payload = read_json(event_path)
    if event_payload["rules"]["main_visibility_threshold"] != 0.3:
        raise ValueError("Unsupported visibility protocol change")
    sequences = [
        s for s, v in protocol["sequences"].items() if canonical_split(v["split"]) == split
    ]
    if not sequences:
        raise ValueError(f"Empty split: {split}")
    gt_root = ROOT / "data/MOT17/train"
    # Verify published evaluation inputs when they are present.
    dependency_paths = [protocol_path, event_path, visibility_path, event_path.with_suffix(".csv")]
    dependencies = [record(p) for p in dependency_paths]
    for item in protocol.get("files", []):
        source = resolve_asset(item["path"], required=False)
        if source is not None and sha(source) != item["sha256"]:
            raise ValueError(f"Frozen asset hash mismatch: {source}")
    config_record = record(config)
    if (
        settings.get("predictions_sha256") is not None
        and path.is_file()
        and sha(path) != settings["predictions_sha256"]
    ):
        raise ValueError("Prediction does not match saved run metadata")
    if (
        config.name == "metadata.json"
        and settings.get("config_sha256")
        and not settings.get("effective_config")
    ):
        raise ValueError("Metadata has only a config hash; provide the actual run config")
    if settings.get("protocol_sha256") not in (None, sha(protocol_path)):
        raise ValueError("Run metadata uses a different protocol")
    truth, lengths = {}, {}
    for seq in sequences:
        source = resolve_asset(protocol["sequences"][seq]["gt"])
        dependencies.extend(
            [record(source), record(resolve_asset(protocol["sequences"][seq]["seqinfo"]))]
        )
        if sha(source) != event_payload["sources"][seq]["gt_sha256"]:
            raise ValueError(f"Event GT differs: {seq}")
        truth[seq] = frame_scoring.load_gt(source)
        lengths[seq] = event_payload["sources"][seq]["length"]
    intervals = defaultdict(list)
    for r in csv_rows(visibility_path):
        intervals[r["sequence"], int(r["gt_id"])].append(
            (int(r["start_frame"]), int(r["end_frame"]), r["state"])
        )
    event_ids = [e["event_id"] for e in event_payload["events"]]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Duplicate event IDs")
    csv_events = csv_rows(event_path.with_suffix(".csv"))
    if set(event_ids) != {e["event_id"] for e in csv_events}:
        raise ValueError("Event CSV and JSON IDs differ")
    # Check selected visibility timelines against GT; gaps remain unknown.
    for (seq, pid), spans in intervals.items():
        if seq not in sequences:
            continue
        expected_start = 1
        for a, b, state in sorted(spans):
            if a != expected_start or b < a:
                raise ValueError(f"Non-contiguous visibility manifest: {seq}/{pid}")
            for f in range(a, b + 1):
                gt = truth[seq].get(f, {}).get(pid)
                expected = (
                    "unannotated"
                    if gt is None
                    else "main_visible"
                    if gt[1] >= 0.3
                    else "low_visible"
                    if gt[1] > 0
                    else "invisible"
                )
                if state != expected:
                    raise ValueError(f"Visibility manifest differs from GT: {seq}/{pid}/{f}")
            expected_start = b + 1
        if expected_start != lengths[seq] + 1:
            raise ValueError(f"Incomplete visibility timeline: {seq}/{pid}")
    if any(
        (e["sequence"], e["gt_id"]) not in intervals
        for e in event_payload["events"]
        if e["sequence"] in sequences
    ):
        raise ValueError("Event target missing visibility timeline")
    output = project_path(output_dir) if output_dir else ROOT / "outputs/evaluations" / run_name
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    input_paths = [Path(d["path"]) for d in dependencies] + [path, config]
    if any(
        output == p or output in p.parents or (p.is_dir() and p in output.parents)
        for p in input_paths
    ):
        raise ValueError("Output must be separate from all input assets")
    if run_type == "tracker":
        files, predictions = tracker_adapter(path, sequences, lengths)
        pred_records = [record(p) for p in files.values()]
        assignments = tracker_matches(predictions, truth)
        events = event_details(
            event_payload["events"],
            protocol,
            sequences,
            strict,
            run_type,
            truth,
            intervals,
            assignments=assignments,
            predictions=predictions,
        )
    else:
        cases, predictions = target_adapter(path, protocol, sequences, options, dependencies)
        pred_records = [record(path)]
        details, people, frames = target_assessment(cases, predictions, truth)
        events = event_details(
            event_payload["events"],
            protocol,
            sequences,
            strict,
            run_type,
            truth,
            intervals,
            target_frames=frames,
        )
    version = importlib.metadata.version("trackeval")
    if version != "1.3.0":
        raise ValueError(f"Expected frozen TrackEval 1.3.0, got {version}")
    output.mkdir(parents=True)
    global_metrics, per_global = dict.fromkeys(GLOBAL_FIELDS), {}
    if run_type == "tracker":
        global_metrics, per_global = global_mot(files, output, gt_root)
    recovery = recovery_summary(events)
    metrics = dict(
        schema_version=1,
        run_name=run_name,
        run_type=run_type,
        split=split,
        strict=strict,
        provisional=not strict,
        units={"rates": "percent", "delays": "seconds"},
        **{"global": global_metrics},
        recovery=recovery,
    )
    if run_type == "target_manager":
        metrics["target"] = target_summary(details, people, recovery)
        write_csv(output / "target_details.csv", details)
        write_csv(output / "initialization_details.csv", people)
    sequence_rows = []
    for seq in sequences:
        rec = recovery_summary([e for e in events if e["sequence"] == seq])
        row = dict(
            sequence=seq,
            provisional=not strict,
            **per_global.get(seq, dict.fromkeys(GLOBAL_FIELDS)),
            **rec,
        )
        if run_type == "target_manager":
            target = target_summary(
                [r for r in details if r["sequence"] == seq],
                [r for r in people if r["sequence"] == seq],
                rec,
            )
            row.update({k: v for k, v in target.items() if not isinstance(v, dict)})
        sequence_rows.append(row)
    write_csv(output / "per_sequence.csv", sequence_rows)
    write_csv(output / "event_details.csv", events, EVENT_FIELDS)
    flat = {"run_name": run_name, "run_type": run_type, "split": split, "provisional": not strict}
    for section in ("global", "recovery", "target"):
        flat.update(
            {
                f"{section}.{k}": v
                for k, v in metrics.get(section, {}).items()
                if not isinstance(v, dict)
            }
        )
    write_csv(output / "summary.csv", [flat])
    if config.suffix.lower() in (".yaml", ".yml"):
        shutil.copyfile(config, output / "config_snapshot.yaml")
    else:
        (output / "config_snapshot.yaml").write_text(
            yaml.safe_dump(portable_record(settings), allow_unicode=True), encoding="utf-8"
        )
    code_paths = sorted(Path(__file__).parent.glob("*.py")) + [
        ROOT / "src/a1_tracker/geometry.py",
        ROOT / "src/a1_tracker/types.py",
    ]
    # Preserve dirty/uncommitted evaluator code as well as its commit and hashes.
    for source in code_paths:
        destination = output / "evaluation_source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    commit = None
    if (ROOT / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
        )
        if revision.returncode == 0:
            commit = revision.stdout.strip()
    manifest = {
        "run_name": run_name,
        "run_type": run_type,
        "prediction_path": str(path),
        "predictions": pred_records,
        "config": config_record,
        "config_snapshot": record(output / "config_snapshot.yaml"),
        "assets": dependencies,
        "evaluation_code_commit": commit,
        "evaluation_code": [record(p) for p in code_paths],
        "evaluation_source_snapshot": "evaluation_source/",
        "trackeval_version": version,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "sequences": sequences,
        "iou_threshold": 0.5,
        "strict": strict,
        "provisional": not strict,
        "event_rules": event_payload["rules"],
        "target_protocol": protocol.get("scoring"),
        "matching": "existing match_detections global greedy one-to-one IoU>=0.5; target assess_frame best-IoU with ambiguity",
        "definitions": {
            "recovery": "existing assess_mode: consecutive 0.2 seconds correct; no crossing manifest gaps",
            "same_new_id": "ID at stable recovery start versus last_visible_frame ID",
            "wrong_id": "any visible post-event wrong output; tracker pre-bound ID assigned to another GT",
            "no_output": "no valid output in any visible post-event frame",
            "overlap": "wrong_id can coexist with subsequent recovery; rates are not a partition",
            "absent": "null: no confirmed absence denominator in the existing protocol",
        },
    }
    for entry in [*dependencies, *pred_records, config_record]:
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Input changed during evaluation: {entry['path']}")
    for name, value in (("metrics.json", metrics), ("evaluation_manifest.json", manifest)):
        (output / name).write_text(
            json.dumps(portable_record(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return metrics
