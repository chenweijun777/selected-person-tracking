"""Replay A0/A1/A2 and targeted ablations under one frozen current protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from a1_tracker.evaluation import evaluate_run
from a1_tracker.evaluation._common import portable_record, project_path
from a1_tracker.evaluation.replay import ReplaySequence, replay

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "validation", "test"], required=True)
    parser.add_argument("--inputs", type=Path, default=ROOT / "reproduction_inputs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/finetuned/target.json")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["A0", "A1", "A2", "NoBank", "NoEMA", "NoHandoff", "NoMotion"],
        default=["A0", "A1", "A2", "NoBank", "NoEMA"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.inputs = project_path(args.inputs)
    args.output = project_path(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    pp = project_path(args.protocol) if args.protocol else ROOT / "protocols/current" / f"{args.split}.json"
    protocol = json.loads(pp.read_text(encoding="utf-8"))
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    config.pop("status", None)
    cases = [
        c
        for c in protocol["cases"]
        if c["selection_frame"] is not None
        and protocol["sequences"][c["sequence"]]["split"]
        in (args.split, "historical" if args.split == "validation" else args.split)
    ]
    sequences = sorted({c["sequence"] for c in cases})
    for seq in sequences:
        replay(
            ReplaySequence(args.inputs / seq),
            [c for c in cases if c["sequence"] == seq],
            config,
            args.modes,
            args.output / seq,
        )
        print("Replayed", seq, flush=True)
    results = []
    for mode in args.modes:
        prediction = args.output / f"{mode}.csv"
        with prediction.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["sequence", "gt_id", "frame", "tracker_id", "valid"]
            )
            writer.writeheader()
            for seq in sequences:
                with (args.output / seq / f"{mode}.csv").open(encoding="utf-8", newline="") as src:
                    writer.writerows(csv.DictReader(src))
        scope = ROOT / "protocols/current" / ("test" if args.split == "test" else "devval")
        metadata = args.output / f"{mode}.json"
        metadata.write_text(
            json.dumps(
                portable_record({
                    "algorithm": mode,
                    "effective_config": config,
                    "evaluation": {
                        "protocol": str(pp),
                        "events": str(scope / "gt_events.json"),
                        "visibility": str(scope / "visibility_intervals.csv"),
                        "episode_manifest": str(scope / "episode_manifest.json"),
                    },
                }),
                indent=2,
            ),
            encoding="utf-8",
        )
        diagnostic = evaluate_run(
            f"{mode}_{args.split}",
            prediction,
            "target_manager",
            metadata,
            args.output / "evaluations" / mode,
            args.split,
            strict=False,
        )
        formal = evaluate_run(
            f"{mode}_{args.split}_formal",
            prediction,
            "target_manager",
            metadata,
            args.output / "formal" / mode,
            args.split,
            strict=True,
        )
        c = diagnostic["target"]["counts"]
        rec = diagnostic["recovery"]
        results.append(
            dict(
                split=args.split,
                mode=mode,
                C=c.get("C", 0),
                W=c.get("W", 0),
                R=c.get("R", 0),
                U=c.get("U", 0),
                A=c.get("A", 0),
                coverage=diagnostic["target"]["coverage_C"],
                person_frames=diagnostic["target"]["n_person_frames"],
                targets=diagnostic["target"]["n_targets"],
                common_recovered=round(rec["recovery_success_rate"] * rec["n_events"] / 100)
                if rec["n_events"]
                else 0,
                common_events=rec["n_events"],
                formal_events=formal["recovery"]["n_events"],
                formal_rate=formal["recovery"]["recovery_success_rate"],
            )
        )
    with (args.output / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
