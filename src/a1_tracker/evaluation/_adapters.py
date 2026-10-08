"""Prediction adapters; GT identifiers remain strictly offline."""

from . import protocol as protocol_io

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from ._common import ROOT, boolean, box, csv_rows, read_json, record, resolve_asset


def tracker_adapter(path, sequences, lengths):
    files, predictions = {}, {}
    for seq in sequences:
        source = path / f"{seq}.txt" if path.is_dir() else path
        if path.is_file() and (len(sequences) != 1 or path.stem != seq):
            raise ValueError("Supply a directory containing every sequence in the requested split")
        files[seq] = source
        frames = defaultdict(dict)
        with source.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 10:
                    raise ValueError(f"Expected 10 MOT columns: {source}")
                frame, pred_id = int(row[0]), int(row[1])
                x, y, w, h, score = map(float, row[2:7])
                if not 1 <= frame <= lengths[seq] or pred_id < 0 or pred_id in frames[frame]:
                    raise ValueError(f"Invalid/duplicate frame-ID: {seq}/{frame}/{pred_id}")
                if not math.isfinite(score):
                    raise ValueError("Non-finite score")
                frames[frame][pred_id] = {
                    "sequence": seq,
                    "frame": frame,
                    "pred_id": pred_id,
                    "bbox_xyxy": box((x, y, x + w, y + h)),
                    "score": score,
                    "valid": True,
                }
        predictions[seq] = frames
    return files, predictions


def target_adapter(path, protocol, sequences, options, dependencies):
    cases = [
        c
        for c in protocol["cases"]
        if c["sequence"] in sequences and c["selection_frame"] is not None
    ]
    mapping_path = Path(
        options.get(
            "episode_manifest",
            ROOT
            / "protocols/frozen"
            / ("test" if "MOT17-11-SDP" in sequences else "devval")
            / "episode_manifest.json",
        )
    ).resolve()
    dependencies.append(record(mapping_path))
    episode_map = {
        (e["sequence"], e["episode_id"]): int(e["target_gt_id"])
        for e in read_json(mapping_path)["episodes"]
    }
    rows = (
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if path.suffix == ".jsonl"
        else csv_rows(path)
    )
    if not rows:
        raise ValueError("Empty target predictions")
    old = "gt_id" in rows[0] and "bbox_xyxy" not in rows[0]
    tracks = {}
    if old:
        for seq in sequences:
            source = resolve_asset(protocol["sequences"][seq]["t2_tracks"])
            dependencies.append(record(source))
            tracks[seq] = protocol_io.load_tracks(source)
    expected = protocol_io.expected_keys(cases)
    result = {}
    for row in rows:
        seq = row["sequence"]
        if seq not in protocol["sequences"]:
            raise ValueError(f"Unknown sequence {seq}")
        if seq not in sequences:
            continue
        person = int(row["gt_id"]) if old else episode_map.get((seq, row["episode_id"]))
        key = (seq, person, int(row.get("frame", row.get("frame_id"))))
        if key not in expected or key in result:
            raise ValueError(f"Unexpected/duplicate target row: {key}")
        valid = boolean(row["valid"])
        tid = int(row["tracker_id"]) if row["tracker_id"] not in ("", None) else None
        bbox = tracks[seq].get(key[2], {}).get(tid) if old else row.get("bbox_xyxy")
        if valid and tid is None:
            raise ValueError(f"Valid output requires tracker_id: {key}")
        if not old and not {"session_target_id", "state"} <= row.keys():
            raise ValueError("TargetResult envelope needs session_target_id and state")
        state = row.get("state", "TRACKING" if valid else "LOST")
        if state not in ("UNSELECTED", "TRACKING", "LOST", "VERIFYING", "NEED_RESELECT"):
            raise ValueError(f"Invalid TargetState: {state}")
        if state == "TRACKING" and not valid:
            raise ValueError("TRACKING must be valid, as required by TargetResult")
        result[key] = {
            "tracker_id": tid,
            "valid": valid,
            "bbox_xyxy": box(bbox) if valid else None,
            "state": state,
            "reason": row.get("reason", "saved_prediction"),
            "session_target_id": row.get("session_target_id"),
            "episode_id": row.get("episode_id"),
            "score": row.get("score"),
        }
    if expected != result.keys():
        raise ValueError(
            f"Missing {len(expected - result.keys())} target frames; save rejected frames too"
        )
    for c in cases:
        key = (c["sequence"], c["gt_id"], c["selection_frame"])
        if result[key]["valid"] and result[key]["tracker_id"] != c["initial_tracker_id"]:
            raise ValueError(f"Output violates frozen initialization: {key}")
        if c["initial_tracker_id"] is None and any(
            result[c["sequence"], c["gt_id"], f]["valid"]
            for f in range(c["selection_frame"], c["end_frame"] + 1)
        ):
            raise ValueError(f"Output without initial binding: {key}")
    return cases, result
