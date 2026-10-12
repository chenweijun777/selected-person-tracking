"""Frozen track and session readers."""

from __future__ import annotations

import csv
import copy
import hashlib
from collections import defaultdict
from pathlib import Path

from a1_tracker.geometry import iou


def load_tracks(path: Path) -> dict[int, dict[int, tuple[float, ...]]]:
    result = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            frame, track_id = (int(row[0]), int(row[1]))
            x, y, w, h = map(float, row[2:6])
            if track_id in result[frame]:
                raise ValueError(f"Duplicate T2 track ID: {path}:{frame}:{track_id}")
            result[frame][track_id] = (x, y, x + w, y + h)
    return result


def expected_keys(cases: list[dict]) -> set[tuple[str, int, int]]:
    return {
        (case["sequence"], case["gt_id"], frame)
        for case in cases
        for frame in range(case["selection_frame"], case["end_frame"] + 1)
    }


def rebind_protocol(protocol: dict, inputs: Path, root: Path) -> dict:
    """Bind regenerated tracks at the fixed selection frames and reference boxes."""
    root, inputs = root.resolve(), inputs.resolve()
    if not inputs.is_relative_to(root):
        raise ValueError("Reproduction inputs must be inside the project")
    result = copy.deepcopy(protocol)
    old_tracks = {value["t2_tracks"] for value in result["sequences"].values()}
    result["files"] = [item for item in result["files"] if item["path"] not in old_tracks]
    matched = failed = 0
    for sequence, settings in result["sequences"].items():
        folder = inputs / sequence
        tracks = load_tracks(folder / "tracks.txt")
        with (folder / "motion.csv").open(encoding="utf-8-sig", newline="") as stream:
            last_frame = max((int(row["frame"]) for row in csv.DictReader(stream)), default=0)
        cases = [case for case in result["cases"] if case["sequence"] == sequence]
        if last_frame < max(case["end_frame"] for case in cases):
            raise ValueError(f"Incomplete sequence inputs: {sequence}")
        settings["t2_tracks"] = (folder / "tracks.txt").relative_to(root).as_posix()
        for filename in ("tracks.txt", "features.npy", "valid.npy", "motion.csv", "remaps.jsonl"):
            path = folder / filename
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result["files"].append({"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest()})
        for case in cases:
            case["initial_tracker_id"] = None
            if case["selection_frame"] is None:
                continue
            if "selection_bbox_xyxy" not in case:
                raise ValueError("Protocol requires a reference box for each selection frame")
            candidates = [
                (iou(case["selection_bbox_xyxy"], box), -track_id, track_id)
                for track_id, box in tracks.get(case["selection_frame"], {}).items()
            ]
            winner = max(candidates) if candidates else None
            if winner is not None and winner[0] >= 0.5:
                case["initial_tracker_id"] = winner[2]
                matched += 1
            else:
                failed += 1
    result["initialization"] = {
        "rule": "Fixed selection frame and reference box; best IoU >= 0.5; failures retained",
        "matched": matched,
        "failed": failed,
    }
    return result
