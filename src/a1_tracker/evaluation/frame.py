"""Selected-target frame classification; GT is offline only."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from a1_tracker.geometry import iou


def load_gt(path: Path):
    by_frame = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if row[6] != "1" or row[7] != "1":
                continue
            frame, target = (int(row[0]), int(row[1]))
            x, y, width, height = map(float, row[2:6])
            by_frame[frame][target] = ((x, y, x + width, y + height), float(row[8]))
    return by_frame


def assess_frame(frame: int, target: int, prediction: dict, gt: dict) -> dict:
    target_box, visibility = gt[frame][target]
    result = {
        "frame": frame,
        "gt_id": target,
        "visibility": visibility,
        "tracker_id": prediction["tracker_id"],
        "state": prediction["state"],
        "reason": prediction["reason"],
        "valid": prediction["valid"],
        "target_iou": 0.0,
        "best_gt_id": None,
        "best_iou": 0.0,
        "ambiguous_overlap": False,
    }
    if not prediction["valid"]:
        result["category"] = "reject"
        return result
    box = tuple(prediction["bbox_xyxy"])
    result["target_iou"] = iou(box, target_box)
    scores = [(iou(box, other_box), other_id) for other_id, (other_box, _) in gt[frame].items()]
    best_iou, best_id = max(scores, key=lambda item: (item[0], -item[1]))
    result["best_gt_id"], result["best_iou"] = (best_id, best_iou)
    result["ambiguous_overlap"] = sum((score >= 0.5 for score, _ in scores)) > 1
    result["category"] = (
        "unmatched" if best_iou < 0.5 else "correct" if best_id == target else "other"
    )
    return result
