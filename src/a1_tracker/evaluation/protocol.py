"""Frozen track and session readers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


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
