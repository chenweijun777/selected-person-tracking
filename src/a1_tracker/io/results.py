from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from a1_tracker.types import TargetResult, Track


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: TextIO | None = None

    def __enter__(self) -> JsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        return self

    def write(self, frame_id: int, result: TargetResult) -> None:
        if self._file is None:
            raise RuntimeError("writer is not open")
        payload = asdict(result)
        payload["state"] = result.state.value
        payload["frame_id"] = frame_id
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def __exit__(self, *_: object) -> None:
        if self._file is not None:
            self._file.close()


class MotWriter:
    """写出 MOTChallenge 十列预测格式。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_frame(self, frame_id: int, tracks: list[Track]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            for track in tracks:
                x1, y1, x2, y2 = track.bbox_xyxy
                width, height = x2 - x1, y2 - y1
                file.write(
                    f"{frame_id},{track.tracker_id},{x1:.3f},{y1:.3f},"
                    f"{width:.3f},{height:.3f},{track.score:.6f},-1,-1,-1\n"
                )
