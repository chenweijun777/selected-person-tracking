from __future__ import annotations

from typing import Any, Protocol

from a1_tracker.types import Detection, Track


class Tracker(Protocol):
    def update(self, detections: list[Detection], image: Any | None = None) -> list[Track]: ...

    def reset(self) -> None: ...
