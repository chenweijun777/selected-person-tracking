from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

BBox = tuple[float, float, float, float]


class TargetState(str, Enum):
    UNSELECTED = "UNSELECTED"
    TRACKING = "TRACKING"
    LOST = "LOST"
    VERIFYING = "VERIFYING"
    NEED_RESELECT = "NEED_RESELECT"


def validate_bbox(bbox: BBox) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid xyxy bbox: {bbox}")


@dataclass(slots=True)
class FramePacket:
    source_id: str
    frame_id: int
    media_timestamp: float
    ingest_timestamp: float
    image: Any


@dataclass(slots=True, frozen=True)
class Detection:
    bbox_xyxy: BBox
    score: float
    class_id: int = 0

    def __post_init__(self) -> None:
        validate_bbox(self.bbox_xyxy)
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")


@dataclass(slots=True, frozen=True)
class Track:
    tracker_id: int
    bbox_xyxy: BBox
    observed: bool
    score: float
    age: int = 1

    def __post_init__(self) -> None:
        validate_bbox(self.bbox_xyxy)


@dataclass(slots=True, frozen=True)
class TargetResult:
    session_target_id: str | None
    tracker_id: int | None
    state: TargetState
    valid: bool
    bbox_xyxy: BBox | None
    timestamp: float
    reason: str

    def __post_init__(self) -> None:
        if self.valid and self.bbox_xyxy is None:
            raise ValueError("valid result requires bbox")
        if not self.valid and self.state == TargetState.TRACKING:
            raise ValueError("TRACKING result must be valid")
