from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from a1_tracker.geometry import iou
from a1_tracker.types import BBox, Detection, Track


@dataclass(slots=True)
class _TrackMemory:
    bbox: BBox
    score: float
    age: int
    missed: int


class GreedyIoUTracker:
    """可测试的轻量基线；正式实验应替换为 ByteTrack/BoT-SORT 适配器。"""

    def __init__(self, match_iou: float = 0.3, max_missed_frames: int = 30) -> None:
        self.match_iou = match_iou
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        self._tracks: dict[int, _TrackMemory] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(self, detections: list[Detection], image: Any | None = None) -> list[Track]:
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        candidates = sorted(
            (
                (iou(memory.bbox, detection.bbox_xyxy), track_id, detection_id)
                for track_id, memory in self._tracks.items()
                for detection_id, detection in enumerate(detections)
            ),
            reverse=True,
        )
        for overlap, track_id, detection_id in candidates:
            if overlap < self.match_iou:
                break
            if track_id in unmatched_tracks and detection_id in unmatched_detections:
                matches.append((track_id, detection_id))
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(detection_id)

        output: list[Track] = []
        for track_id, detection_id in matches:
            detection = detections[detection_id]
            memory = self._tracks[track_id]
            memory.bbox = detection.bbox_xyxy
            memory.score = detection.score
            memory.age += 1
            memory.missed = 0
            output.append(Track(track_id, memory.bbox, True, memory.score, memory.age))

        for track_id in unmatched_tracks:
            memory = self._tracks[track_id]
            memory.age += 1
            memory.missed += 1
            if memory.missed > self.max_missed_frames:
                del self._tracks[track_id]

        for detection_id in sorted(unmatched_detections):
            detection = detections[detection_id]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _TrackMemory(detection.bbox_xyxy, detection.score, 1, 0)
            output.append(Track(track_id, detection.bbox_xyxy, True, detection.score, 1))

        return sorted(output, key=lambda track: track.tracker_id)
