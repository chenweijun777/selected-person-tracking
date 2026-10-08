from __future__ import annotations

from dataclasses import dataclass

from a1_tracker.detectors.base import Detector
from a1_tracker.profiling import StageTimer
from a1_tracker.target import TargetManager
from a1_tracker.trackers.base import Tracker
from a1_tracker.types import FramePacket, TargetResult, Track


@dataclass(slots=True)
class PipelineOutput:
    tracks: list[Track]
    target: TargetResult


class TrackingPipeline:
    def __init__(self, detector: Detector, tracker: Tracker, target: TargetManager) -> None:
        self.detector = detector
        self.tracker = tracker
        self.target = target
        self.timer = StageTimer()

    def process(self, frame: FramePacket) -> PipelineOutput:
        with self.timer.measure("detect"):
            detections = self.detector.detect(frame)
        with self.timer.measure("track"):
            tracks = self.tracker.update(detections, frame.image)
        with self.timer.measure("target"):
            target = self.target.update(tracks, frame.media_timestamp)
        return PipelineOutput(tracks=tracks, target=target)
