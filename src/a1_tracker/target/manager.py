from __future__ import annotations

from uuid import uuid4

from a1_tracker.geometry import contains_point, iou
from a1_tracker.types import BBox, TargetResult, TargetState, Track


class TargetManager:
    """指定目标状态机基线。

    当前用空间重叠验证候选；后续接入 ReID 时保持 public API 不变。
    """

    def __init__(
        self,
        lost_timeout_frames: int = 60,
        verify_observations: int = 3,
        candidate_iou: float = 0.2,
    ) -> None:
        self.lost_timeout_frames = lost_timeout_frames
        self.verify_observations = verify_observations
        self.candidate_iou = candidate_iou
        self.reset()

    def reset(self) -> None:
        self.session_target_id: str | None = None
        self.tracker_id: int | None = None
        self.state = TargetState.UNSELECTED
        self.last_bbox: BBox | None = None
        self.lost_frames = 0
        self.candidate_id: int | None = None
        self.candidate_hits = 0

    def select_by_point(self, tracks: list[Track], point: tuple[float, float]) -> bool:
        candidates = [
            track for track in tracks if track.observed and contains_point(track.bbox_xyxy, point)
        ]
        if len(candidates) != 1:
            return False
        return self.select_track(candidates[0])

    def select_track(self, track: Track) -> bool:
        self.session_target_id = uuid4().hex[:12]
        self.tracker_id = track.tracker_id
        self.state = TargetState.TRACKING
        self.last_bbox = track.bbox_xyxy
        self.lost_frames = 0
        self.candidate_id = None
        self.candidate_hits = 0
        return True

    def update(self, tracks: list[Track], timestamp: float) -> TargetResult:
        if self.state == TargetState.UNSELECTED:
            return self._result(False, None, timestamp, "waiting_for_selection")

        bound = next((track for track in tracks if track.tracker_id == self.tracker_id), None)
        if bound is not None and bound.observed:
            self.last_bbox = bound.bbox_xyxy
            self.state = TargetState.TRACKING
            self.lost_frames = 0
            self.candidate_id = None
            self.candidate_hits = 0
            return self._result(True, bound.bbox_xyxy, timestamp, "bound_track_observed")

        self.lost_frames += 1
        if self.lost_frames > self.lost_timeout_frames:
            self.state = TargetState.NEED_RESELECT
            return self._result(False, None, timestamp, "recovery_timeout")

        candidate = self._best_candidate(tracks)
        if candidate is None:
            self.state = TargetState.LOST
            self.candidate_id = None
            self.candidate_hits = 0
            return self._result(False, None, timestamp, "bound_track_missing")

        if candidate.tracker_id == self.candidate_id:
            self.candidate_hits += 1
        else:
            self.candidate_id = candidate.tracker_id
            self.candidate_hits = 1

        if self.candidate_hits < self.verify_observations:
            self.state = TargetState.VERIFYING
            return self._result(False, None, timestamp, "candidate_accumulating_evidence")

        self.tracker_id = candidate.tracker_id
        self.last_bbox = candidate.bbox_xyxy
        self.state = TargetState.TRACKING
        self.lost_frames = 0
        return self._result(True, candidate.bbox_xyxy, timestamp, "candidate_rebound")

    def _best_candidate(self, tracks: list[Track]) -> Track | None:
        if self.last_bbox is None:
            return None
        ranked = sorted(
            ((iou(self.last_bbox, track.bbox_xyxy), track) for track in tracks if track.observed),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] < self.candidate_iou:
            return None
        return ranked[0][1]

    def _result(
        self, valid: bool, bbox: BBox | None, timestamp: float, reason: str
    ) -> TargetResult:
        return TargetResult(
            session_target_id=self.session_target_id,
            tracker_id=self.tracker_id,
            state=self.state,
            valid=valid,
            bbox_xyxy=bbox,
            timestamp=timestamp,
            reason=reason,
        )
