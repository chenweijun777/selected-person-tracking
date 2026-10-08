"""A0 and A1 target policies on a shared tracker output."""

from __future__ import annotations

from typing import Any, Literal, Protocol
from uuid import uuid4

import numpy as np

from a1_tracker.geometry import contains_point
from a1_tracker.types import BBox, TargetResult, TargetState, Track


class AppearanceEncoder(Protocol):
    def encode(self, image: Any, tracks: list[Track]) -> list[np.ndarray | None]: ...


class TargetBaseline:
    """A0 locks the initial ID; A1 immediately rebinds by anchor cosine similarity.

    A1 deliberately has no threshold, memory update or confirmation. It is a
    naive comparator, so an incorrect recovery is possible and must be measured.
    """

    def __init__(self, strategy: Literal["A0", "A1"], encoder: AppearanceEncoder | None = None):
        if strategy not in {"A0", "A1"}:
            raise ValueError(f"unknown target baseline: {strategy}")
        if strategy == "A1" and encoder is None:
            raise ValueError("A1 requires an appearance encoder")
        self.strategy = strategy
        self.encoder = encoder
        self.reset()

    def reset(self) -> None:
        self.session_target_id: str | None = None
        self.tracker_id: int | None = None
        self.state = TargetState.UNSELECTED
        self.anchor: np.ndarray | None = None

    def select_by_point(self, tracks: list[Track], point: tuple[float, float], image: Any) -> bool:
        candidates = [t for t in tracks if t.observed and contains_point(t.bbox_xyxy, point)]
        if len(candidates) != 1:
            return False
        return self.select_track(candidates[0], image)

    def select_track(self, track: Track, image: Any) -> bool:
        if not track.observed:
            return False
        anchor = None
        if self.strategy == "A1":
            assert self.encoder is not None
            anchor = self.encoder.encode(image, [track])[0]
            if anchor is None:
                return False
        self.session_target_id = uuid4().hex[:12]
        self.tracker_id = track.tracker_id
        self.anchor = anchor
        self.state = TargetState.TRACKING
        return True

    def update(self, tracks: list[Track], timestamp: float, image: Any = None) -> TargetResult:
        if self.state == TargetState.UNSELECTED:
            return self._result(False, None, timestamp, "waiting_for_selection")
        bound = next((t for t in tracks if t.observed and t.tracker_id == self.tracker_id), None)
        if bound is not None:
            self.state = TargetState.TRACKING
            return self._result(True, bound.bbox_xyxy, timestamp, "bound_track_observed")
        self.state = TargetState.LOST
        if self.strategy == "A0":
            return self._result(False, None, timestamp, "bound_track_missing")

        if image is None:
            raise ValueError("A1 recovery requires the current frame image")
        candidates = [t for t in tracks if t.observed]
        if not candidates:
            return self._result(False, None, timestamp, "no_recovery_candidate")
        assert self.encoder is not None and self.anchor is not None
        features = self.encoder.encode(image, candidates)
        if len(features) != len(candidates):
            raise RuntimeError("appearance encoder returned a different number of features")
        ranked = [
            (float(np.dot(self.anchor, feature)), t)
            for t, feature in zip(candidates, features, strict=True)
            if feature is not None
        ]
        if not ranked:
            return self._result(False, None, timestamp, "no_valid_appearance")
        # Deterministic tie handling; the anchor never changes after selection.
        _, winner = max(ranked, key=lambda item: (item[0], -item[1].tracker_id))
        self.tracker_id = winner.tracker_id
        self.state = TargetState.TRACKING
        return self._result(True, winner.bbox_xyxy, timestamp, "appearance_rebound")

    def _result(
        self, valid: bool, bbox: BBox | None, timestamp: float, reason: str
    ) -> TargetResult:
        return TargetResult(
            self.session_target_id, self.tracker_id, self.state, valid, bbox, timestamp, reason
        )


class BaselinePair:
    """Select and run both strategies on identical T2 tracks and input frames."""

    def __init__(self, encoder: AppearanceEncoder):
        self.a0 = TargetBaseline("A0")
        self.a1 = TargetBaseline("A1", encoder)

    def select_track(self, track: Track, image: Any) -> bool:
        # Encode first. Failure must not leave A0 selected while A1 is unselected.
        if not self.a1.select_track(track, image):
            return False
        self.a0.select_track(track, image)
        self.a0.session_target_id = self.a1.session_target_id
        return True

    def select_by_point(self, tracks: list[Track], point: tuple[float, float], image: Any) -> bool:
        candidates = [t for t in tracks if t.observed and contains_point(t.bbox_xyxy, point)]
        if len(candidates) != 1:
            return False
        return self.select_track(candidates[0], image)

    def update(
        self, tracks: list[Track], timestamp: float, image: Any
    ) -> tuple[TargetResult, TargetResult]:
        return self.a0.update(tracks, timestamp, image), self.a1.update(tracks, timestamp, image)
