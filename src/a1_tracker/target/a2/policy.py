"""Standalone selected-person recovery policy.

Adopted motion, continuity, evidence and template stages live here. Earlier
method bodies are named by responsibility to preserve exact fallback order;
there is no inheritance from, or import of, a historical A2 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, log, sqrt
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from a1_tracker.geometry import iou
from a1_tracker.types import TargetResult, TargetState, Track

from .geometry import corners, geometry
from .memory import AppearanceBank, BankConfig


class AppearanceEncoder(Protocol):
    def encode(self, image: Any, tracks: list[Track]) -> list[np.ndarray | None]: ...


@dataclass(frozen=True)
class A2Config:
    verify_bound_appearance: bool = True
    min_observation_score: float = 0.25
    min_box_width_px: float = 24
    min_box_height_px: float = 48
    min_tracking_similarity: float = 0.75
    bound_mismatch_observations: int = 3
    min_recovery_similarity: float = 0.6
    min_recovery_margin: float = 0.05
    min_memory_update_similarity: float = 0.9
    min_memory_update_score: float = 0.5
    max_memory_update_box_iou: float = 0.3
    memory_alpha: float = 0.1
    search_seconds: float = 6.0
    confirm_seconds: float = 0.15
    confirm_observations: int = 3
    max_confirmation_gap_seconds: float = 0.2
    motion_enabled: bool = True
    history_seconds: float = 0.5
    min_motion_observations: int = 3
    extrapolation_seconds: float = 0.5
    position_radius_x: float = 0.75
    position_radius_y: float = 0.5
    position_growth_x: float = 1.5
    position_growth_y: float = 0.75
    min_position_radius_px: float = 20.0
    max_speed_widths_per_second: float = 8.0
    max_speed_heights_per_second: float = 4.0
    height_ratio: float = 1.5
    height_log_growth: float = 0.35
    aspect_ratio: float = 1.7
    aspect_log_growth: float = 0.2
    max_log_scale_rate: float = 0.5
    motion_penalty_weight: float = 0.05
    max_motion_penalty: float = 0.1
    confirmation_deviation: float = 1.5
    hard_motion_deviation: float = 2.0
    max_pause_observations: int = 2
    max_pause_seconds: float = 0.2
    same_id_enabled: bool = True
    same_id_window_seconds: float = 1.0
    same_id_grace_seconds: float = 0.5
    same_id_grace_similarity: float = 0.1
    same_id_motion_deviation: float = 1.0
    initial_min_box_width_px: float | None = None
    initial_min_box_height_px: float | None = None
    fusion_appearance_enabled: bool = False
    fusion_geometry_enabled: bool = False
    fusion_small_box_enabled: bool = False
    fusion_strong_similarity: float = 0.8
    fusion_aspect_deviation: float = 2.0
    fusion_geometry_similarity: float = 0.05
    fusion_tight_deviation: float = 0.5
    fusion_min_box_width_px: float = 20.0
    fusion_max_overlap: float = 0.3
    fusion_confirm_observations: int = 3
    fusion_confirm_seconds: float = 0.1
    bank_enabled: bool = True
    bank_capacity: int = 5

    def _validate_appearance(self) -> None:
        for name in (
            "min_observation_score",
            "min_tracking_similarity",
            "min_recovery_similarity",
            "min_recovery_margin",
            "min_memory_update_similarity",
            "min_memory_update_score",
            "max_memory_update_box_iou",
            "memory_alpha",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            min(
                self.min_box_width_px,
                self.min_box_height_px,
                self.search_seconds,
                self.max_confirmation_gap_seconds,
            )
            <= 0
        ):
            raise ValueError("box and time limits must be positive")
        if (
            self.confirm_seconds < 0
            or self.confirm_observations < 1
            or self.bound_mismatch_observations < 1
        ):
            raise ValueError("confirmation limits are invalid")

    def _validate_motion(self):
        self._validate_appearance()
        names = (
            "history_seconds",
            "extrapolation_seconds",
            "position_radius_x",
            "position_radius_y",
            "position_growth_x",
            "position_growth_y",
            "min_position_radius_px",
            "max_speed_widths_per_second",
            "max_speed_heights_per_second",
            "height_log_growth",
            "aspect_log_growth",
            "max_log_scale_rate",
        )
        if any(not isfinite(getattr(self, n)) or getattr(self, n) <= 0 for n in names):
            raise ValueError("Motion bounds must be finite and positive")
        if (
            self.min_motion_observations < 2
            or not isfinite(self.height_ratio)
            or (not isfinite(self.aspect_ratio))
            or (self.height_ratio <= 1)
            or (self.aspect_ratio <= 1)
        ):
            raise ValueError("Invalid motion observation or scale limits")

    def _validate_confirmation(self):
        self._validate_motion()
        values = (
            self.motion_penalty_weight,
            self.max_motion_penalty,
            self.confirmation_deviation,
            self.hard_motion_deviation,
            self.max_pause_seconds,
        )
        if not all(isfinite(v) for v in values):
            raise ValueError("Soft motion limits must be finite")
        if not 0 <= self.motion_penalty_weight <= self.max_motion_penalty <= 1:
            raise ValueError("Invalid motion penalty bounds")
        if not 1 <= self.confirmation_deviation < self.hard_motion_deviation:
            raise ValueError("Confirmation region must be inside hard exclusion region")
        if (
            not isinstance(self.max_pause_observations, int)
            or self.max_pause_observations < 0
            or self.max_pause_seconds <= 0
        ):
            raise ValueError("Invalid confirmation pause limits")

    def _validate_continuity(self):
        self._validate_confirmation()
        values = (
            self.same_id_window_seconds,
            self.same_id_grace_seconds,
            self.same_id_grace_similarity,
            self.same_id_motion_deviation,
        )
        if not all(isfinite(v) for v in values):
            raise ValueError("Same-ID limits must be finite")
        if not 0 < self.same_id_grace_seconds <= self.same_id_window_seconds <= self.search_seconds:
            raise ValueError("Same-ID windows must fit inside the recovery deadline")
        if not 0 <= self.same_id_grace_similarity <= self.min_tracking_similarity:
            raise ValueError("Same-ID grace threshold must not exceed tracking threshold")
        if not 0 < self.same_id_motion_deviation <= self.confirmation_deviation:
            raise ValueError("Same-ID geometry must fit inside the confirmation region")

    def _validate_selection(self):
        self._validate_continuity()
        for name, regular in (
            ("initial_min_box_width_px", self.min_box_width_px),
            ("initial_min_box_height_px", self.min_box_height_px),
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or not 0 < value <= regular):
                raise ValueError("Initial size bounds must be positive and within regular bounds")

    def _validate_evidence(self):
        self._validate_selection()
        values = (
            self.fusion_strong_similarity,
            self.fusion_aspect_deviation,
            self.fusion_geometry_similarity,
            self.fusion_tight_deviation,
            self.fusion_min_box_width_px,
            self.fusion_max_overlap,
            self.fusion_confirm_seconds,
        )
        if not all(isfinite(v) for v in values):
            raise ValueError("Fusion limits must be finite")
        if not self.min_tracking_similarity <= self.fusion_strong_similarity <= 1:
            raise ValueError("Strong appearance must support normal tracking")
        if not 0 <= self.fusion_geometry_similarity <= self.same_id_grace_similarity:
            raise ValueError("Weak appearance floor must fit inside grace")
        if not 0 < self.fusion_tight_deviation <= self.same_id_motion_deviation:
            raise ValueError("Corroborating geometry must be tighter than normal geometry")
        if (
            not self.same_id_motion_deviation
            <= self.fusion_aspect_deviation
            <= self.hard_motion_deviation
        ):
            raise ValueError("Shape exception must fit inside hard motion bounds")
        if not 0 < self.fusion_min_box_width_px <= self.min_box_width_px:
            raise ValueError("Small-box width must be positive and bounded")
        if not 0 <= self.fusion_max_overlap <= 1:
            raise ValueError("Overlap limit must be in [0, 1]")
        if (
            type(self.fusion_confirm_observations) is not int
            or self.fusion_confirm_observations < 2
            or self.fusion_confirm_seconds <= 0
        ):
            raise ValueError("Fusion requires repeated observations and positive time")

    def __post_init__(self):
        self._validate_evidence()
        if self.bank_capacity < 1:
            raise ValueError("bank_capacity must be positive")


class A2Target:
    def _appearance_init(self, encoder: AppearanceEncoder, config: A2Config | None = None):
        self.encoder = encoder
        self.config = config or A2Config()
        self.reset()

    def _appearance_reset(self) -> None:
        self.session_target_id: str | None = None
        self.tracker_id: int | None = None
        self.state = TargetState.UNSELECTED
        self.anchor: np.ndarray | None = None
        self.memory: np.ndarray | None = None
        self.last_reliable_timestamp: float | None = None
        self.candidate_id: int | None = None
        self.candidate_first_timestamp: float | None = None
        self.candidate_last_timestamp: float | None = None
        self.candidate_observations = 0
        self.memory_updates = 0
        self.recoveries = 0
        self.bound_mismatch_count = 0
        self.last_debug: dict[str, float | int | None] = {}

    def _appearance_select_track(self, track: Track, image: Any, timestamp: float) -> bool:
        if not self._eligible(track):
            return False
        feature = self.encoder.encode(image, [track])[0]
        normalized = self._normalize(feature)
        if normalized is None:
            return False
        self.session_target_id = uuid4().hex[:12]
        self.tracker_id = track.tracker_id
        self.anchor = normalized
        self.memory = normalized.copy()
        self.last_reliable_timestamp = timestamp
        self.state = TargetState.TRACKING
        self.bound_mismatch_count = 0
        self._clear_candidate()
        return True

    def _appearance_eligible(self, track: Track) -> bool:
        box = track.bbox_xyxy
        return (
            track.observed
            and track.score >= self.config.min_observation_score
            and (box[2] - box[0] >= self.config.min_box_width_px)
            and (box[3] - box[1] >= self.config.min_box_height_px)
        )

    @staticmethod
    def _normalize(feature: np.ndarray | None) -> np.ndarray | None:
        if feature is None:
            return None
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if np.isfinite(norm) and norm > 0 else None

    def _appearance_similarity(self, feature: np.ndarray) -> float:
        assert self.anchor is not None and self.memory is not None
        return max(float(np.dot(self.anchor, feature)), float(np.dot(self.memory, feature)))

    def _maybe_update_memory(self, feature: np.ndarray, bound: Track, tracks: list[Track]) -> None:
        assert self.anchor is not None and self.memory is not None
        if bound.score < self.config.min_memory_update_score:
            return
        if (
            min(float(np.dot(self.anchor, feature)), float(np.dot(self.memory, feature)))
            < self.config.min_memory_update_similarity
        ):
            return
        if any(
            iou(bound.bbox_xyxy, other.bbox_xyxy) > self.config.max_memory_update_box_iou
            for other in tracks
            if other.tracker_id != bound.tracker_id and other.observed
        ):
            return
        updated = self._normalize(
            (1 - self.config.memory_alpha) * self.memory + self.config.memory_alpha * feature
        )
        if updated is not None:
            self.memory = updated
            self.memory_updates += 1

    def _appearance_clear_candidate(self) -> None:
        self.candidate_id = None
        self.candidate_first_timestamp = None
        self.candidate_last_timestamp = None
        self.candidate_observations = 0

    def _result(
        self, valid: bool, box: tuple[float, ...] | None, timestamp: float, reason: str
    ) -> TargetResult:
        return TargetResult(
            self.session_target_id, self.tracker_id, self.state, valid, box, timestamp, reason
        )

    def _motion_init(self, encoder, config=None):
        self._appearance_init(encoder, config or A2Config())

    def _motion_reset(self):
        self._appearance_reset()
        self.history = []
        self.motion_timestamp = None
        self.frame_motion_valid = False
        self.motion_debug = {}
        self.candidate_diagnostics = []

    def _motion_select_track(self, track, image, timestamp):
        if not self._appearance_select_track(track, image, timestamp):
            return False
        self.history = [(timestamp, corners(track.bbox_xyxy))]
        self.motion_timestamp = timestamp
        return True

    def _motion_advance_motion(self, timestamp, warp, valid):
        if not isfinite(timestamp):
            raise ValueError("Non-finite motion timestamp")
        if self.motion_timestamp is not None and timestamp < self.motion_timestamp:
            raise ValueError("Motion timestamps must be nondecreasing")
        matrix = np.asarray(warp, dtype=float) if warp is not None else None
        self.frame_motion_valid = bool(
            valid
            and matrix is not None
            and (matrix.shape == (2, 3))
            and np.all(np.isfinite(matrix))
            and (np.linalg.det(matrix[:, :2]) > 0)
        )
        if timestamp != self.motion_timestamp:
            if self.frame_motion_valid:
                self.history = [
                    (t, points @ matrix[:, :2].T + matrix[:, 2]) for t, points in self.history
                ]
            else:
                self.history = []
        self.motion_timestamp = timestamp

    def _motion_remember(self, result, timestamp):
        if result.reason not in ("bound_verified", "bound_id_continuity", "candidate_confirmed"):
            return
        if result.reason == "candidate_confirmed":
            self.history = []
        self.history = [
            (t, p)
            for t, p in self.history
            if timestamp - t <= self.config.history_seconds and t < timestamp
        ]
        self.history.append((timestamp, corners(result.bbox_xyxy)))

    def _prediction(self, timestamp):
        t, points = self.history[-1]
        center, width, height = geometry(points)
        velocity, scale_rate = (np.zeros(2), np.zeros(2))
        if len(self.history) >= self.config.min_motion_observations:
            estimates, rates = ([], [])
            for (t0, p0), (t1, p1) in zip(self.history, self.history[1:]):
                if t1 <= t0:
                    continue
                c0, w0, h0 = geometry(p0)
                c1, w1, h1 = geometry(p1)
                estimates.append((c1 - c0) / (t1 - t0))
                rates.append(np.log([w1 / w0, h1 / h0]) / (t1 - t0))
            if estimates:
                limit = np.array(
                    [
                        width * self.config.max_speed_widths_per_second,
                        height * self.config.max_speed_heights_per_second,
                    ]
                )
                velocity = np.clip(np.median(estimates, axis=0), -limit, limit)
                scale_rate = np.clip(
                    np.median(rates, axis=0),
                    -self.config.max_log_scale_rate,
                    self.config.max_log_scale_rate,
                )
        dt = max(0.0, timestamp - t)
        horizon = min(dt, self.config.extrapolation_seconds)
        size = np.array([width, height]) * np.exp(scale_rate * horizon)
        return (center + velocity * horizon, size[0], size[1], dt)

    def _motion_motion_candidates(self, candidates, timestamp):
        active = self.config.motion_enabled and self.frame_motion_valid and bool(self.history)
        self.motion_debug = {
            "motion_gate_active": active,
            "before_motion": len(candidates),
            "after_motion": len(candidates),
            "motion_rejected": 0,
        }
        self.candidate_diagnostics = []
        if not active:
            return candidates
        center, width, height, dt = self._prediction(timestamp)
        rx = max(
            self.config.min_position_radius_px,
            width * (self.config.position_radius_x + self.config.position_growth_x * dt),
        )
        ry = max(
            self.config.min_position_radius_px,
            height * (self.config.position_radius_y + self.config.position_growth_y * dt),
        )
        kept = []
        self.motion_debug.update(
            predicted_x=float(center[0]),
            predicted_y=float(center[1]),
            predicted_width=float(width),
            predicted_height=float(height),
            lost_seconds=dt,
            radius_x=rx,
            radius_y=ry,
        )
        for track in candidates:
            c, w, h = geometry(corners(track.bbox_xyxy))
            distance = float(np.sum(((c - center) / [rx, ry]) ** 2))
            height_change = abs(log(h / height))
            aspect_change = abs(log(w / h / (width / height)))
            reason = (
                "position"
                if distance > 1
                else "height"
                if height_change
                > log(self.config.height_ratio) + self.config.height_log_growth * dt
                else "aspect"
                if aspect_change
                > log(self.config.aspect_ratio) + self.config.aspect_log_growth * dt
                else "pass"
            )
            self.candidate_diagnostics.append(
                {
                    "candidate_id": track.tracker_id,
                    "gate_reason": reason,
                    "position_distance": distance,
                    "height_log_change": height_change,
                    "aspect_log_change": aspect_change,
                }
            )
            if reason == "pass":
                kept.append(track)
        self.motion_debug.update(
            after_motion=len(kept), motion_rejected=len(candidates) - len(kept)
        )
        return kept

    def _motion_update(self, tracks, timestamp, image, *, motion_warp=None, motion_valid=False):
        self._advance_motion(timestamp, motion_warp, motion_valid)
        self.motion_debug = {"motion_gate_active": False}
        self.candidate_diagnostics = []
        result = self._update_policy(tracks, timestamp, image)
        self.last_debug.update(self.motion_debug)
        self._remember(result, timestamp)
        return result

    def _confirmation_init(self, encoder, config=None):
        self._motion_init(encoder, config or A2Config())

    def _confirmation_reset(self):
        self._motion_reset()
        self.motion_candidates_by_id = {}
        self._clear_candidate()

    def _confirmation_clear_candidate(self):
        self._appearance_clear_candidate()
        self.candidate_paused_observations = 0
        self.candidate_pause_start = None
        self.candidate_elapsed_seconds = 0.0
        self.candidate_paused_seconds = 0.0
        self.candidate_last_was_paused = False

    def _confirmation_motion_candidates(self, candidates, timestamp):
        self._motion_motion_candidates(candidates, timestamp)
        self.motion_candidates_by_id = {}
        if not self.motion_debug["motion_gate_active"]:
            return candidates
        dt = self.motion_debug["lost_seconds"]
        height_limit = log(self.config.height_ratio) + self.config.height_log_growth * dt
        aspect_limit = log(self.config.aspect_ratio) + self.config.aspect_log_growth * dt
        kept = []
        for track, item in zip(candidates, self.candidate_diagnostics, strict=True):
            components = {
                "position": sqrt(item["position_distance"]),
                "height": item["height_log_change"] / height_limit,
                "aspect": item["aspect_log_change"] / aspect_limit,
            }
            component = max(components, key=components.get)
            deviation = components[component]
            accepted = deviation <= self.config.hard_motion_deviation
            item.update(
                gate_reason="pass" if accepted else f"hard_{component}",
                motion_deviation=deviation,
                motion_penalty=min(
                    self.config.max_motion_penalty, self.config.motion_penalty_weight * deviation**2
                ),
                confirmation_eligible=deviation <= self.config.confirmation_deviation,
            )
            self.motion_candidates_by_id[track.tracker_id] = item
            if accepted:
                kept.append(track)
        self.motion_debug.update(
            after_motion=len(kept), motion_rejected=len(candidates) - len(kept)
        )
        return kept

    def _confirmation_confirm(self, chosen, timestamp, eligible):
        consecutive = (
            chosen.tracker_id == self.candidate_id
            and self.candidate_last_timestamp is not None
            and (
                0
                <= timestamp - self.candidate_last_timestamp
                <= self.config.max_confirmation_gap_seconds
            )
        )
        if not consecutive:
            self._clear_candidate()
            if not eligible:
                self.state = TargetState.LOST
                return self._result(False, None, timestamp, "motion_confirmation_not_ready")
            self.candidate_id = chosen.tracker_id
            self.candidate_first_timestamp = timestamp
            self.candidate_observations = 1
        elif not eligible:
            if self.candidate_pause_start is None:
                self.candidate_pause_start = self.candidate_last_timestamp
            self.candidate_paused_observations += 1
            if (
                self.candidate_paused_observations > self.config.max_pause_observations
                or timestamp - self.candidate_pause_start > self.config.max_pause_seconds
            ):
                self._clear_candidate()
                self.state = TargetState.LOST
                return self._result(False, None, timestamp, "motion_pause_exhausted")
            self.candidate_paused_seconds += timestamp - self.candidate_last_timestamp
            self.candidate_last_timestamp = timestamp
            self.candidate_last_was_paused = True
            self.state = TargetState.VERIFYING
            self._confirmation_debug(paused=True)
            return self._result(False, None, timestamp, "candidate_motion_paused")
        elif (
            self.candidate_pause_start is not None
            and timestamp - self.candidate_pause_start > self.config.max_pause_seconds
        ):
            self._clear_candidate()
            self.candidate_id = chosen.tracker_id
            self.candidate_first_timestamp = timestamp
            self.candidate_observations = 1
        else:
            self.candidate_observations += 1
            if self.candidate_last_was_paused:
                self.candidate_paused_seconds += timestamp - self.candidate_last_timestamp
            self.candidate_pause_start = None
            self.candidate_paused_observations = 0
            self.candidate_last_was_paused = False
        self.candidate_last_timestamp = timestamp
        assert self.candidate_first_timestamp is not None
        self.candidate_elapsed_seconds = max(
            0.0, timestamp - self.candidate_first_timestamp - self.candidate_paused_seconds
        )
        self._confirmation_debug(paused=False)
        if (
            self.candidate_observations < self.config.confirm_observations
            or self.candidate_elapsed_seconds < self.config.confirm_seconds
        ):
            self.state = TargetState.VERIFYING
            return self._result(False, None, timestamp, "candidate_confirming")
        self.tracker_id = chosen.tracker_id
        self.bound_mismatch_count = 0
        self.last_reliable_timestamp = timestamp
        self.state = TargetState.TRACKING
        self.recoveries += 1
        self._clear_candidate()
        return self._result(True, chosen.bbox_xyxy, timestamp, "candidate_confirmed")

    def _confirmation_debug(self, paused):
        self.last_debug.update(
            confirmation_paused=paused,
            paused_observations=self.candidate_paused_observations,
            confirmation_observations=self.candidate_observations,
            confirmation_elapsed_seconds=self.candidate_elapsed_seconds,
        )

    def _confirmation_update_policy(
        self, tracks: list[Track], timestamp: float, image: Any
    ) -> TargetResult:
        self.last_debug = {
            "bound_similarity": None,
            "best_similarity": None,
            "candidate_margin": None,
            "candidate_id": None,
        }
        if self.state == TargetState.UNSELECTED:
            return self._result(False, None, timestamp, "waiting_for_selection")
        if self.state == TargetState.NEED_RESELECT:
            return self._result(False, None, timestamp, "reselect_required")
        assert self.anchor is not None and self.memory is not None
        assert self.last_reliable_timestamp is not None
        bound = next((t for t in tracks if t.tracker_id == self.tracker_id and t.observed), None)
        if self.state == TargetState.TRACKING and bound is not None:
            feature = (
                self._normalize(self.encoder.encode(image, [bound])[0])
                if self._eligible(bound)
                else None
            )
            if not self.config.verify_bound_appearance:
                if feature is not None:
                    self.last_debug["bound_similarity"] = self._similarity(feature)
                    self._maybe_update_memory(feature, bound, tracks)
                self.last_reliable_timestamp = timestamp
                self.bound_mismatch_count = 0
                self._clear_candidate()
                return self._result(True, bound.bbox_xyxy, timestamp, "bound_id_continuity")
            if feature is not None:
                similarity = self._similarity(feature)
                self.last_debug["bound_similarity"] = similarity
                if similarity >= self.config.min_tracking_similarity:
                    self.last_reliable_timestamp = timestamp
                    self.bound_mismatch_count = 0
                    self._clear_candidate()
                    self._maybe_update_memory(feature, bound, tracks)
                    return self._result(True, bound.bbox_xyxy, timestamp, "bound_verified")
            self.bound_mismatch_count += 1
            if self.bound_mismatch_count < self.config.bound_mismatch_observations:
                return self._result(True, bound.bbox_xyxy, timestamp, "bound_unverified_grace")
        else:
            self.bound_mismatch_count = 0
        if timestamp - self.last_reliable_timestamp > self.config.search_seconds:
            self.state = TargetState.NEED_RESELECT
            self._clear_candidate()
            return self._result(False, None, timestamp, "recovery_timeout")
        candidates = [t for t in tracks if self._eligible(t)]
        if not candidates:
            self._clear_candidate()
            self.state = TargetState.LOST
            return self._result(False, None, timestamp, "no_recovery_candidate")
        candidates = self._motion_candidates(candidates, timestamp)
        if not candidates:
            self._clear_candidate()
            self.state = TargetState.LOST
            return self._result(False, None, timestamp, "motion_rejected_all")
        features = self.encoder.encode(image, candidates)
        ranked = []
        for track, raw_feature in zip(candidates, features, strict=True):
            feature = self._normalize(raw_feature)
            if feature is None:
                continue
            similarity = self._similarity(feature)
            item = self.motion_candidates_by_id.get(track.tracker_id, {})
            penalty = item.get("motion_penalty", 0.0)
            ranking_score = similarity - penalty
            item.update(appearance_similarity=similarity, ranking_score=ranking_score)
            ranked.append((ranking_score, -track.tracker_id, similarity, track))
        if not ranked:
            self._clear_candidate()
            self.state = TargetState.LOST
            return self._result(False, None, timestamp, "no_valid_appearance")
        ranked.sort(reverse=True)
        qualified = [r for r in ranked if r[2] >= self.config.min_recovery_similarity]
        if not qualified:
            self.last_debug.update(
                best_similarity=ranked[0][2], candidate_id=ranked[0][3].tracker_id
            )
            self._clear_candidate()
            self.state = TargetState.LOST
            return self._result(False, None, timestamp, "appearance_below_threshold")
        ranking_score, _, similarity, chosen = qualified[0]
        alternatives = [r[0] for r in ranked if r[3].tracker_id != chosen.tracker_id]
        margin = ranking_score - max(alternatives) if alternatives else None
        item = self.motion_candidates_by_id.get(chosen.tracker_id, {})
        self.last_debug.update(
            best_similarity=similarity,
            best_ranking_score=ranking_score,
            candidate_margin=margin,
            candidate_id=chosen.tracker_id,
            motion_penalty=item.get("motion_penalty", 0.0),
        )
        if margin is not None and margin < self.config.min_recovery_margin:
            self._clear_candidate()
            self.state = TargetState.LOST
            return self._result(False, None, timestamp, "appearance_ambiguous")
        return self._confirm(chosen, timestamp, item.get("confirmation_eligible", True))

    def _continuity_init(self, encoder, config=None):
        self._confirmation_init(encoder, config or A2Config())

    def _continuity_reset(self):
        self._confirmation_reset()
        self.last_continuity_timestamp = None

    def _continuity_select_track(self, track, image, timestamp):
        selected = self._motion_select_track(track, image, timestamp)
        if selected:
            self.last_continuity_timestamp = timestamp
        return selected

    def _continuity_deviation(self, bound, timestamp):
        if not (self.config.motion_enabled and self.frame_motion_valid and self.history):
            return None
        center, width, height, dt = self._prediction(timestamp)
        rx = max(
            self.config.min_position_radius_px,
            width * (self.config.position_radius_x + self.config.position_growth_x * dt),
        )
        ry = max(
            self.config.min_position_radius_px,
            height * (self.config.position_radius_y + self.config.position_growth_y * dt),
        )
        c, w, h = geometry(corners(bound.bbox_xyxy))
        return max(
            sqrt(float(np.sum(((c - center) / [rx, ry]) ** 2))),
            abs(log(h / height))
            / (log(self.config.height_ratio) + self.config.height_log_growth * dt),
            abs(log(w / h / (width / height)))
            / (log(self.config.aspect_ratio) + self.config.aspect_log_growth * dt),
        )

    def _continuity_update_policy(self, tracks, timestamp, image):
        debug = {"same_id_accepted": False}
        if (
            self.config.same_id_enabled
            and self._same_id_resume_allowed()
            and (self.state in (TargetState.TRACKING, TargetState.LOST, TargetState.VERIFYING))
            and (self.last_continuity_timestamp is not None)
        ):
            bound = next(
                (t for t in tracks if t.tracker_id == self.tracker_id and t.observed), None
            )
            if bound is not None and self._eligible(bound):
                feature = self._normalize(self.encoder.encode(image, [bound])[0])
                similarity = self._similarity(feature) if feature is not None else None
                needs_continuity = self.state != TargetState.TRACKING or (
                    similarity is not None and similarity < self.config.min_tracking_similarity
                )
                if needs_continuity and similarity is not None:
                    gap = timestamp - self.last_continuity_timestamp
                    appearance_age = timestamp - self.last_reliable_timestamp
                    deviation = self._continuity_deviation(bound, timestamp)
                    debug.update(
                        same_id_gap_seconds=gap,
                        same_id_appearance_age=appearance_age,
                        same_id_deviation=deviation,
                        same_id_similarity=similarity,
                    )
                    geometry_ok = (
                        deviation is not None and deviation <= self.config.same_id_motion_deviation
                    )
                    in_window = (
                        0 <= gap <= self.config.same_id_window_seconds
                        and 0 <= appearance_age <= self.config.search_seconds
                    )
                    verified = similarity >= self.config.min_tracking_similarity
                    grace = (
                        similarity >= self.config.same_id_grace_similarity
                        and appearance_age <= self.config.same_id_grace_seconds
                    )
                    if geometry_ok and in_window and (verified or grace):
                        was_lost = self.state != TargetState.TRACKING
                        self.state = TargetState.TRACKING
                        self._clear_candidate()
                        if verified:
                            self.last_reliable_timestamp = timestamp
                            self.bound_mismatch_count = 0
                            self._maybe_update_memory(feature, bound, tracks)
                            reason = "same_id_resumed"
                        else:
                            self.bound_mismatch_count += 1
                            reason = "same_id_appearance_grace"
                        self.recoveries += int(was_lost)
                        self.last_debug = dict(
                            bound_similarity=similarity,
                            best_similarity=None,
                            candidate_margin=None,
                            candidate_id=None,
                            **debug,
                        )
                        self.last_debug["same_id_accepted"] = True
                        return self._result(True, bound.bbox_xyxy, timestamp, reason)
        result = self._confirmation_update_policy(tracks, timestamp, image)
        self.last_debug.update(debug)
        return result

    def _continuity_remember(self, result, timestamp):
        if result.reason in ("same_id_resumed", "same_id_appearance_grace"):
            self._motion_remember(replace(result, reason="bound_verified"), timestamp)
            self.last_continuity_timestamp = timestamp
        else:
            self._motion_remember(result, timestamp)
            if result.reason in ("bound_verified", "bound_id_continuity", "candidate_confirmed"):
                self.last_continuity_timestamp = timestamp

    def _reseed_init(self, encoder, config=None):
        self._continuity_init(encoder, config or A2Config())

    def _reseed_reset(self):
        self._continuity_reset()
        self.gmc_chain_broken = False
        self._clear_seed()

    def _clear_seed(self):
        self.seed_history = []
        self.seed_id = None
        self.seed_first_timestamp = None
        self.seed_observations = 0

    def _reseed_select_track(self, track, image, timestamp):
        selected = self._continuity_select_track(track, image, timestamp)
        if selected:
            self.gmc_chain_broken = False
            self._clear_seed()
        return selected

    def _advance_motion(self, timestamp, warp, valid):
        previous_timestamp = self.motion_timestamp
        had_history = bool(self.history)
        self._motion_advance_motion(timestamp, warp, valid)
        if timestamp == previous_timestamp:
            return
        if not self.frame_motion_valid:
            self.gmc_chain_broken |= had_history
            self._clear_seed()
        elif self.seed_history:
            matrix = np.asarray(warp, dtype=float)
            self.seed_history = [
                (t, p @ matrix[:, :2].T + matrix[:, 2]) for t, p in self.seed_history
            ]

    def _seed_deviation(self, bound, timestamp):
        t, points = self.seed_history[-1]
        center, width, height = geometry(points)
        c, w, h = geometry(corners(bound.bbox_xyxy))
        dt = timestamp - t
        rx = max(
            self.config.min_position_radius_px,
            width * (self.config.position_radius_x + self.config.position_growth_x * dt),
        )
        ry = max(
            self.config.min_position_radius_px,
            height * (self.config.position_radius_y + self.config.position_growth_y * dt),
        )
        return max(
            sqrt(float(np.sum(((c - center) / [rx, ry]) ** 2))),
            abs(log(h / height))
            / (log(self.config.height_ratio) + self.config.height_log_growth * dt),
            abs(log(w / h / (width / height)))
            / (log(self.config.aspect_ratio) + self.config.aspect_log_growth * dt),
        )

    def _seed_debug(self, reason, timestamp, deviation=None):
        self.last_debug.update(
            gmc_chain_broken=self.gmc_chain_broken,
            gmc_seed_reason=reason,
            gmc_seed_observations=self.seed_observations,
            gmc_seed_elapsed_seconds=timestamp - self.seed_first_timestamp
            if self.seed_first_timestamp is not None
            else 0.0,
            gmc_seed_deviation=deviation,
        )

    def _reseed_update_policy(self, tracks, timestamp, image):
        result = self._continuity_update_policy(tracks, timestamp, image)
        if (
            result.valid
            or self.history
            or (not self.gmc_chain_broken)
            or (not self.config.same_id_enabled)
            or (not self._same_id_resume_allowed())
            or (not self.config.motion_enabled)
            or (self.state in (TargetState.UNSELECTED, TargetState.NEED_RESELECT))
        ):
            self._clear_seed()
            self._seed_debug("not_needed", timestamp)
            return result
        gap = timestamp - self.last_continuity_timestamp
        appearance_age = timestamp - self.last_reliable_timestamp
        if not (
            0 <= gap <= self.config.same_id_window_seconds
            and 0 <= appearance_age <= self.config.search_seconds
        ):
            self._clear_seed()
            self._seed_debug("outside_original_window", timestamp)
            return result
        bound = next((t for t in tracks if t.tracker_id == self.tracker_id and t.observed), None)
        if bound is None or not self._eligible(bound):
            self._clear_seed()
            self._seed_debug("missing_or_ineligible_same_id", timestamp)
            return result
        feature = self._normalize(self.encoder.encode(image, [bound])[0])
        similarity = self._similarity(feature) if feature is not None else None
        if similarity is None or similarity < self.config.min_tracking_similarity:
            self._clear_seed()
            self._seed_debug("same_id_appearance_not_verified", timestamp)
            return result
        deviation = None
        consecutive = (
            self.seed_history
            and self.seed_id == bound.tracker_id
            and self.frame_motion_valid
            and (
                0
                <= timestamp - self.seed_history[-1][0]
                <= self.config.max_confirmation_gap_seconds
            )
        )
        if consecutive and timestamp == self.seed_history[-1][0]:
            self._seed_debug("duplicate_timestamp", timestamp)
            return result
        if consecutive:
            deviation = self._seed_deviation(bound, timestamp)
            consecutive = deviation <= self.config.same_id_motion_deviation
        if not consecutive:
            self._clear_seed()
            self.seed_id = bound.tracker_id
            self.seed_first_timestamp = timestamp
            self.seed_observations = 1
            reason = "fresh_seed" if deviation is None else "geometry_restart"
        else:
            self.seed_observations += 1
            reason = "geometry_confirming"
        self.seed_history = [
            (t, p) for t, p in self.seed_history if timestamp - t <= self.config.history_seconds
        ]
        self.seed_history.append((timestamp, corners(bound.bbox_xyxy)))
        self._seed_debug(reason, timestamp, deviation)
        ready = (
            self.frame_motion_valid
            and consecutive
            and (
                self.seed_observations
                >= max(self.config.confirm_observations, self.config.min_motion_observations)
            )
            and (len(self.seed_history) >= self.config.min_motion_observations)
            and (timestamp - self.seed_first_timestamp >= self.config.confirm_seconds)
        )
        if not ready:
            return result
        was_lost = self.state != TargetState.TRACKING
        self.history = [(t, p.copy()) for t, p in self.seed_history]
        self.gmc_chain_broken = False
        self.state = TargetState.TRACKING
        self.last_reliable_timestamp = timestamp
        self.bound_mismatch_count = 0
        self._clear_candidate()
        self._maybe_update_memory(feature, bound, tracks)
        self.recoveries += int(was_lost)
        self._seed_debug("promoted", timestamp, deviation)
        self.last_debug.update(bound_similarity=similarity, same_id_accepted=True)
        self._clear_seed()
        return self._result(True, bound.bbox_xyxy, timestamp, "gmc_same_id_reseeded")

    def _reseed_remember(self, result, timestamp):
        if result.reason == "gmc_same_id_reseeded":
            self._continuity_remember(replace(result, reason="same_id_resumed"), timestamp)
        else:
            self._continuity_remember(result, timestamp)
        if result.reason in (
            "bound_verified",
            "bound_id_continuity",
            "candidate_confirmed",
            "same_id_resumed",
            "same_id_appearance_grace",
            "gmc_same_id_reseeded",
        ):
            self.gmc_chain_broken = False
            self._clear_seed()

    def _selection_init(self, encoder, config=None):
        self._reseed_init(encoder, config or A2Config())

    def _selection_reset(self):
        self._reseed_reset()
        self._initializing = False

    def _selection_select_track(self, track, image, timestamp):
        self._initializing = True
        try:
            return self._reseed_select_track(track, image, timestamp)
        finally:
            self._initializing = False

    def _eligible(self, track):
        if not self._initializing:
            return self._appearance_eligible(track)
        width = self.config.initial_min_box_width_px
        height = self.config.initial_min_box_height_px
        box = track.bbox_xyxy
        return (
            track.observed
            and track.score >= self.config.min_observation_score
            and (box[2] - box[0] >= (self.config.min_box_width_px if width is None else width))
            and (box[3] - box[1] >= (self.config.min_box_height_px if height is None else height))
        )

    def _evidence_init(self, encoder, config=None):
        self._selection_init(encoder, config or A2Config())

    def _evidence_reset(self):
        self._selection_reset()
        self._clear_fusion()

    def _clear_fusion(self):
        self.fusion_key = None
        self.fusion_first_timestamp = None
        self.fusion_last_timestamp = None
        self.fusion_observations = 0

    def _evidence_select_track(self, track, image, timestamp):
        self._clear_fusion()
        return self._selection_select_track(track, image, timestamp)

    def _components(self, bound, timestamp):
        center, width, height, dt = self._prediction(timestamp)
        rx = max(
            self.config.min_position_radius_px,
            width * (self.config.position_radius_x + self.config.position_growth_x * dt),
        )
        ry = max(
            self.config.min_position_radius_px,
            height * (self.config.position_radius_y + self.config.position_growth_y * dt),
        )
        c, w, h = geometry(corners(bound.bbox_xyxy))
        return (
            sqrt(float(np.sum(((c - center) / [rx, ry]) ** 2))),
            abs(log(h / height))
            / (log(self.config.height_ratio) + self.config.height_log_growth * dt),
            abs(log(w / h / (width / height)))
            / (log(self.config.aspect_ratio) + self.config.aspect_log_growth * dt),
        )

    def _evidence_update_policy(self, tracks, timestamp, image):
        result = self._reseed_update_policy(tracks, timestamp, image)
        debug = {
            "fusion_route": "",
            "fusion_status": "not_needed",
            "fusion_similarity": None,
            "fusion_position": None,
            "fusion_height": None,
            "fusion_aspect": None,
            "fusion_observations": 0,
            "fusion_elapsed_seconds": 0.0,
        }
        self.last_debug.update(debug)
        cfg = self.config
        enabled = (
            cfg.fusion_appearance_enabled
            or cfg.fusion_geometry_enabled
            or cfg.fusion_small_box_enabled
        )
        if (
            result.valid
            or not enabled
            or (not cfg.same_id_enabled)
            or (not self._same_id_resume_allowed())
            or (not cfg.motion_enabled)
            or (not self.frame_motion_valid)
            or (not self.history)
            or self.gmc_chain_broken
            or (self.state in (TargetState.UNSELECTED, TargetState.NEED_RESELECT))
        ):
            self._clear_fusion()
            return result
        gap = timestamp - self.last_continuity_timestamp
        age = timestamp - self.last_reliable_timestamp
        if not (0 <= gap <= cfg.same_id_window_seconds and 0 <= age <= cfg.search_seconds):
            self._clear_fusion()
            self.last_debug["fusion_status"] = "outside_window"
            return result
        bound = next((t for t in tracks if t.observed and t.tracker_id == self.tracker_id), None)
        if (
            bound is None
            or bound.score < cfg.min_observation_score
            or bound.bbox_xyxy[2] - bound.bbox_xyxy[0] < cfg.fusion_min_box_width_px
            or (bound.bbox_xyxy[3] - bound.bbox_xyxy[1] < cfg.min_box_height_px)
        ):
            self._clear_fusion()
            self.last_debug["fusion_status"] = "missing_or_low_quality"
            return result
        if any(
            iou(bound.bbox_xyxy, t.bbox_xyxy) > cfg.fusion_max_overlap
            for t in tracks
            if t.observed and t.tracker_id != bound.tracker_id
        ):
            self._clear_fusion()
            self.last_debug["fusion_status"] = "crowded"
            return result
        feature = self._normalize(self.encoder.encode(image, [bound])[0])
        if feature is None:
            self._clear_fusion()
            self.last_debug["fusion_status"] = "no_appearance"
            return result
        similarity = self._similarity(feature)
        position, height, aspect = self._components(bound, timestamp)
        self.last_debug.update(
            fusion_similarity=similarity,
            fusion_position=position,
            fusion_height=height,
            fusion_aspect=aspect,
        )
        regular = self._eligible(bound)
        route = None
        if (
            regular
            and cfg.fusion_appearance_enabled
            and (similarity >= cfg.fusion_strong_similarity)
            and (max(position, height) <= cfg.same_id_motion_deviation)
            and (aspect <= cfg.fusion_aspect_deviation)
        ):
            route = "appearance"
        elif (
            regular
            and cfg.fusion_geometry_enabled
            and (similarity >= cfg.fusion_geometry_similarity)
            and (max(position, height, aspect) <= cfg.fusion_tight_deviation)
            and (age <= cfg.same_id_grace_seconds)
        ):
            route = "geometry"
        elif (
            not regular
            and cfg.fusion_small_box_enabled
            and (similarity >= cfg.min_tracking_similarity)
            and (max(position, height, aspect) <= cfg.fusion_tight_deviation)
            and (age <= cfg.same_id_grace_seconds)
        ):
            route = "small_box"
        if route is None:
            self._clear_fusion()
            self.last_debug["fusion_status"] = "insufficient_evidence"
            return result
        key = (bound.tracker_id, route)
        consecutive = (
            key == self.fusion_key
            and self.fusion_last_timestamp is not None
            and (0 <= timestamp - self.fusion_last_timestamp <= cfg.max_confirmation_gap_seconds)
        )
        if not consecutive:
            self._clear_fusion()
            self.fusion_key = key
            self.fusion_first_timestamp = timestamp
        if timestamp != self.fusion_last_timestamp:
            self.fusion_observations += 1
            self.fusion_last_timestamp = timestamp
        elapsed = timestamp - self.fusion_first_timestamp
        self.last_debug.update(
            fusion_route=route,
            fusion_status="confirming",
            fusion_observations=self.fusion_observations,
            fusion_elapsed_seconds=elapsed,
        )
        if (
            self.fusion_observations < cfg.fusion_confirm_observations
            or elapsed < cfg.fusion_confirm_seconds
        ):
            return result
        self.state = TargetState.TRACKING
        self.recoveries += 1
        self._clear_candidate()
        self._clear_seed()
        if route == "appearance":
            self.last_reliable_timestamp = timestamp
            self.bound_mismatch_count = 0
            self._maybe_update_memory(feature, bound, tracks)
        self.last_debug.update(
            fusion_status="accepted", bound_similarity=similarity, same_id_accepted=True
        )
        self._clear_fusion()
        return self._result(True, bound.bbox_xyxy, timestamp, f"same_id_evidence_{route}")

    def _remember(self, result, timestamp):
        if result.reason.startswith("same_id_evidence_"):
            self._reseed_remember(replace(result, reason="same_id_appearance_grace"), timestamp)
        else:
            self._reseed_remember(result, timestamp)

    def __init__(self, encoder, config=None):
        self._evidence_init(encoder, config or A2Config())

    def reset(self):
        self._evidence_reset()
        self.bank = AppearanceBank(BankConfig(capacity=self.config.bank_capacity))
        self._bank_scoring = False
        self._bank_frame = 0
        self._identity_remaps = ()
        self._identity_frame_id = None
        self._alias_candidate_pending = False
        self._suppress_alias_same_id = False

    def select_track(self, track, image, timestamp):
        if not self._evidence_select_track(track, image, timestamp):
            return False
        self.bank = AppearanceBank(BankConfig(capacity=self.config.bank_capacity))
        self._bank_frame = 0
        self._bank_scoring = False
        return True

    def _motion_candidates(self, candidates, timestamp):
        kept = self._confirmation_motion_candidates(candidates, timestamp)
        self._bank_scoring = self.config.bank_enabled
        return kept

    def _similarity(self, feature):
        baseline = self._appearance_similarity(feature)
        if self._bank_scoring:
            return AppearanceBank.score(feature, self.bank.templates, baseline)
        return baseline

    def _clear_candidate(self):
        self._bank_scoring = False
        self._alias_candidate_pending = False
        self._confirmation_clear_candidate()

    def _same_id_resume_allowed(self):
        return not self._suppress_alias_same_id

    def _consume_identity_remaps(self, tracks, timestamp, image):
        remaps, self._identity_remaps = (self._identity_remaps, ())
        details = []
        if self.state in (TargetState.UNSELECTED, TargetState.NEED_RESELECT):
            return details
        young_counts = {r.young_id: sum(x.young_id == r.young_id for x in remaps) for r in remaps}
        old_counts = {r.old_id: sum(x.old_id == r.old_id for x in remaps) for r in remaps}
        for event in remaps:
            if event.young_id not in (self.tracker_id, self.candidate_id, self.seed_id):
                continue
            record = {"young_id": event.young_id, "old_id": event.old_id, "accepted": False}
            details.append(record)
            if (
                event.frame_id != self._identity_frame_id
                or young_counts[event.young_id] != 1
                or old_counts[event.old_id] != 1
                or (event.old_id in young_counts)
                or (event.young_id in old_counts)
            ):
                record["reason"] = "stale_or_conflicting_event"
                continue
            current = [t for t in tracks if t.tracker_id == event.old_id and t.observed]
            if (
                len(current) != 1
                or any(t.tracker_id == event.young_id for t in tracks)
                or (not np.allclose(current[0].bbox_xyxy, event.bbox_xyxy, rtol=0, atol=0.0021))
                or (not self._eligible(current[0]))
            ):
                record["reason"] = "observation_not_stitched"
                continue
            candidate_only = self.tracker_id != event.young_id
            self._bank_scoring = candidate_only
            try:
                feature = self._normalize(self.encoder.encode(image, current)[0])
                similarity = self._similarity(feature) if feature is not None else None
            finally:
                self._bank_scoring = False
            deviation = self._continuity_deviation(current[0], timestamp)
            threshold = (
                self.config.min_recovery_similarity
                if candidate_only
                else self.config.min_tracking_similarity
            )
            motion_limit = (
                self.config.confirmation_deviation
                if candidate_only
                else self.config.same_id_motion_deviation
            )
            record.update(similarity=similarity, motion_deviation=deviation)
            if (
                similarity is None
                or similarity < threshold
                or deviation is None
                or (not np.isfinite(deviation))
                or (deviation > motion_limit)
            ):
                record["reason"] = "A2_handoff_gate_rejected"
                continue
            if self.tracker_id == event.young_id:
                self.tracker_id = event.old_id
            if self.candidate_id == event.young_id:
                self.candidate_id = event.old_id
                self._alias_candidate_pending = True
            if self.seed_id == event.young_id:
                self.seed_id = event.old_id
            if self.fusion_key is not None and self.fusion_key[-1] == event.young_id:
                self.fusion_key = (*self.fusion_key[:-1], event.old_id)
            if self.bank.last_id == event.young_id:
                self.bank.last_id = event.old_id
            self.bank.templates = [
                (v, f, event.old_id if i == event.young_id else i)
                for v, f, i in self.bank.templates
            ]
            record.update(accepted=True, reason="identity_renamed")
        return details

    def _confirm(self, chosen, timestamp, eligible):
        self._bank_scoring = False
        return self._confirmation_confirm(chosen, timestamp, eligible)

    def _update_policy(self, tracks, timestamp, image):
        self._bank_scoring = False
        identity_debug = self._consume_identity_remaps(tracks, timestamp, image)
        self._suppress_alias_same_id = self._alias_candidate_pending
        try:
            return self._evidence_update_policy(tracks, timestamp, image)
        finally:
            self._bank_scoring = False
            self._suppress_alias_same_id = False
            self.last_debug["identity_handoffs"] = identity_debug

    def update(
        self,
        tracks,
        timestamp,
        image,
        *,
        motion_warp=None,
        motion_valid=False,
        identity_remaps=(),
        identity_frame_id=None,
    ):
        self._identity_remaps = tuple(identity_remaps)
        self._identity_frame_id = identity_frame_id
        previous_id = self.tracker_id
        result = self._motion_update(
            tracks, timestamp, image, motion_warp=motion_warp, motion_valid=motion_valid
        )
        for handoff in self.last_debug.get("identity_handoffs", []):
            if handoff["accepted"] and previous_id == handoff["young_id"]:
                previous_id = handoff["old_id"]
        self._bank_frame += 1
        added = False
        if self.config.bank_enabled:
            bound = next((t for t in tracks if t.tracker_id == previous_id), None)
            verified = (
                result.valid
                and result.tracker_id == previous_id
                and (result.reason in ("bound_verified", "same_id_resumed"))
            )
            feature = (
                self.encoder.encode(image, [bound])[0] if verified and bound is not None else None
            )
            normalized = self._normalize(feature)
            similarity = self._similarity(normalized) if normalized is not None else None
            added = self.bank.observe(
                result=result,
                bound=bound,
                tracks=tracks,
                feature=feature,
                similarity=similarity,
                timestamp=timestamp,
                frame=self._bank_frame,
                motion_valid=self.frame_motion_valid,
            )
        self.last_debug.update(bank_size=len(self.bank.templates), bank_updated=added)
        return result
