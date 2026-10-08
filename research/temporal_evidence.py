"""Unadopted research variant; production entry points never import this file.

Collected independently from the original experiment. All extension bodies
needed here are included, so no earlier research variant is required.
"""

from dataclasses import dataclass, replace
from math import isfinite
from a1_tracker.geometry import iou
from a1_tracker.target.a2 import A2Config, A2Target
from a1_tracker.types import TargetState


@dataclass(frozen=True)
class A2TemporalConfig(A2Config):
    temporal_hold_enabled: bool = False
    joint_recovery_enabled: bool = False
    temporal_hold_floor: float = 0.6
    temporal_hold_max_observations: int = 2
    temporal_hold_max_seconds: float = 0.15
    temporal_chain_max_seconds: float = 0.75
    joint_similarity_floor: float = 0.65
    joint_margin: float = 0.1
    joint_motion_deviation: float = 0.5
    joint_min_score: float = 0.5
    joint_max_overlap: float = 0.3
    joint_max_lost_seconds: float = 1.5
    joint_confirm_observations: int = 5
    joint_confirm_seconds: float = 0.3

    def __post_init__(self):
        super().__post_init__()
        if not (
            0 <= self.temporal_hold_floor < self.min_recovery_similarity
            and 0 <= self.joint_similarity_floor < self.min_recovery_similarity
        ):
            raise ValueError("Weak evidence floors must be below the strong threshold")
        for name in ("joint_margin", "joint_min_score", "joint_max_overlap"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "temporal_hold_max_seconds",
            "temporal_chain_max_seconds",
            "joint_motion_deviation",
            "joint_max_lost_seconds",
            "joint_confirm_seconds",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.temporal_hold_max_observations < 1 or self.joint_confirm_observations < 3:
            raise ValueError("Invalid temporal observation counts")


class TemporalEvidenceTarget(A2Target):
    def __init__(self, encoder, config=None):
        super().__init__(encoder, config or A2TemporalConfig())

    def reset(self):
        super().reset()
        self._clear_temporal()

    def _clear_temporal(self):
        self.hold = None
        self.joint = None

    def select_track(self, track, image, timestamp):
        if not super().select_track(track, image, timestamp):
            return False
        self._clear_temporal()
        return True

    def _strict_support(self, chosen, item, margin, tracks, timestamp):
        cfg = self.config
        return (
            self.frame_motion_valid
            and (not self.gmc_chain_broken)
            and self.motion_debug.get("motion_gate_active", False)
            and (len(self.history) >= cfg.min_motion_observations)
            and (item.get("motion_deviation", float("inf")) <= cfg.joint_motion_deviation)
            and (margin is None or margin >= cfg.joint_margin)
            and (chosen.score >= cfg.joint_min_score)
            and (timestamp - self.last_reliable_timestamp <= cfg.joint_max_lost_seconds)
            and (
                not any(
                    (
                        iou(chosen.bbox_xyxy, t.bbox_xyxy) > cfg.joint_max_overlap
                        for t in tracks
                        if t.observed and t.tracker_id != chosen.tracker_id
                    )
                )
            )
        )

    def _advance_hold(self, chosen, timestamp, strong, weak):
        cfg = self.config
        pending = self.hold
        consecutive = (
            pending is not None
            and pending["id"] == chosen.tracker_id
            and (0 <= timestamp - pending["last"] <= cfg.max_confirmation_gap_seconds)
            and (timestamp - pending["first"] <= cfg.temporal_chain_max_seconds)
        )
        if not consecutive:
            self.hold = None
            if not strong:
                return False
            self.hold = pending = {
                "id": chosen.tracker_id,
                "first": timestamp,
                "last": timestamp,
                "observations": 1,
                "seconds": 0.0,
                "was_strong": True,
                "weak_observations": 0,
                "weak_first": None,
            }
        elif timestamp != pending["last"]:
            if strong:
                if pending["was_strong"]:
                    pending["seconds"] += timestamp - pending["last"]
                pending["observations"] += 1
                pending["weak_observations"], pending["weak_first"] = (0, None)
                pending["was_strong"] = True
            elif weak:
                if pending["weak_first"] is None:
                    pending["weak_first"] = pending["last"]
                pending["weak_observations"] += 1
                if (
                    pending["weak_observations"] > cfg.temporal_hold_max_observations
                    or timestamp - pending["weak_first"] > cfg.temporal_hold_max_seconds
                ):
                    self.hold = None
                    return False
                pending["was_strong"] = False
            else:
                self.hold = None
                return False
            pending["last"] = timestamp
        self.last_debug.update(
            temporal_strong_observations=pending["observations"],
            temporal_strong_seconds=pending["seconds"],
            temporal_weak_observations=pending["weak_observations"],
        )
        return (
            strong
            and pending["observations"] >= cfg.confirm_observations
            and (pending["seconds"] + 1e-09 >= cfg.confirm_seconds)
        )

    def _advance_joint(self, chosen, timestamp, supported):
        cfg = self.config
        if not supported:
            self.joint = None
            return False
        pending = self.joint
        if (
            pending is None
            or pending["id"] != chosen.tracker_id
            or (not 0 <= timestamp - pending["last"] <= cfg.max_confirmation_gap_seconds)
        ):
            self.joint = pending = {
                "id": chosen.tracker_id,
                "first": timestamp,
                "last": timestamp,
                "observations": 1,
            }
        elif timestamp != pending["last"]:
            pending["observations"] += 1
            pending["last"] = timestamp
        elapsed = timestamp - pending["first"]
        self.last_debug.update(joint_observations=pending["observations"], joint_seconds=elapsed)
        return (
            pending["observations"] >= cfg.joint_confirm_observations
            and elapsed + 1e-09 >= cfg.joint_confirm_seconds
        )

    def _accept(self, chosen, timestamp, route):
        self.tracker_id = chosen.tracker_id
        self.state = TargetState.TRACKING
        self.last_reliable_timestamp = timestamp
        self.bound_mismatch_count = 0
        self.recoveries += 1
        self._clear_candidate()
        self._clear_temporal()
        self.last_debug["temporal_status"] = route + "_accepted"
        return self._result(True, chosen.bbox_xyxy, timestamp, f"candidate_{route}_confirmed")

    def _update_policy(self, tracks, timestamp, image):
        result = super()._update_policy(tracks, timestamp, image)
        self.last_debug.update(
            temporal_status="inactive",
            temporal_candidate_id=None,
            temporal_similarity=None,
            temporal_margin=None,
            temporal_strong_observations=0,
            temporal_strong_seconds=0.0,
            temporal_weak_observations=0,
            joint_observations=0,
            joint_seconds=0.0,
        )
        cfg = self.config
        if (
            result.valid
            or not (cfg.temporal_hold_enabled or cfg.joint_recovery_enabled)
            or self.state in (TargetState.UNSELECTED, TargetState.NEED_RESELECT)
            or (not self.frame_motion_valid)
            or self.gmc_chain_broken
            or ("before_motion" not in self.motion_debug)
        ):
            self._clear_temporal()
            return result
        ranked = []
        for t in tracks:
            item = self.motion_candidates_by_id.get(t.tracker_id, {})
            if not self._eligible(t) or "appearance_similarity" not in item:
                continue
            ranked.append((item["ranking_score"], -t.tracker_id, t, item))
        if not ranked:
            self._clear_temporal()
            self.last_debug["temporal_status"] = "no_scored_motion_candidate"
            return result
        ranked.sort(key=lambda row: row[:2], reverse=True)
        ranking, _, chosen, item = ranked[0]
        similarity = item["appearance_similarity"]
        margin = ranking - ranked[1][0] if len(ranked) > 1 else None
        strict = self._strict_support(chosen, item, margin, tracks, timestamp)
        strong = (
            similarity >= cfg.min_recovery_similarity
            and (margin is None or margin >= cfg.min_recovery_margin)
            and item.get("confirmation_eligible", False)
        )
        weak = strict and similarity >= cfg.temporal_hold_floor
        self.last_debug.update(
            temporal_candidate_id=chosen.tracker_id,
            temporal_similarity=similarity,
            temporal_margin=margin,
        )
        if cfg.temporal_hold_enabled and self._advance_hold(chosen, timestamp, strong, weak):
            return self._accept(chosen, timestamp, "temporal")
        if cfg.joint_recovery_enabled and self._advance_joint(
            chosen, timestamp, strict and similarity >= cfg.joint_similarity_floor
        ):
            return self._accept(chosen, timestamp, "joint")
        if self.hold or self.joint:
            self.state = TargetState.VERIFYING
            self.last_debug["temporal_status"] = (
                "hold_paused" if self.hold and (not strong) else "confirming"
            )
            return self._result(False, None, timestamp, "candidate_temporal_pending")
        self.last_debug["temporal_status"] = "insufficient_evidence"
        return result

    def _remember(self, result, timestamp):
        if result.reason in ("candidate_temporal_confirmed", "candidate_joint_confirmed"):
            result = replace(result, reason="candidate_confirmed")
        super()._remember(result, timestamp)
