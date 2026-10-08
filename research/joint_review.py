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
class A2JointReviewConfig(A2Config):
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
    accumulated_recovery_enabled: bool = False
    recovered_review_enabled: bool = False
    accumulated_required_seconds: float = 0.2
    accumulated_max_span_seconds: float = 1.0
    accumulated_pause_floor: float = 0.5
    review_similarity_floor: float = 0.5
    review_competitor_delta: float = 0.05
    review_bad_observations: int = 2
    review_resume_observations: int = 3
    review_resume_seconds: float = 0.2
    joint_review_enabled: bool = False
    review_continuity_deviation: float = 1.0

    def _validate_hold(self):
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

    def _validate_accumulation(self):
        self._validate_hold()
        for name in (
            "accumulated_required_seconds",
            "accumulated_max_span_seconds",
            "review_resume_seconds",
        ):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "accumulated_pause_floor",
            "review_similarity_floor",
            "review_competitor_delta",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.accumulated_pause_floor > self.joint_similarity_floor:
            raise ValueError("Pause floor must not exceed joint evidence floor")
        if self.review_bad_observations < 1 or self.review_resume_observations < 2:
            raise ValueError("Invalid review observation counts")
        if self.accumulated_recovery_enabled and (
            not self.joint_recovery_enabled or self.joint_min_score <= 0
        ):
            raise ValueError("Accumulation requires joint recovery and a positive quality scale")

    def __post_init__(self):
        self._validate_accumulation()
        if (
            not isfinite(self.review_continuity_deviation)
            or not 0 < self.review_continuity_deviation <= self.hard_motion_deviation
        ):
            raise ValueError("Invalid review continuity deviation")


class JointReviewTarget(A2Target):
    def _hold_init(self, encoder, config=None):
        super().__init__(encoder, config or A2JointReviewConfig())

    def _hold_reset(self):
        super().reset()
        self._clear_temporal()

    def _clear_temporal(self):
        self.hold = None
        self.joint = None

    def _hold_select_track(self, track, image, timestamp):
        if not super().select_track(track, image, timestamp):
            return False
        self._clear_temporal()
        return True

    def _hold_strict_support(self, chosen, item, margin, tracks, timestamp):
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

    def _hold_advance_joint(self, chosen, timestamp, supported):
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

    def _hold_update_policy(self, tracks, timestamp, image):
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

    def _hold_remember(self, result, timestamp):
        if result.reason in ("candidate_temporal_confirmed", "candidate_joint_confirmed"):
            result = replace(result, reason="candidate_confirmed")
        super()._remember(result, timestamp)

    def _accumulation_init(self, encoder, config=None):
        self._hold_init(encoder, config or A2JointReviewConfig())

    def _accumulation_reset(self):
        self._hold_reset()
        self.weighted_frame = None
        self.review = None
        self._review_untrusted = False

    def _accumulation_select_track(self, track, image, timestamp):
        if not self._hold_select_track(track, image, timestamp):
            return False
        self.review = None
        self.weighted_frame = None
        self._review_untrusted = False
        return True

    def _accumulation_maybe_update_memory(self, feature, bound, tracks):
        if not self._review_untrusted:
            super()._maybe_update_memory(feature, bound, tracks)

    def _strict_support(self, chosen, item, margin, tracks, timestamp):
        if not self.config.accumulated_recovery_enabled:
            return self._hold_strict_support(chosen, item, margin, tracks, timestamp)
        cfg = self.config
        deviation = item.get("motion_deviation", float("inf"))
        supported = (
            self.frame_motion_valid
            and (not self.gmc_chain_broken)
            and self.motion_debug.get("motion_gate_active", False)
            and (len(self.history) >= cfg.min_motion_observations)
            and (deviation <= cfg.confirmation_deviation)
            and (margin is None or margin >= cfg.joint_margin)
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
        quality = min(1.0, max(0.0, chosen.score) / cfg.joint_min_score)
        weight = quality / (1.0 + deviation**2)
        self.weighted_frame = {
            "strict": supported,
            "weight": weight,
            "similarity": item["appearance_similarity"],
        }
        return supported

    def _advance_joint(self, chosen, timestamp, supported):
        if not self.config.accumulated_recovery_enabled:
            return self._hold_advance_joint(chosen, timestamp, supported)
        cfg, evidence = (self.config, self.weighted_frame)
        pending = self.joint
        consecutive = (
            pending is not None
            and pending["id"] == chosen.tracker_id
            and (0 <= timestamp - pending["last"] <= cfg.max_confirmation_gap_seconds)
            and (timestamp - pending["first"] <= cfg.accumulated_max_span_seconds)
        )
        if not consecutive:
            self.joint = None
            if not supported:
                return False
            self.joint = pending = {
                "id": chosen.tracker_id,
                "first": timestamp,
                "last": timestamp,
                "observations": 1,
                "credit": 0.0,
                "weight": evidence["weight"],
                "was_supported": True,
                "pauses": 0,
                "pause_first": None,
            }
        elif timestamp != pending["last"]:
            if supported:
                if pending["was_supported"]:
                    pending["credit"] += (timestamp - pending["last"]) * min(
                        pending["weight"], evidence["weight"]
                    )
                pending["observations"] += 1
                pending["weight"], pending["was_supported"] = (evidence["weight"], True)
                pending["pauses"], pending["pause_first"] = (0, None)
            elif (
                evidence
                and evidence["strict"]
                and (evidence["similarity"] >= cfg.accumulated_pause_floor)
            ):
                if pending["pause_first"] is None:
                    pending["pause_first"] = pending["last"]
                pending["pauses"] += 1
                if (
                    pending["pauses"] > cfg.temporal_hold_max_observations
                    or timestamp - pending["pause_first"] > cfg.temporal_hold_max_seconds
                ):
                    self.joint = None
                    return False
                pending["was_supported"] = False
            else:
                self.joint = None
                return False
            pending["last"] = timestamp
        elapsed = timestamp - pending["first"]
        self.last_debug.update(
            joint_observations=pending["observations"],
            joint_seconds=elapsed,
            accumulated_credit_seconds=pending["credit"],
            accumulated_weight=evidence["weight"],
            accumulated_pauses=pending["pauses"],
        )
        return (
            supported
            and pending["observations"] >= cfg.joint_confirm_observations
            and (elapsed + 1e-09 >= cfg.joint_confirm_seconds)
            and (pending["credit"] + 1e-09 >= cfg.accumulated_required_seconds)
        )

    def _reference_similarity(self, feature):
        return max((float(v @ feature) for v in self.review["references"]))

    def _accumulation_review_bound(self, tracks, timestamp, image):
        cfg, review = (self.config, self.review)
        debug = {
            "review_status": "inactive",
            "review_similarity": None,
            "review_competitor_similarity": None,
            "review_deviation": None,
            "review_bad_observations": 0,
            "review_trust": None,
        }
        if not cfg.recovered_review_enabled or review is None or review["id"] != self.tracker_id:
            return (tracks, debug)
        debug.update(review_status="missing", review_trust=review["trust"])
        bound = next((t for t in tracks if t.observed and t.tracker_id == self.tracker_id), None)
        if bound is None:
            review["good_first"], review["good_observations"] = (None, 0)
            return (tracks, debug)
        feature = self._normalize(self.encoder.encode(image, [bound])[0])
        score = self._reference_similarity(feature) if feature is not None else None
        deviation = self._continuity_deviation(bound, timestamp)
        competitors = [
            t
            for t in tracks
            if t.tracker_id != bound.tracker_id
            and self._eligible(t)
            and ((d := self._continuity_deviation(t, timestamp)) is not None)
            and (d <= cfg.confirmation_deviation)
        ]
        scores = (
            [
                self._reference_similarity(v)
                for raw in self.encoder.encode(image, competitors)
                if (v := self._normalize(raw)) is not None
            ]
            if competitors
            else []
        )
        competitor = max(scores) if scores else None
        bad = (
            score is None
            or score < cfg.review_similarity_floor
            or (competitor is not None and competitor > score + cfg.review_competitor_delta)
            or (deviation is not None and deviation > cfg.hard_motion_deviation)
        )
        gap = timestamp - review["last_observed"]
        distinct = timestamp != review["last_observed"]
        if distinct:
            review["last_observed"] = timestamp
            if bad:
                review["bad_observations"] += 1
                review["trust"] = max(0.0, review["trust"] - 0.2)
                review["good_first"], review["good_observations"] = (None, 0)
                if (
                    review["bad_observations"] >= cfg.review_bad_observations
                    or gap > cfg.max_confirmation_gap_seconds
                    or (deviation is not None and deviation > cfg.hard_motion_deviation)
                ):
                    review["suspect"] = True
            else:
                review["bad_observations"] = 0
                review["trust"] = min(1.0, review["trust"] + 0.05)
                if review["good_first"] is None or gap > cfg.max_confirmation_gap_seconds:
                    review["good_first"], review["good_observations"] = (timestamp, 0)
                review["good_observations"] += 1
                if (
                    review["suspect"]
                    and review["good_observations"] >= cfg.review_resume_observations
                    and (timestamp - review["good_first"] + 1e-09 >= cfg.review_resume_seconds)
                ):
                    review["suspect"] = False
        debug.update(
            review_status="blocked" if review["suspect"] else "weak" if bad else "supported",
            review_similarity=score,
            review_competitor_similarity=competitor,
            review_deviation=deviation,
            review_bad_observations=review["bad_observations"],
            review_trust=review["trust"],
        )
        if review["suspect"]:
            return (
                [
                    replace(t, observed=False) if t.tracker_id == bound.tracker_id else t
                    for t in tracks
                ],
                debug,
            )
        return (tracks, debug)

    def _accumulation_update_policy(self, tracks, timestamp, image):
        previous_id = self.tracker_id
        previous_reliable = self.last_reliable_timestamp
        previous_continuity = self.last_continuity_timestamp
        filtered, debug = self._review_bound(tracks, timestamp, image)
        self._review_untrusted = debug["review_status"] in ("weak", "blocked")
        self.weighted_frame = None
        result = self._hold_update_policy(filtered, timestamp, image)
        if debug["review_status"] == "weak" and result.valid and (result.tracker_id == previous_id):
            self.last_reliable_timestamp = previous_reliable
            self.last_continuity_timestamp = previous_continuity
            result = replace(result, reason="recovered_identity_weak_grace")
        self.last_debug.update(debug)
        self.last_debug.setdefault("accumulated_credit_seconds", 0.0)
        self.last_debug.setdefault("accumulated_weight", None)
        self.last_debug.setdefault("accumulated_pauses", 0)
        if (
            self.config.recovered_review_enabled
            and result.valid
            and (result.tracker_id != previous_id)
        ):
            references = [
                self.anchor.copy(),
                self.memory.copy(),
                *(v.copy() for v, _, _ in self.bank.templates),
            ]
            self.review = {
                "id": result.tracker_id,
                "references": references,
                "trust": 0.5,
                "last_observed": timestamp,
                "bad_observations": 0,
                "suspect": False,
                "good_first": None,
                "good_observations": 0,
            }
            self.last_debug.update(review_status="new_binding", review_trust=0.5)
        return result

    def __init__(self, encoder, config=None):
        self._accumulation_init(encoder, config or A2JointReviewConfig())

    def reset(self):
        self._accumulation_reset()
        self._joint_review_uncertain = False

    def select_track(self, track, image, timestamp):
        accepted = self._accumulation_select_track(track, image, timestamp)
        if accepted:
            self._joint_review_uncertain = False
        return accepted

    def _maybe_update_memory(self, feature, bound, tracks):
        if not self._joint_review_uncertain:
            self._accumulation_maybe_update_memory(feature, bound, tracks)

    def _review_bound(self, tracks, timestamp, image):
        self._joint_review_uncertain = False
        cfg, review = (self.config, self.review)
        if not cfg.joint_review_enabled:
            return self._accumulation_review_bound(tracks, timestamp, image)
        debug = {
            "review_status": "inactive",
            "review_similarity": None,
            "review_competitor_similarity": None,
            "review_deviation": None,
            "review_bad_observations": 0,
            "review_trust": None,
            "joint_review_reason": "inactive",
            "joint_review_motion_supported": False,
            "joint_review_gap_seconds": None,
        }
        if not cfg.recovered_review_enabled or review is None or review["id"] != self.tracker_id:
            return (tracks, debug)
        bound = next((t for t in tracks if t.observed and t.tracker_id == self.tracker_id), None)
        if bound is None:
            review["good_first"], review["good_observations"] = (None, 0)
            debug.update(
                review_status="missing", review_trust=review["trust"], joint_review_reason="missing"
            )
            return (tracks, debug)
        feature = self._normalize(self.encoder.encode(image, [bound])[0])
        score = self._reference_similarity(feature) if feature is not None else None
        deviation = self._continuity_deviation(bound, timestamp)
        competitors = [
            t
            for t in tracks
            if t.tracker_id != bound.tracker_id
            and self._eligible(t)
            and ((d := self._continuity_deviation(t, timestamp)) is not None)
            and (d <= cfg.confirmation_deviation)
        ]
        scores = (
            [
                self._reference_similarity(v)
                for raw in self.encoder.encode(image, competitors)
                if (v := self._normalize(raw)) is not None
            ]
            if competitors
            else []
        )
        competitor = max(scores) if scores else None
        gap = timestamp - review["last_observed"]
        motion_ok = (
            self.frame_motion_valid
            and (not self.gmc_chain_broken)
            and (len(self.history) >= cfg.min_motion_observations)
            and (deviation is not None)
            and (deviation <= cfg.review_continuity_deviation)
        )
        continuous = 0 <= gap <= cfg.max_confirmation_gap_seconds
        low = score is None or score < cfg.review_similarity_floor
        rival = (
            score is not None
            and competitor is not None
            and (competitor > score + cfg.review_competitor_delta)
        )
        severe = deviation is not None and deviation > cfg.hard_motion_deviation
        bad = score is None or rival or severe or (low and (not (continuous and motion_ok)))
        strong = not low and (not bad) and motion_ok
        reason = (
            "invalid_feature"
            if score is None
            else "competitor"
            if rival
            else "severe_motion"
            if severe
            else "low_after_gap"
            if low and (not continuous)
            else "low_without_motion"
            if low and (not motion_ok)
            else "continuous_low_appearance"
            if low
            else "identity_supported"
        )
        if timestamp != review["last_observed"]:
            review["last_observed"] = timestamp
            if bad:
                review["bad_observations"] += 1
                review["trust"] = max(0.0, review["trust"] - 0.2)
                review["good_first"], review["good_observations"] = (None, 0)
                if (
                    review["bad_observations"] >= cfg.review_bad_observations
                    or gap > cfg.max_confirmation_gap_seconds
                    or severe
                ):
                    review["suspect"] = True
            else:
                review["bad_observations"] = 0
                if strong:
                    review["trust"] = min(1.0, review["trust"] + 0.05)
                    if review["good_first"] is None or not continuous:
                        review["good_first"], review["good_observations"] = (timestamp, 0)
                    review["good_observations"] += 1
                    if (
                        review["suspect"]
                        and review["good_observations"] >= cfg.review_resume_observations
                        and (timestamp - review["good_first"] + 1e-09 >= cfg.review_resume_seconds)
                    ):
                        review["suspect"] = False
                else:
                    review["good_first"], review["good_observations"] = (None, 0)
        status = (
            "blocked"
            if review["suspect"]
            else "weak"
            if bad
            else "continuity_supported"
            if low
            else "supported"
        )
        self._joint_review_uncertain = status == "continuity_supported"
        debug.update(
            review_status=status,
            review_similarity=score,
            review_competitor_similarity=competitor,
            review_deviation=deviation,
            review_bad_observations=review["bad_observations"],
            review_trust=review["trust"],
            joint_review_reason=reason,
            joint_review_motion_supported=motion_ok,
            joint_review_gap_seconds=gap,
        )
        if review["suspect"]:
            return (
                [
                    replace(t, observed=False) if t.tracker_id == bound.tracker_id else t
                    for t in tracks
                ],
                debug,
            )
        return (tracks, debug)

    def _update_policy(self, tracks, timestamp, image):
        previous_id, reliable = (self.tracker_id, self.last_reliable_timestamp)
        result = self._accumulation_update_policy(tracks, timestamp, image)
        if self._joint_review_uncertain and result.tracker_id == previous_id:
            bound = next(
                (t for t in tracks if t.tracker_id == previous_id and self._eligible(t)), None
            )
            if bound is not None and self.state != TargetState.NEED_RESELECT:
                self.last_reliable_timestamp = reliable
                self.state = TargetState.TRACKING
                self._clear_candidate()
                self._clear_temporal()
                result = self._result(
                    True, bound.bbox_xyxy, timestamp, "recovered_identity_continuity"
                )
        return result

    def _remember(self, result, timestamp):
        if result.reason == "recovered_identity_continuity":
            result = replace(result, reason="bound_id_continuity")
        self._hold_remember(result, timestamp)
