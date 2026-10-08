"""Causal, bounded appearance templates for offline recovery experiments."""

from dataclasses import dataclass

import numpy as np

from a1_tracker.geometry import iou


def normalize(feature):
    if feature is None:
        return None
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if np.isfinite(norm) and norm > 0 else None


@dataclass(frozen=True)
class BankConfig:
    capacity: int = 5
    min_similarity: float = 0.5
    min_score: float = 0.5
    max_overlap: float = 0.3
    min_observations: int = 3
    min_duration: float = 0.2
    max_gap: float = 0.2
    sample_interval: float = 0.2
    novelty_similarity: float = 0.95


class AppearanceBank:
    """Only verified same-ID observations can add templates; GT is never input."""

    def __init__(self, config=None):
        config = config or BankConfig()
        if config.capacity < 1:
            raise ValueError("capacity must be positive")
        self.config = config
        self.templates = []  # (feature, frame, tracker_id)
        self.last_id = self.last_time = self.start_time = self.last_sample = None
        self.observations = 0

    def snapshot(self):
        return [(v.copy(), f, identifier) for v, f, identifier in self.templates]

    def observe(
        self, *, result, bound, tracks, feature, similarity, timestamp, frame, motion_valid
    ):
        cfg = self.config
        valid = (
            result.valid
            and result.reason in ("bound_verified", "same_id_resumed")
            and bound is not None
            and motion_valid
            and similarity is not None
            and similarity >= cfg.min_similarity
            and bound.score >= cfg.min_score
            and bound.observed
            and bound.bbox_xyxy[2] - bound.bbox_xyxy[0] >= 24
            and bound.bbox_xyxy[3] - bound.bbox_xyxy[1] >= 48
            and not any(
                iou(bound.bbox_xyxy, t.bbox_xyxy) > cfg.max_overlap
                for t in tracks
                if t.observed and t.tracker_id != bound.tracker_id
            )
        )
        vector = normalize(feature) if valid else None
        if vector is None:
            self.last_id = self.last_time = self.start_time = None
            self.observations = 0
            return False
        if (
            self.last_id != bound.tracker_id
            or self.last_time is None
            or not 0 <= timestamp - self.last_time <= cfg.max_gap
        ):
            self.start_time, self.observations = timestamp, 0
        if self.last_time != timestamp:
            self.observations += 1
        self.last_time, self.last_id = timestamp, bound.tracker_id
        if (
            self.observations < cfg.min_observations
            or timestamp - self.start_time + 1e-9 < cfg.min_duration
            or (
                self.last_sample is not None
                and timestamp - self.last_sample + 1e-9 < cfg.sample_interval
            )
        ):
            return False
        if (
            self.templates
            and max(float(v @ vector) for v, _, _ in self.templates) >= cfg.novelty_similarity
        ):
            return False
        self.last_sample = timestamp
        pool = self.templates + [(vector.copy(), frame, bound.tracker_id)]
        # Keep the first trusted template; then greedily retain diverse views.
        selected = [pool[0]]
        remaining = list(range(1, len(pool)))
        while remaining and len(selected) < cfg.capacity:
            chosen = min(
                remaining, key=lambda j: max(float(pool[j][0] @ v) for v, _, _ in selected)
            )
            selected.append(pool[chosen])
            remaining.remove(chosen)
        changed = [t[1:] for t in selected] != [t[1:] for t in self.templates]
        self.templates = selected
        return changed

    @staticmethod
    def score(feature, templates, baseline):
        return max([baseline, *(float(v @ feature) for v, _, _ in templates)])
