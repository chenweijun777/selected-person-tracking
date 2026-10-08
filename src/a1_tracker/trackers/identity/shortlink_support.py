# ruff: noqa: C408
"""Appearance gallery and causal geometry used by ShortLink.

Contains no residual association or reactivation stage. Gallery sampling and
geometry are copied exactly from the frozen pre-removal implementation.
"""

import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching

__all__ = ["ShortLinkGallery", "TrackState", "admissible_pairs", "box_iou", "dynamic_geometry"]

PARAMETERS = dict(recovery_update_delay=3, gallery_interval=3)


def admissible_pairs(similarity, geometry, cosine_min=0.80, margin_min=0.05):
    """Require absolute identity consistency and unique row/column winners."""
    sims = np.asarray(similarity, dtype=float)
    valid = np.zeros(sims.shape, dtype=bool)
    if not sims.size:
        return valid
    for i, j in zip(
        *np.nonzero(np.asarray(geometry, dtype=bool) & np.isfinite(sims) & (sims >= cosine_min))
    ):
        row = np.delete(sims[i], j)
        col = np.delete(sims[:, j], i)
        row_other = max((x for x in row if np.isfinite(x)), default=-1.0)
        col_other = max((x for x in col if np.isfinite(x)), default=-1.0)
        if sims[i, j] - row_other >= margin_min and sims[i, j] - col_other >= margin_min:
            valid[i, j] = True
    return valid


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0))
    return float(area / max(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - area, 1e-6))


def dynamic_geometry(self, tracks, detections):
    raw = matching.iou_distance(tracks, detections)
    mask = np.zeros(raw.shape, dtype=bool)
    deviations = np.zeros(raw.shape, dtype=float)
    ages = []
    for i, t in enumerate(tracks):
        age = max(0.0, (self.tracker.frame_id - t.end_frame) / self.fps)
        ages.append(age)
        box = np.asarray(t.xyxy)
        size = box[2:] - box[:2]
        center = (box[:2] + box[2:]) / 2
        radius = np.array([min(8.0, 1.0 + 1.5 * age), min(4.0, 0.5 + 0.75 * age)])
        for j, d in enumerate(detections):
            db = np.asarray(d.xyxy)
            ds = np.maximum(db[2:] - db[:2], 1e-6)
            ratios = size / ds
            offset = np.abs(center - (db[:2] + db[2:]) / 2) / ds
            deviations[i, j] = float(np.max(offset / radius))
            mask[i, j] = bool(
                np.all(ratios >= 0.5)
                and np.all(ratios <= 2.0)
                and np.all(offset <= radius)
                and 1.0 - raw[i, j] + 1e-7 >= max(0.0, 0.2 - 0.2 * age)
            )
    return raw, mask, deviations, ages


class ShortLinkGallery:
    def ensure(self, indices):
        missing = sorted(set(map(int, indices)) - set(self.features))
        if missing:
            self.features.update(self.feature_provider(missing))

    def seed_threshold(self, sample_count):
        return 0.25 if self.options["seed_low"] and sample_count < 2 else 0.50

    def update_gallery(self):
        frame = self.tracker.frame_id
        boxes = np.asarray([d["box"] for d in self.inputs]).reshape(-1, 4)
        eligible = []
        for t in self.tracker.tracked_stracks:
            idx = int(t.idx)
            bank = self.gallery.get(t.track_id, [])
            if not t.is_activated or t.frame_id != frame:
                continue
            if self.inputs[idx]["score"] < self.seed_threshold(len(bank)):
                continue
            if (
                frame - self.reactivated.get(t.track_id, -10000)
                < PARAMETERS["recovery_update_delay"]
            ):
                continue
            if bank and frame - bank[-1][0] < PARAMETERS["gallery_interval"]:
                continue
            others = np.delete(boxes, idx, axis=0)
            if len(others):
                box = boxes[idx]
                lo = np.maximum(box[:2], others[:, :2])
                hi = np.minimum(box[2:], others[:, 2:])
                area = np.prod(np.maximum(hi - lo, 0), axis=1)
                union = (
                    np.prod(box[2:] - box[:2])
                    + np.prod(others[:, 2:] - others[:, :2], axis=1)
                    - area
                )
                if np.any(area / np.maximum(union, 1e-6) >= 0.5):
                    continue
            eligible.append((t, idx))
        self.ensure([idx for t, idx in eligible])
        for t, idx in eligible:
            if idx not in self.features:
                continue
            bank = self.gallery.setdefault(t.track_id, [])
            self.gallery_records.append(
                dict(
                    frame=frame,
                    id=int(t.track_id),
                    det_index=idx,
                    score=float(self.inputs[idx]["score"]),
                    seed=bool(len(bank) < 2),
                    box=list(self.inputs[idx]["box"]),
                )
            )
            bank.append((frame, self.features[idx].copy()))
            self.gallery[t.track_id] = bank[-3:]
        alive = {t.track_id for t in self.tracker.tracked_stracks + self.tracker.lost_stracks}
        self.gallery = {k: v for k, v in self.gallery.items() if k in alive}
