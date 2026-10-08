# ruff: noqa: C408, RUF059
"""Stitch young tracklets to Lost identities after native BoT-SORT association."""

import numpy as np

from a1_tracker.identity_events import IdentityRemap

from .shortlink_support import (
    ShortLinkGallery,
    TrackState,
    admissible_pairs,
    box_iou,
    dynamic_geometry,
)


class ShortIdentity(ShortLinkGallery):
    def __init__(
        self,
        tracker,
        options,
        feature_provider,
        fps,
        confirm=1,
        *,
        short_cosine_min=0.50,
        margin_min=0.03,
        max_young_age_seconds=0.5,
    ):
        if not isinstance(confirm, int) or isinstance(confirm, bool) or confirm < 1:
            raise ValueError("ShortLink confirmation count must be a positive integer")
        cosine = 0.50 if short_cosine_min is None else short_cosine_min
        if not np.isfinite(cosine) or not -1 <= cosine <= 1:
            raise ValueError("ShortLink cosine threshold must be finite and in [-1, 1]")
        if not np.isfinite(margin_min) or not 0 <= margin_min <= 2:
            raise ValueError("ShortLink margin must be finite and in [0, 2]")
        if not np.isfinite(max_young_age_seconds) or max_young_age_seconds <= 0:
            raise ValueError("ShortLink young age must be positive and finite")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("ShortLink FPS must be positive and finite")
        self.tracker = tracker
        self.options = options
        self.feature_provider = feature_provider
        self.gallery = {}
        self.gallery_records = []
        self.reactivated = {}
        self.features = {}
        self.inputs = []
        self.lost = []
        self.logs = []
        self.identity_remaps = []
        self.original_first = tracker._first_association
        self.original_init = tracker._init_new_tracks
        tracker._first_association = self.first
        tracker._init_new_tracks = self.init
        self.fps = fps
        self.confirm = confirm
        self.short_cosine_min = cosine
        self.margin_min = margin_min
        self.max_young_age_seconds = max_young_age_seconds
        self.pending = {}
        self.removed_sink = None
        self.original_unconfirmed = tracker._unconfirmed_association
        tracker._unconfirmed_association = self.unconfirmed

    def unconfirmed(self, unconfirmed, u_detection, detections, activated, removed):
        self.removed_sink = removed
        return self.original_unconfirmed(unconfirmed, u_detection, detections, activated, removed)

    def init(self, u_detection, detections, activated, refind=None):
        result = self.original_init(u_detection, detections, activated, refind)
        self.link_short_tracks(activated, refind)
        return result

    def frame_input(self, features, inputs):
        self.features = features
        self.inputs = inputs
        self.lost = []
        self.logs = []
        self.identity_remaps = []

    def first(self, pool, detections, activated, refind):
        self.lost = [t for t in pool if t.state == TrackState.Lost]
        result = self.original_first(pool, detections, activated, refind)
        for t in refind:
            self.reactivated[t.track_id] = self.tracker.frame_id
        return result

    def link_short_tracks(self, activated, refind):
        frame = self.tracker.frame_id
        observed = {
            t.track_id: t
            for t in activated + list(refind or [])
            if t.state == TrackState.Tracked and t.frame_id == frame
        }
        young = [
            t
            for t in observed.values()
            if 0 <= (frame - t.start_frame) / self.fps <= self.max_young_age_seconds
            and self.inputs[int(t.idx)]["score"] >= 0.25
        ]
        lost = [
            t
            for t in self.lost
            if t.state == TrackState.Lost
            and 0.5 <= (frame - t.end_frame) / self.fps <= self.tracker.max_frames_lost / self.fps
            and len(self.gallery.get(t.track_id, [])) >= 2
        ]
        if not young or not lost:
            self.pending = {}
            return
        # Score all high detections and all usable Lost galleries for mutual
        # margins, including competitors outside the eligible young-track set.
        high = [i for i, d in enumerate(self.inputs) if d["score"] >= 0.25]
        self.ensure(high)
        competitors = [
            t
            for t in self.lost
            if t.state == TrackState.Lost and len(self.gallery.get(t.track_id, [])) >= 2
        ]
        sims = np.full((len(competitors), len(high)), np.nan)
        for i, t in enumerate(competitors):
            bank = self.gallery[t.track_id]
            for j, di in enumerate(high):
                feat = self.features.get(di)
                if feat is not None:
                    sims[i, j] = float(np.mean([np.dot(v, feat) for _, v in bank]))
        detections = [
            type("ObservedBox", (), {"xyxy": np.asarray(self.inputs[i]["box"]), "angle": None})()
            for i in high
        ]
        raw, geometry, deviation, ages = dynamic_geometry(self, competitors, detections)
        allowed = admissible_pairs(sims, geometry, self.short_cosine_min, self.margin_min)
        young_by_det = {int(t.idx): t for t in young}
        eligible = {t.track_id for t in lost}
        next_pending = {}
        for i, t in enumerate(competitors):
            if t.track_id not in eligible:
                continue
            for j, di in enumerate(high):
                if di not in young_by_det:
                    continue
                y = young_by_det[di]
                key = (t.track_id, y.track_id)
                self.logs.append(
                    dict(
                        frame=frame,
                        stage="short_identity_query",
                        old_id=int(t.track_id),
                        young_id=int(y.track_id),
                        det_index=di,
                        young_age_seconds=(frame - y.start_frame) / self.fps,
                        lost_age_seconds=ages[i],
                        gallery_frames=[int(fr) for fr, _ in self.gallery[t.track_id]],
                        similarity=None if not np.isfinite(sims[i, j]) else float(sims[i, j]),
                        row_margin=None
                        if not np.isfinite(sims[i, j])
                        else float(
                            sims[i, j]
                            - max(
                                (v for k, v in enumerate(sims[i]) if k != j and np.isfinite(v)),
                                default=-1,
                            )
                        ),
                        column_margin=None
                        if not np.isfinite(sims[i, j])
                        else float(
                            sims[i, j]
                            - max(
                                (v for k, v in enumerate(sims[:, j]) if k != i and np.isfinite(v)),
                                default=-1,
                            )
                        ),
                        geometry=bool(geometry[i, j]),
                        allowed=bool(allowed[i, j]),
                        normalized_deviation=float(deviation[i, j]),
                    )
                )
                if not allowed[i, j]:
                    continue
                prev = self.pending.get(key)
                box = np.asarray(self.inputs[di]["box"])
                consecutive = (
                    prev is not None
                    and prev["frame"] == frame - 1
                    and box_iou(prev["box"], box) >= 0.5
                )
                count = prev["count"] + 1 if consecutive else 1
                start = prev["start"] if consecutive else frame
                if count >= self.confirm:
                    self.stitch(t, y, activated, refind)
                    self.logs.append(
                        dict(
                            frame=frame,
                            stage="short_identity_link",
                            old_id=int(t.track_id),
                            young_id=int(y.track_id),
                            det_index=di,
                            similarity=float(sims[i, j]),
                            confirmation_frames=list(range(start, frame + 1)),
                            gallery_frames=[int(fr) for fr, _ in self.gallery.get(t.track_id, [])],
                        )
                    )
                else:
                    next_pending[key] = dict(frame=frame, box=box.copy(), count=count, start=start)
        self.pending = next_pending

    def stitch(self, old, young, activated, refind):
        assert old.state == TrackState.Lost and young.state == TrackState.Tracked
        assert refind is not None and self.removed_sink is not None
        old.re_activate(young, self.tracker.frame_id, new_id=False)
        # The current young posterior uses the observed trajectory; retaining
        # the long-gap prediction here would briefly shift the restored box.
        old.mean = young.mean.copy()
        old.covariance = young.covariance.copy()
        young.mark_removed()
        self.removed_sink.append(young)
        activated[:] = [t for t in activated if t.track_id != young.track_id]
        refind[:] = [t for t in refind if t.track_id != young.track_id]
        refind.append(old)
        self.reactivated[old.track_id] = self.tracker.frame_id
        self.identity_remaps.append(
            IdentityRemap(
                int(self.tracker.frame_id),
                int(young.track_id),
                int(old.track_id),
                tuple(float(v) for v in old.xyxy),
            )
        )
