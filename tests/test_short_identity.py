"""Promoted causal identity tests from the frozen ShortLink2 experiment."""

import os
from pathlib import Path

import pytest

os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / "outputs/.ultralytics")
)
pytest.importorskip("ultralytics")
import unittest
from types import SimpleNamespace as NS

import numpy as np
from ultralytics.trackers.bot_sort import BOTrack
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
from ultralytics.trackers.utils.stracks import merge_track_pools

from a1_tracker.trackers.identity.short_identity import ShortIdentity
from a1_tracker.trackers.identity.shortlink_support import TrackState


class ShortIdentityTests(unittest.TestCase):
    def subject(self, confirm=2):
        o = ShortIdentity.__new__(ShortIdentity)
        o.fps = 30
        o.confirm = confirm
        o.options = {"cosine_min": 0.525}
        o.short_cosine_min = 0.525
        o.margin_min = 0.05
        o.max_young_age_seconds = 0.5
        o.pending = {}
        o.logs = []
        o.features = {}
        o.reactivated = {}
        o.identity_remaps = []
        kf = KalmanFilterXYWH()
        old = BOTrack(np.array([15, 45, 30, 90, 0]), 0.9, 0)
        old.activate(kf, 1)
        old.frame_id = 10
        old.mark_lost()
        young = BOTrack(np.array([75, 45, 30, 90, 0]), 0.8, 0)
        young.activate(kf, 68)
        young.frame_id = 70
        young.idx = 0
        young.is_activated = True
        o.gallery = {old.track_id: [(1, np.array([1.0, 0.0])), (4, np.array([1.0, 0.0]))]}
        o.lost = [old]
        o.removed_sink = []
        o.inputs = [{"box": [60, 0, 90, 90], "score": 0.8}]
        o.feature_provider = lambda ids: {i: np.array([0.8, 0.6]) for i in ids}
        o.tracker = NS(
            frame_id=70,
            max_frames_lost=150,
            tracked_stracks=[young],
            lost_stracks=[old],
            removed_stracks=[],
        )
        return o, old, young

    def test_pending_keeps_young_output_and_current_associations(self):
        o, old, y = self.subject()
        activated = [y]
        refind = []
        o.link_short_tracks(activated, refind)
        self.assertEqual(activated, [y])
        self.assertFalse(refind)
        self.assertEqual(old.state, TrackState.Lost)
        self.assertEqual(y.state, TrackState.Tracked)
        self.assertEqual(next(iter(o.pending.values()))["count"], 1)

    def test_confirm_restores_old_id_without_duplicate_output_or_removed_old(self):
        o, old, y = self.subject()
        o.link_short_tracks([y], [])
        old_id, young_id = old.track_id, y.track_id
        mean = y.mean.copy()
        o.tracker.frame_id = y.frame_id = 71
        o.features = {}
        o.logs = []
        activated = [y]
        refind = []
        o.link_short_tracks(activated, refind)
        self.assertEqual(old.track_id, old_id)
        self.assertEqual(y.track_id, young_id)
        self.assertEqual(old.frame_id, 71)
        self.assertEqual(y.state, TrackState.Removed)
        np.testing.assert_array_equal(old.mean, mean)
        merge_track_pools(o.tracker, activated, refind, [], o.removed_sink)
        self.assertEqual([t.track_id for t in o.tracker.tracked_stracks], [old_id])
        self.assertFalse(o.tracker.lost_stracks)
        self.assertEqual(o.tracker.removed_stracks, [y])

    def test_mature_track_cannot_be_relabelled(self):
        o, old, y = self.subject()
        y.start_frame = 1
        o.link_short_tracks([y], [])
        self.assertFalse(o.pending)
        self.assertFalse(o.logs)
        self.assertEqual(old.state, TrackState.Lost)

    def test_three_observations_and_gap_reset_are_causal(self):
        o, old, y = self.subject(confirm=3)
        o.link_short_tracks([y], [])
        o.tracker.frame_id = y.frame_id = 71
        o.link_short_tracks([y], [])
        self.assertEqual(old.state, TrackState.Lost)
        o.tracker.frame_id = y.frame_id = 73
        o.link_short_tracks([y], [])
        self.assertEqual(next(iter(o.pending.values()))["count"], 1)
        o.tracker.frame_id = y.frame_id = 74
        o.link_short_tracks([y], [])
        o.tracker.frame_id = y.frame_id = 75
        o.link_short_tracks([y], [])
        self.assertEqual(old.state, TrackState.Tracked)

    def test_ambiguous_candidates_do_not_link(self):
        o, _old, y = self.subject()
        y2 = BOTrack(np.array([76, 45, 30, 90, 1]), 0.8, 0)
        y2.activate(KalmanFilterXYWH(), 68)
        y2.frame_id = 70
        y2.idx = 1
        o.inputs.append({"box": [61, 0, 91, 90], "score": 0.8})
        o.link_short_tracks([y, y2], [])
        self.assertFalse(o.pending)

    def test_expired_lost_and_wrong_identity_are_rejected(self):
        o, _old, y = self.subject()
        o.tracker.max_frames_lost = 30
        o.link_short_tracks([y], [])
        self.assertFalse(o.pending)
        o.tracker.max_frames_lost = 150
        o.features = {0: np.array([0.3, np.sqrt(0.91)])}
        o.link_short_tracks([y], [])
        self.assertFalse(o.pending)

    def test_short_threshold_controls_linking(self):
        for threshold, expected in [(0.50, True), (0.525, True), (0.55, False)]:
            o, old, y = self.subject(confirm=1)
            o.short_cosine_min = threshold
            o.features = {0: np.array([0.53, np.sqrt(1 - 0.53**2)])}
            o.link_short_tracks([y], [])
            self.assertEqual(old.state == TrackState.Tracked, expected)

    def test_extended_young_age_is_in_seconds(self):
        for age, expected in [(0.5, False), (0.75, True), (1.0, True)]:
            o, old, y = self.subject(confirm=1)
            y.start_frame = 52  # 18/30 = 0.6 seconds old
            o.max_young_age_seconds = age
            o.link_short_tracks([y], [])
            self.assertEqual(old.state == TrackState.Tracked, expected)

    def test_short_margin_checks_the_competing_detection(self):
        for margin, expected in [(0.03, True), (0.05, True), (0.08, False)]:
            o, old, y = self.subject(confirm=1)
            o.margin_min = margin
            o.inputs.append({"box": [300, 0, 330, 90], "score": 0.8})
            o.features = {0: np.array([0.8, 0.6]), 1: np.array([0.74, np.sqrt(1 - 0.74**2)])}
            o.link_short_tracks([y], [])
            self.assertEqual(old.state == TrackState.Tracked, expected)


if __name__ == "__main__":
    unittest.main()
