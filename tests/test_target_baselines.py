import numpy as np

from a1_tracker.target import BaselinePair, TargetBaseline
from a1_tracker.types import TargetState, Track


class FeatureEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, image, tracks):
        return [self.vectors.get((image, t.tracker_id)) for t in tracks]


def track(track_id, bbox=(0, 0, 20, 40), observed=True):
    return Track(track_id, bbox, observed, 0.9)


def test_a0_keeps_initial_id_even_when_another_track_appears():
    manager = TargetBaseline("A0")
    assert manager.update([track(1)], 0).state == TargetState.UNSELECTED
    assert manager.select_track(track(7), None)
    session = manager.session_target_id
    missing = manager.update([track(23)], 1)
    assert (missing.state, missing.valid, missing.tracker_id) == (TargetState.LOST, False, 7)
    assert manager.update([track(7, observed=False)], 2).state == TargetState.LOST
    found = manager.update([track(7)], 3)
    assert (found.state, found.tracker_id, found.session_target_id) == (
        TargetState.TRACKING,
        7,
        session,
    )


def test_a1_rebinds_to_nearest_appearance_after_t2_id_loss():
    encoder = FeatureEncoder(
        {
            ("first", 7): np.array([1.0, 0.0]),
            ("next", 23): np.array([0.1, 0.9]),
            ("next", 24): np.array([0.9, 0.1]),
        }
    )
    pair = BaselinePair(encoder)
    assert pair.select_by_point([track(7), track(8, (30, 0, 50, 40))], (10, 10), "first")
    session = pair.a0.session_target_id
    a0, a1 = pair.update([track(23), track(24)], 1, "next")
    assert a0.state == TargetState.LOST and not a0.valid and a0.tracker_id == 7
    assert a1.state == TargetState.TRACKING and a1.valid and a1.tracker_id == 24
    assert a1.reason == "appearance_rebound"
    assert a1.session_target_id == session == a0.session_target_id


def test_pair_selection_is_atomic_when_anchor_cannot_be_encoded():
    pair = BaselinePair(FeatureEncoder({}))
    assert not pair.select_track(track(7), "first")
    assert pair.a0.state == pair.a1.state == TargetState.UNSELECTED


def test_a1_waits_if_candidates_have_no_valid_appearance():
    manager = TargetBaseline("A1", FeatureEncoder({("first", 7): np.array([1.0, 0.0])}))
    manager.select_track(track(7), "first")
    result = manager.update([track(23)], 1, "next")
    assert result.state == TargetState.LOST and result.reason == "no_valid_appearance"


def test_point_selection_rejects_ambiguous_overlap_without_mutating_state():
    image = "first"
    overlapping = [track(7, (0, 0, 20, 40)), track(8, (5, 0, 25, 40))]
    a0 = TargetBaseline("A0")
    assert not a0.select_by_point(overlapping, (10, 10), image)
    assert a0.state == TargetState.UNSELECTED

    pair = BaselinePair(FeatureEncoder({(image, 7): np.array([1.0, 0.0])}))
    assert not pair.select_by_point(overlapping, (10, 10), image)
    assert pair.a0.state == pair.a1.state == TargetState.UNSELECTED
    assert pair.select_by_point([track(7), track(8, (30, 0, 50, 40))], (10, 10), image)
    assert pair.a0.tracker_id == pair.a1.tracker_id == 7
