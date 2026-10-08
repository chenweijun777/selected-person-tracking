from a1_tracker.target import TargetManager
from a1_tracker.types import TargetState, Track


def track(track_id: int, bbox=(0, 0, 20, 40)) -> Track:
    return Track(track_id, bbox, True, 0.9)


def test_state_flow_from_selection_to_lost_and_rebound() -> None:
    manager = TargetManager(lost_timeout_frames=4, verify_observations=2, candidate_iou=0.2)
    manager.select_track(track(7))

    assert manager.update([track(7)], 0.0).state == TargetState.TRACKING
    assert manager.update([], 0.1).state == TargetState.LOST

    first_candidate = manager.update([track(23, (1, 0, 21, 40))], 0.2)
    assert first_candidate.state == TargetState.VERIFYING
    assert not first_candidate.valid

    rebound = manager.update([track(23, (2, 0, 22, 40))], 0.3)
    assert rebound.state == TargetState.TRACKING
    assert rebound.valid
    assert rebound.tracker_id == 23


def test_timeout_requests_reselection() -> None:
    manager = TargetManager(lost_timeout_frames=1)
    manager.select_track(track(1))
    manager.update([], 0.1)
    result = manager.update([], 0.2)
    assert result.state == TargetState.NEED_RESELECT
    assert not result.valid


def test_select_by_point_does_not_mutate_when_no_track_hit() -> None:
    manager = TargetManager()
    assert not manager.select_by_point([track(1)], (100, 100))
    assert manager.state == TargetState.UNSELECTED


def test_select_by_point_rejects_ambiguous_or_unobserved_track() -> None:
    manager = TargetManager()
    assert not manager.select_by_point([track(1), track(2, (5, 0, 25, 40))], (10, 10))
    assert manager.state == TargetState.UNSELECTED
    assert not manager.select_by_point([Track(3, (0, 0, 20, 40), False, 0.9)], (10, 10))
    assert manager.state == TargetState.UNSELECTED
    assert manager.select_by_point([track(1)], (10, 10))
    assert manager.tracker_id == 1
