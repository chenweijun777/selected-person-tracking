from a1_tracker.trackers import GreedyIoUTracker
from a1_tracker.types import Detection


def test_tracker_keeps_id_for_overlapping_detection() -> None:
    tracker = GreedyIoUTracker(match_iou=0.2)
    first = tracker.update([Detection((0, 0, 20, 40), 0.9)])[0]
    second = tracker.update([Detection((2, 0, 22, 40), 0.8)])[0]
    assert first.tracker_id == second.tracker_id
    assert second.age == 2


def test_tracker_creates_new_id_for_distant_detection() -> None:
    tracker = GreedyIoUTracker(match_iou=0.2)
    first = tracker.update([Detection((0, 0, 20, 40), 0.9)])[0]
    second = tracker.update([Detection((100, 100, 120, 140), 0.8)])[0]
    assert first.tracker_id != second.tracker_id
