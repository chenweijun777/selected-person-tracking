import pytest

from a1_tracker.geometry import contains_point, iou


def test_iou_for_partially_overlapping_boxes() -> None:
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_contains_point_includes_boundary() -> None:
    assert contains_point((10, 10, 20, 20), (10, 15))
    assert not contains_point((10, 10, 20, 20), (9, 15))
