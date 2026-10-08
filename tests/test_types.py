import pytest

from a1_tracker.types import Detection, TargetResult, TargetState


def test_invalid_bbox_is_rejected() -> None:
    with pytest.raises(ValueError):
        Detection((5, 5, 4, 6), 0.9)


def test_tracking_result_must_be_valid() -> None:
    with pytest.raises(ValueError):
        TargetResult("target", 1, TargetState.TRACKING, False, None, 0.0, "invalid")
