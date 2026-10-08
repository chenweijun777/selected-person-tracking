import runpy
from pathlib import Path

import pytest

module = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/target_recovery_metrics.py")
)
frame_metrics = module["frame_metrics"]
event_metrics = module["event_metrics"]
initialization_metrics = module["initialization_metrics"]


def frame(mode, number, category="correct", valid="True", visibility="1", ambiguous="False"):
    return {
        "detector": "yolo26n-one2many_960",
        "sequence": "MOT17-02-SDP",
        "gt_id": "7",
        "frame": str(number),
        "strategy": mode,
        "category": category,
        "valid": valid,
        "visibility": visibility,
        "ambiguous_overlap": ambiguous,
    }


def test_full_frame_denominator_keeps_wrong_reject_and_ambiguity():
    a0 = [
        frame("A0", 1),
        frame("A0", 2, "other"),
        frame("A0", 3, valid="False"),
        frame("A0", 4, ambiguous="True"),
        frame("A0", 5, visibility="0"),
    ]
    a1 = [frame("A1", n, visibility="0" if n == 5 else "1") for n in range(1, 6)]
    metrics = frame_metrics(a0 + a1, ("A0", "A1"), "yolo26n-one2many_960")
    assert metrics["development"]["A0"]["person_frames"] == 4
    assert metrics["development"]["A0"]["correct_pct"] == 25
    assert metrics["development"]["A0"]["W"] == 1
    assert metrics["development"]["A0"]["R"] == 1
    assert metrics["development"]["A0"]["A"] == 1


def test_different_person_frame_sets_fail():
    with pytest.raises(ValueError, match="Different evaluated"):
        frame_metrics([frame("A0", 1), frame("A1", 2)], ("A0", "A1"), "yolo26n-one2many_960")


def test_initialization_uses_same_people_for_each_method():
    rows = [
        {
            "detector": "yolo26n-one2many_960",
            "sequence": "MOT17-02-SDP",
            "gt_id": "7",
            "strategy": mode,
            "initialization_success": success,
        }
        for mode, success in (("A0", "True"), ("A1", "False"))
    ]
    metrics = initialization_metrics(rows, ("A0", "A1"), "yolo26n-one2many_960")
    assert metrics["development"]["A0"]["success_pct"] == 100
    assert metrics["development"]["A1"]["success_pct"] == 0


def test_event_gains_losses_and_wrong_can_coexist():
    rows = []
    for event_id, a0, a1, wrong, delay in (
        ("one", "11", "", "13", ""),
        ("two", "", "12", "10", "0.5"),
        ("three", "14", "15", "", "0.25"),
    ):
        rows.append(
            {
                "event_id": event_id,
                "split": "development",
                "common_cohort": "True",
                "event_review_status": "needs_review",
                "A0_ever_rejected_before_stable": "True" if event_id != "three" else "False",
                "A0_stable_correct_start": a0,
                "A0_first_wrong_frame": "",
                "A0_delay_s": "0.1" if a0 else "",
                "A1_stable_correct_start": a1,
                "A1_first_wrong_frame": wrong,
                "A1_delay_s": delay,
            }
        )
    metrics = event_metrics(rows, ("A0", "A1"), "A0")["development"]
    assert metrics["events"] == 3
    assert metrics["methods"]["A1"]["gain_over_baseline"] == 1
    assert metrics["methods"]["A1"]["loss_from_baseline"] == 1
    assert metrics["methods"]["A1"]["preserved_baseline"] == 1
    assert metrics["methods"]["A1"]["wrong_events"] == 2
    assert metrics["methods"]["A1"]["median_delay_s"] == 0.375
    assert metrics["a0_outage"]["events"] == 2
    assert metrics["a0_outage"]["methods"]["A1"]["gain_over_baseline"] == 1
