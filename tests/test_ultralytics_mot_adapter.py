"""Adapter contract for the six bundled MOT implementations."""

from pathlib import Path

import numpy as np
import pytest

from a1_tracker.trackers.ultralytics_mot import UltralyticsMOTTracker
from a1_tracker.types import Detection

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "T4": "tracker_t4_tracktrack_no_reid.yaml",
    "T5": "tracker_t5_tracktrack_reid.yaml",
    "T6": "tracker_t6_ocsort_native.yaml",
    "T7": "tracker_t7_deepocsort_no_reid.yaml",
    "T8": "tracker_t8_deepocsort_reid.yaml",
    "T9": "tracker_t9_fasttrack_native.yaml",
}


@pytest.mark.parametrize("group", CONFIGS)
def test_empty_frame_coordinates_scores_and_reset(group, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(ROOT / "outputs" / ".ultralytics"))
    tracker = UltralyticsMOTTracker(ROOT / "configs" / CONFIGS[group])
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    empty, stats = tracker.update_with_stats([], image)
    assert empty == [] and stats["high"] == stats["low"] == stats["ignored"] == 0
    detections = [Detection((10, 20, 110, 220), 0.95), Detection((300, 50, 400, 250), 0.15)]
    _, stats = tracker.update_with_stats(detections, image)
    assert stats["high"] == 1
    assert stats["high"] + stats["low"] + stats["ignored"] == 2
    for _ in range(4):
        tracks, _ = tracker.update_with_stats(detections, image)
    assert tracks
    assert all(t.tracker_id > 0 and t.bbox_xyxy[2] > t.bbox_xyxy[0] for t in tracks)
    assert any(abs(t.score - 0.95) < 1e-4 for t in tracks)
    assert any(abs(t.bbox_xyxy[0] - 10) < 5 and abs(t.bbox_xyxy[1] - 20) < 5 for t in tracks)
    if group in {"T5", "T8"}:
        assert tracker.encoder_calls > 0 and tracker.encoded_boxes > 0
        assert tracker.appearance_association_calls > 0
    tracker.reset()
    assert (
        tracker.encoder_calls == tracker.encoded_boxes == tracker.appearance_association_calls == 0
    )
    assert tracker.tracker.frame_id == 0
