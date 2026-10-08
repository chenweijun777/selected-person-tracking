import json

import numpy as np
import pytest

from a1_tracker.target.baselines import TargetBaseline
from a1_tracker.types import Detection, Track
from a1_tracker.video_app import FrameSource, apply_action, canonical_detections, run_session


def test_click_loss_ambiguous_reselection_and_reset_are_atomic():
    target = TargetBaseline("A0")
    first = Track(7, (0, 0, 20, 40), True, 0.9)
    other = Track(9, (10, 0, 30, 40), True, 0.9)
    assert apply_action(target, [first], None, {"action": "click", "x": 5, "y": 5})["success"]
    session = target.session_target_id
    assert not target.update([other], 1).valid
    assert not apply_action(target, [first, other], None, {"action": "click", "x": 15, "y": 5})[
        "success"
    ]
    assert target.session_target_id == session
    assert apply_action(target, [other], None, {"action": "click", "x": 25, "y": 5})["success"]
    assert target.session_target_id != session
    assert target.update([other], 2).tracker_id == 9
    apply_action(target, [], None, {"action": "reset"})
    assert target.update([other], 3).state.value == "UNSELECTED"


def test_cache_quantization_handles_threshold_and_degenerate_box():
    values = [
        Detection((0.0006, 0, 10.0012, 20), 0.2499998),
        Detection((0, 0, 0.0001, 1), 0.9),
        Detection((0, 0, 2, 3), 0.0994),
    ]
    got = canonical_detections(values, 0.1)
    assert len(got) == 1
    assert got[0].score == 0.25
    assert got[0].bbox_xyxy == (0.001, 0, 10.001999999999999, 20)


def test_session_exports_every_frame_loss_reselect_and_refuses_overwrite(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "sequence"
    (source / "img1").mkdir(parents=True)
    (source / "seqinfo.ini").write_text("[Sequence]\nframeRate=10\nseqLength=4\nimDir=img1\n")
    for i in range(1, 5):
        cv2.imwrite(str(source / f"img1/{i:06d}.jpg"), np.zeros((64, 96, 3), dtype=np.uint8))

    class Detector:
        def detect(self, frame):
            return []

    class Tracker:
        def reset(self):
            self.frame = 0

        def update(self, detections, image):
            self.frame += 1
            return (
                []
                if self.frame == 2
                else [Track(7 if self.frame == 1 else 8, (0, 0, 30, 50), True, 0.9)]
            )

    output = tmp_path / "session"
    summary = run_session(
        str(source),
        output,
        Detector(),
        Tracker(),
        save_video=True,
        actions=[
            {"frame": 1, "action": "click", "x": 10, "y": 10},
            {"frame": 3, "action": "click", "x": 10, "y": 10},
            {"frame": 4, "action": "reset"},
        ],
    )
    rows = [json.loads(s) for s in (output / "target.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["TRACKING", "LOST", "TRACKING", "UNSELECTED"]
    assert [r["timestamp"] for r in rows] == [0, 0.1, 0.2, 0.3]
    assert summary["complete_source"] and summary["frames"] == 4
    assert summary["latency"]["active_total_ms"]["p95_ms"] > 0
    replay = FrameSource(str(output / "annotated.mp4"))
    try:
        timestamps = [replay.read()[1] for _ in range(4)]
        assert timestamps == [0, 0.1, 0.2, 0.3]
        assert replay.read() is None
    finally:
        replay.close()
    with pytest.raises(FileExistsError):
        run_session(str(source), output, Detector(), Tracker())
