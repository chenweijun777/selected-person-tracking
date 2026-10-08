"""Online integration invariants: causal clicks, FPS and frozen appearance modes."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from a1_tracker.joint_runtime import FrameBoxAppearance, JointRuntime, JointTarget, JointTracker
from a1_tracker.types import Detection, Track
from a1_tracker.video_app import apply_action, run_session

ROOT = Path(__file__).resolve().parents[1]


class Encoder:
    def __init__(self):
        self.images = []

    def encode(self, image, tracks):
        self.images.append(image.copy())
        feature = np.zeros(512, dtype=np.float32)
        feature[0] = 1
        return [feature.copy() for _ in tracks]


def target():
    return JointTarget(
        ROOT / "configs/pretrained/target.json",
        FrameBoxAppearance(None, "448_RGB", encoder=Encoder()),
    )


def test_appearance_modes_keep_bgr_at_api_and_frame_local_geometry_keys():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:] = [10, 20, 30]
    tracks = [Track(1, (0, 0, 30, 50), True, 0.9)]
    for mode, expected in [("224_BGR", [10, 20, 30]), ("448_RGB", [10, 20, 30])]:
        encoder = Encoder()
        appearance = FrameBoxAppearance(None, mode, encoder=encoder)
        appearance.encode(image, tracks)
        appearance.encode(image, [Track(99, tracks[0].bbox_xyxy, True, 0.9)])
        assert len(encoder.images) == 1  # same crop, different ID
        assert encoder.images[0][0, 0].tolist() == expected
        appearance.encode(image, [Track(1, (1, 0, 31, 50), True, 0.9)])
        assert len(encoder.images) == 2  # same ID, different crop
        appearance.new_frame()
        appearance.encode(image, tracks)
        assert len(encoder.images) == 3


def test_runtime_uses_rgb448_for_both_reid_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    import a1_tracker.reid.ultralytics_onnx as reid

    modes = []

    class ConfiguredEncoder(Encoder):
        def __init__(self, model, *, mode):
            super().__init__()
            modes.append(mode)

    monkeypatch.setattr(reid, "UltralyticsOnnxAppearance", ConfiguredEncoder)
    runtime = JointRuntime(
        30,
        ROOT / "configs/pretrained/tracker.yaml",
        ROOT / "configs/pretrained/target.json",
        ROOT / "models/yolo26n-reid.onnx",
    )
    assert modes == ["448_RGB", "448_RGB"]
    assert runtime.provenance()["target"]["appearance"] == "448_RGB"


def test_shortlink_yaml_changes_only_short_stage_and_survives_reset(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    cfg = yaml.safe_load((ROOT / "configs/pretrained/tracker.yaml").read_text())
    cfg.update(
        shortlink_cosine_min=0.55,
        shortlink_margin_min=0.08,
        shortlink_confirm=3,
        shortlink_max_young_age_seconds=1.0,
    )
    path = tmp_path / "tracker.yaml"
    path.write_text(yaml.safe_dump(cfg))
    tracker = JointTracker(
        path, None, 14, appearance=FrameBoxAppearance(None, "448_RGB", encoder=Encoder())
    )
    for _ in range(2):
        assert "cosine_min" not in tracker.extension.options
        assert tracker.extension.short_cosine_min == 0.55
        assert tracker.extension.margin_min == 0.08
        assert tracker.extension.confirm == 3
        assert tracker.extension.max_young_age_seconds == 1.0
        assert tracker.base.tracker.max_frames_lost == 70
        assert not tracker.base.settings["with_reid"]
        tracker.reset()


def test_unmatched_birth_does_not_run_residual_rescue(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    tracker = JointTracker(
        ROOT / "configs/pretrained/tracker.yaml",
        None,
        30,
        appearance=FrameBoxAppearance(None, "448_RGB", encoder=Encoder()),
    )
    image = np.zeros((90, 90, 3), np.uint8)
    original = Detection((10, 10, 40, 80), 0.9)
    for _ in range(4):
        old_id = tracker.update([original], image)[0].tracker_id
    assert len(tracker.extension.gallery[old_id]) == 2
    for _ in range(6):
        tracker.update([], image)
    # Score fusion rejects this native match. Its IoU and appearance would
    # satisfy the removed residual rescue, while its Lost age is below .5 s.
    result = tracker.update([Detection((25, 10, 55, 80), 0.4)], image)
    assert all(t.tracker_id != old_id for t in result)
    assert any(t.track_id == old_id for t in tracker.base.tracker.lost_stracks)
    assert not any(r["stage"].startswith("lost_") for r in tracker.last_logs)
    assert not hasattr(tracker.extension, "identity_cost")


def test_paused_clicks_do_not_add_recovery_observations_or_advance_motion_twice():
    subject = target()
    image = np.zeros((90, 90, 3), np.uint8)
    first = Track(7, (0, 0, 30, 70), True, 0.9)
    subject.new_frame(0.0, np.eye(2, 3), True)
    assert subject.select_track(first, image)
    result = subject.update([first], 0.0, image)
    assert subject.update([first], 0.0, image) is result
    assert subject.policy._bank_frame == 1
    before = len(subject.policy.history)
    assert not apply_action(
        subject,
        [first, Track(8, first.bbox_xyxy, True, 0.9)],
        image,
        {"action": "click", "x": 10, "y": 10},
    )["success"]
    assert subject.update([first], 0.0, image) is result
    assert len(subject.policy.history) == before
    # Lost then new identity; refreshing the same paused frame cannot confirm it.
    subject.new_frame(0.1, np.eye(2, 3), True)
    subject.update([], 0.1, image)
    subject.new_frame(0.2, np.eye(2, 3), True)
    other = Track(9, (0, 0, 30, 70), True, 0.9)
    result = subject.update([other], 0.2, image)
    count = subject.policy.candidate_observations
    for _ in range(10):
        assert subject.update([other], 0.2, image) is result
    assert subject.policy.candidate_observations == count
    assert subject.tracker_id == 7


def test_search_timeout_uses_media_time_and_reselection_starts_fresh_identity():
    subject = target()
    image = np.zeros((90, 90, 3), np.uint8)
    track = Track(7, (0, 0, 30, 70), True, 0.9)
    subject.new_frame(0.0)
    assert subject.select_track(track, image)
    old_session = subject.session_target_id
    subject.update([track], 0.0, image)
    subject.new_frame(6.0)
    assert subject.update([], 6.0, image).state.value == "LOST"
    subject.new_frame(6.01)
    assert subject.update([], 6.01, image).state.value == "NEED_RESELECT"
    small = Track(8, (0, 0, 5, 10), True, 0.9)
    assert not subject.select_track(small, image)
    assert subject.session_target_id == old_session
    assert subject.select_track(track, image)
    assert subject.session_target_id != old_session
    assert subject.policy.recoveries == 0
    assert subject.update([track], 6.01, image).valid
    subject.reset()
    assert subject.update([track], 6.01, image).state.value == "UNSELECTED"


@pytest.mark.parametrize("fps,buffer", [(14.0, 70), (30.0, 150), (25.0, 125), (29.97, 150)])
def test_tracker_retention_and_reset_reattach_single_hooks(fps, buffer, tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    pytest.importorskip("ultralytics")
    tracker = JointTracker(
        ROOT / "configs/pretrained/tracker.yaml",
        None,
        fps,
        appearance=FrameBoxAppearance(None, "448_RGB", encoder=Encoder()),
    )
    assert tracker.base.tracker.max_frames_lost == buffer
    image = np.zeros((90, 90, 3), np.uint8)
    detection = Detection((10, 10, 40, 80), 0.9)
    first = tracker.update([detection], image)
    assert len(first) == 1
    old = tracker.extension
    tracker.reset()
    assert tracker.extension is not old
    assert not tracker.extension.gallery and not tracker.extension.pending
    assert tracker.extension.original_first.__self__ is tracker.base.tracker
    assert tracker.base.tracker._first_association.__self__ is tracker.extension
    assert len(tracker.update([detection], image)) == 1


def test_joint_session_exports_configuration_motion_and_state_diagnostics(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "sequence"
    (source / "img1").mkdir(parents=True)
    (source / "seqinfo.ini").write_text("[Sequence]\nframeRate=1\nseqLength=8\nimDir=img1\n")
    for frame in range(1, 9):
        cv2.imwrite(str(source / f"img1/{frame:06d}.jpg"), np.zeros((90, 90, 3), np.uint8))

    class Detector:
        def detect(self, packet):
            return []

    class Tracker:
        def reset(self):
            self.frame = 0

        def update(self, detections, image):
            self.frame += 1
            return [Track(7, (0, 0, 30, 70), True, 0.9)] if self.frame == 1 else []

    subject = target()

    def begin(image, stamp):
        subject.new_frame(stamp)

    runtime = SimpleNamespace(
        tracker=Tracker(),
        target=subject,
        begin_frame=begin,
        motion_warp=np.eye(2, 3),
        motion_info={
            "valid": False,
            "reason": "test",
            "matches": 0,
            "inlier_ratio": 0.0,
            "residual_px": None,
        },
        provenance=lambda: {
            "target_config": {"min_recovery_similarity": 0.65, "search_seconds": 6}
        },
        diagnostics=lambda: {"target": subject.diagnostics()},
    )
    output = tmp_path / "joint"
    summary = run_session(
        str(source),
        output,
        Detector(),
        None,
        runtime_factory=lambda fps: runtime,
        actions=[{"frame": 1, "action": "click", "x": 10, "y": 10}],
    )
    rows = [json.loads(r) for r in (output / "target.jsonl").read_text().splitlines()]
    assert rows[0]["state"] == "TRACKING"
    assert rows[-1]["state"] == "NEED_RESELECT"
    assert len((output / "diagnostics.jsonl").read_text().splitlines()) == 8
    assert len((output / "camera_motion.csv").read_text().splitlines()) == 9
    assert summary["runtime"]["target_config"]["search_seconds"] == 6
    assert summary["complete_source"]
