import json
from pathlib import Path

import numpy as np

from a1_tracker.gui_analysis_cache import (
    AnalysisCache,
    CachedDetector,
    CachedRuntime,
    PersistentAppearance,
    RecordingRuntime,
)
from a1_tracker.identity_events import IdentityFrame, IdentityRemap
from a1_tracker.joint_runtime import FrameBoxAppearance, JointTarget
from a1_tracker.types import Track
from a1_tracker.video_app import run_session

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/finetuned/target.json"


class Encoder:
    def __init__(self):
        self.calls = 0

    def encode(self, image, tracks):
        self.calls += 1
        return [np.eye(1, 512, dtype=np.float32).ravel() for _ in tracks]


def test_features_reuse_exact_frame_and_crop_but_never_id(tmp_path):
    from test_tracking_gui import make_video

    cache = AnalysisCache(tmp_path / "analysis.sqlite", make_video(tmp_path))
    encoder = Encoder()
    features = PersistentAppearance(cache, encoder)
    features.frame = 1
    track = Track(7, (0.0, 0.0, 30.0, 60.0), True, 0.9)
    image = np.zeros((64, 96, 3), np.uint8)
    expected = features.encode(image, [track])[0]
    got = features.encode(image, [Track(88, track.bbox_xyxy, True, 0.9)])[0]
    np.testing.assert_array_equal(expected, got)
    assert encoder.calls == 1
    features.encode(image, [Track(7, (1.0, 0.0, 31.0, 60.0), True, 0.9)])
    assert encoder.calls == 2
    features.frame = 2
    features.encode(image, [track])
    assert encoder.calls == 3
    cache.close()


class NativeTracker:
    def __init__(self):
        self.calls = 0
        self.generation = 5
        self.reset()
        self.last_logs, self.last_gallery = [], []

    def reset(self):
        self.frame = 0
        self.generation += 1
        self.identity_frame = IdentityFrame(self.generation, 0)

    def update(self, detections, image):
        self.calls += 1
        self.frame += 1
        box = (0.0, 0.0, 30.0, 60.0)
        remaps = (IdentityRemap(3, 7, 8, box),) if self.frame == 3 else ()
        self.identity_frame = IdentityFrame(self.generation, self.frame, remaps)
        return [Track(7 if self.frame < 3 else 8, box, True, 0.9)]


class NativeRuntime:
    def __init__(self, encoder):
        self.tracker = NativeTracker()
        self.target = JointTarget(
            CONFIG,
            FrameBoxAppearance(None, "448_RGB", encoder=encoder),
            identity_source=lambda: self.tracker.identity_frame,
        )
        self.motion_warp = np.eye(2, 3)
        self.motion_info = {
            "valid": True,
            "reason": "test",
            "matches": 40,
            "inlier_ratio": 1.0,
            "residual_px": 0.0,
        }
        self.motion_calls = 0

    def begin_frame(self, image, timestamp):
        self.motion_calls += 1
        self.target.new_frame(
            timestamp,
            self.motion_warp,
            True,
            frame_id=self.tracker.frame + 1,
            tracker_generation=self.tracker.generation,
        )

    def diagnostics(self):
        return {"target": self.target.diagnostics()}

    def provenance(self):
        return {
            "appearance": {"tracker": {}, "target": self.target.appearance.stats()},
            "config_files": {str(ROOT / "models/yolo26n-reid.onnx"): "test_model"},
        }


def test_cached_reselection_matches_fresh_a2_and_delivers_handoff_without_native_calls(
    tmp_path, monkeypatch
):
    from test_tracking_gui import Detector, make_video

    source = make_video(tmp_path)
    output = tmp_path / "first"
    cache = AnalysisCache(output / "analysis.sqlite", source)
    encoder = Encoder()
    native = NativeRuntime(encoder)
    actions = [{"frame": 1, "action": "click", "x": 10, "y": 30}]
    summary = run_session(
        str(source),
        output,
        Detector(),
        None,
        save_video=False,
        runtime_factory=lambda fps: RecordingRuntime(native, cache),
        actions=actions,
    )
    cache.finish(summary)
    original_calls = encoder.calls
    assert native.tracker.calls == native.motion_calls == 4
    replay = CachedRuntime(cache, CONFIG)
    second = tmp_path / "second"
    cached_summary = run_session(
        str(source),
        second,
        CachedDetector(cache),
        None,
        save_video=False,
        runtime_factory=lambda fps: replay,
        actions=actions,
    )
    assert native.tracker.calls == native.motion_calls == 4
    assert encoder.calls == original_calls
    assert replay.encoder.hits > 0 and replay.encoder.misses == 0
    assert cached_summary["runtime"]["analysis_cache"]["detection_calls"] == 0
    for name in ("target.jsonl", "tracks.txt", "camera_motion.csv"):
        if name == "target.jsonl":

            def rows(folder, name=name):
                data = [json.loads(v) for v in (folder / name).read_text().splitlines()]
                for row in data:
                    row.pop("session_target_id")
                return data

            assert rows(output) == rows(second)
        else:
            assert (output / name).read_bytes() == (second / name).read_bytes()
    assert replay.cache.read(3)["identity"]["remaps"][0]["old_id"] == 8
    cached_rows = [json.loads(v) for v in (second / "target.jsonl").read_text().splitlines()]
    assert cached_rows[2]["tracker_id"] == 8 and cached_rows[2]["state"] == "TRACKING"
    cache.close()
    reopened = AnalysisCache(tmp_path / "unused.sqlite", source)
    assert reopened.restore(output / "analysis.sqlite")
    assert reopened.runtime is None and reopened.encoder is None

    def forbidden_model(*args, **kwargs):
        raise AssertionError("Known appearance features must not reload ONNX")

    monkeypatch.setattr(
        "a1_tracker.reid.ultralytics_onnx.UltralyticsOnnxAppearance", forbidden_model
    )
    restored_runtime = CachedRuntime(reopened, CONFIG)
    restored_output = tmp_path / "restored"
    restored_summary = run_session(
        str(source),
        restored_output,
        CachedDetector(reopened),
        None,
        save_video=False,
        runtime_factory=lambda fps: restored_runtime,
        actions=actions,
    )
    assert restored_summary["runtime"]["analysis_cache"]["appearance_misses"] == 0
    assert (restored_output / "tracks.txt").read_bytes() == (output / "tracks.txt").read_bytes()
    reopened.close()
