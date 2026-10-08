import json
from types import SimpleNamespace

import numpy as np
import pytest

from a1_tracker.buffered_playback import BufferedGuiControl, FrameStore, PlaybackClock
from a1_tracker.types import TargetState
from a1_tracker.video_app import run_session


@pytest.mark.parametrize("fps", [14.0, 25.0, 30.0, 29.97, 60.0])
def test_clock_uses_original_fps_and_keeps_full_last_frame_duration(fps):
    stamp = [100.0]
    clock = PlaybackClock(fps, 300, now=lambda: stamp[0])
    clock.play()
    stamp[0] += 2.0
    assert clock.due() == 1 + int(2 * fps + 1e-8)
    clock.pause()
    paused = clock.frame
    stamp[0] += 200.0
    assert clock.due() == paused
    clock.play()
    stamp[0] += 1 / fps
    assert clock.due() == paused + 1
    clock.pause()
    clock.frame = 299
    clock.play()
    stamp[0] += 1 / fps
    assert clock.due() == 300 and clock.playing
    stamp[0] += 1 / fps + 1e-8
    assert clock.due() == 300 and not clock.playing


def test_render_delay_changes_preview_frame_without_slowing_media_clock():
    stamp = [0.0]
    clock = PlaybackClock(30.0, 300, now=lambda: stamp[0])
    clock.play()
    stamp[0] = 3.2
    assert clock.due() == 97
    assert clock.step() == 98 and not clock.playing


def test_store_writes_final_paused_selection_once_and_random_reads_correct_frames(tmp_path):
    pytest.importorskip("cv2")
    store = FrameStore(tmp_path / "缓存.avi")
    values = {"frame": 1, "input_fps": 29.97, "paused": True, "timestamp": 0.0}
    store.capture(np.zeros((64, 96, 3), np.uint8), values)
    store.capture(np.full((64, 96, 3), 30, np.uint8), {**values, "selected": True})
    for frame in range(2, 7):
        store.capture(np.full((64, 96, 3), frame * 30, np.uint8), {**values, "frame": frame})
    store.finish()
    assert len(store.items) == 6
    assert store.items[1]["selected"]
    for frame in (6, 1, 3, 4, 2):
        got = store.read(frame)
        assert abs(float(got["canvas"].mean()) - frame * 30) < 3
        assert got["frame"] == frame
    store.close_reader()
    store.finish()  # repeat cleanup is harmless


def test_buffered_replay_preserves_pipeline_frames_and_reselection_actions(tmp_path):
    from test_tracking_gui import Detector, Tracker, make_video

    actions = [
        {"frame": 1, "action": "click", "x": 10, "y": 30, "replayed": True},
        {"frame": 3, "action": "reset"},
    ]
    control = BufferedGuiControl(tmp_path / "session/preview.avi", actions)
    summary = run_session(
        str(make_video(tmp_path)),
        tmp_path / "session",
        Detector(),
        Tracker(),
        controller=control,
        save_video=False,
    )
    control.store.finish()
    assert summary["frames"] == 4 and summary["complete_source"]
    rows = [json.loads(v) for v in (tmp_path / "session/target.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["TRACKING", "TRACKING", "UNSELECTED", "UNSELECTED"]
    assert len(control.store.items) == 4
    assert control.store.items[3]["result"].state.value == "UNSELECTED"
    exported = json.loads((tmp_path / "session/actions.json").read_text())
    assert exported[0]["replayed"] and "replayed" not in exported[1]


def test_gui_model_and_configs_are_the_named_finetuned_profile(monkeypatch, tmp_path):
    from a1_tracker import joint_runtime, tracking_gui, video_app
    from a1_tracker.detectors import ultralytics

    calls = {}

    def detector(model, *args, **kwargs):
        calls["model"] = model
        calls["args"] = args
        calls["detector_options"] = kwargs
        return "detector"

    def runtime(fps, tracker, target, appearance):
        calls["runtime"] = fps, tracker, target, appearance
        return "runtime"

    def session(*args, **kwargs):
        assert kwargs["runtime_factory"](29.97) == "runtime"
        calls["provenance"] = kwargs["provenance"]
        return {"passed": True}

    monkeypatch.setattr(ultralytics, "UltralyticsDetector", detector)
    monkeypatch.setattr(joint_runtime, "JointRuntime", runtime)
    monkeypatch.setattr(video_app, "run_session", session)
    assert tracking_gui.run_gui_session("input.mp4", tmp_path / "out", True, None)["passed"]
    assert calls["model"].endswith("models\\best.pt") or calls["model"].endswith("models/best.pt")
    assert calls["args"] == (960, 0.001, 0.7)
    assert calls["detector_options"]["nms"]
    fps, tracker, target, appearance = calls["runtime"]
    assert fps == 29.97
    assert tracker.parts[-2:] == ("finetuned", "tracker.yaml")
    assert target.parts[-2:] == ("finetuned", "target.json")
    assert appearance.name == "yolo26n-reid.onnx"
    assert calls["provenance"]["profile"] == "finetuned26s_20261007"


def test_tk_buffered_playback_and_reselect_at_displayed_frame(tmp_path, monkeypatch):
    import time

    from test_tracking_gui import Detector, Tracker, make_video

    tk = pytest.importorskip("tkinter")
    from a1_tracker.tracking_gui import TrackingGUI

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()

    def runner(source, output, save_video, control):
        return run_session(
            str(source), output, Detector(), Tracker(), save_video=save_video, controller=control
        )

    gui = TrackingGUI(root, runner=runner, output_root=tmp_path / "outputs")
    monkeypatch.setattr(
        "a1_tracker.tracking_gui.messagebox.showerror", lambda *a, **kw: pytest.fail(str(a))
    )

    def until(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline
            root.update()
            time.sleep(0.005)

    try:
        gui.load_video(make_video(tmp_path, frames=12))
        gui.start()
        until(lambda: gui.ready)
        gui.viewport = (0, 0, 96, 64)
        gui._click(SimpleNamespace(x=10, y=30))
        until(lambda: gui.shown["result"].valid)
        gui.toggle_play()
        until(lambda: gui.clock is not None)
        assert gui.clock.fps == 10.0
        gui.toggle_play()  # pause original-rate playback
        assert gui.shown["paused"]
        gui.step()
        assert gui.shown["frame"] == 2
        previous_output = gui.output
        gui.reset()  # recompute with original click + new reset at frame 2
        until(lambda: gui.clock is not None)
        assert gui.output != previous_output
        assert gui.clock.frame == 2 and not gui.clock.playing
        assert gui.shown["result"].state == TargetState.UNSELECTED
        actions = json.loads((gui.output / "actions.json").read_text())
        assert [(a["frame"], a["action"]) for a in actions] == [(1, "click"), (2, "reset")]
        assert gui.replay_actions[-1]["action"] == "reset"
        assert len(gui.buffer.items) == 12
    finally:
        if gui.worker is not None:
            gui.control.send("stop")
            gui.worker.join(timeout=10)
        gui._close_playback()
        root.destroy()
