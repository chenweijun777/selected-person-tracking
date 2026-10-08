import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from a1_tracker.gui_control import GuiControl, image_point, image_viewport
from a1_tracker.types import Track
from a1_tracker.video_app import run_session


def test_letterbox_mapping_matches_actual_rounded_preview():
    viewport = image_viewport((1920, 1080), (1001, 801))
    left, top, width, height = viewport
    assert width == 1001 and height == 563
    assert image_point((left + width / 2, top + height / 2), (1920, 1080), viewport) == (960, 540)
    assert image_point((0, 0), (1920, 1080), viewport) is None
    assert image_point((left + width, top), (1920, 1080), viewport) is None
    with pytest.raises(ValueError):
        image_viewport((0, 1), (1, 1))


def test_latest_preview_is_bounded_and_commands_preserve_order():
    control = GuiControl()
    for frame in range(100):
        control.present(None, frame=frame)
    assert control.frames.qsize() == 1
    assert control.frames.get_nowait()["frame"] == 99
    control.send("pause")
    control.send("click", frame=99, x=10, y=20)
    assert [v["action"] for v in control.poll()] == ["pause", "click"]
    assert control.poll() == []


def make_video(tmp_path, frames=4):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    assert writer.isOpened()
    for frame in range(frames):
        writer.write(np.full((64, 96, 3), frame * 20, np.uint8))
    writer.release()
    return path


class Detector:
    def detect(self, packet):
        return []


class Tracker:
    def reset(self):
        pass

    def update(self, detections, image):
        return [Track(7, (0, 0, 30, 60), True, 0.9)]


def test_gui_controller_pause_click_step_stale_reset_and_stop(tmp_path):
    source = make_video(tmp_path)
    control = GuiControl()
    output = tmp_path / "session"
    result = {}

    def work():
        try:
            result["summary"] = run_session(
                str(source), output, Detector(), Tracker(), controller=control
            )
        except Exception as exc:  # noqa: BLE001 - propagate worker failures into test assertions
            result["error"] = exc

    worker = threading.Thread(target=work)
    worker.start()
    try:
        first = control.frames.get(timeout=10)
        assert first["frame"] == 1 and first["paused"]
        control.send("click", frame=1, x=10, y=30)
        assert control.events.get(timeout=10)["success"]
        selected = control.frames.get(timeout=10)
        assert selected["result"].state.value == "TRACKING"
        control.send("step")
        while True:
            second = control.frames.get(timeout=10)
            if second["frame"] == 2:
                break
        assert second["paused"]
        control.send("click", frame=1, x=10, y=30)
        rejected = control.events.get(timeout=10)
        assert rejected["reason"] == "stale_frame_input" and not rejected["success"]
        control.send("reset", frame=2)
        assert control.events.get(timeout=10)["reason"] == "selection_cleared"
        control.send("play")
        worker.join(timeout=10)
        assert not worker.is_alive()
    finally:
        control.send("stop")
        worker.join(timeout=10)
    assert "error" not in result
    assert result["summary"]["frames"] == 4
    assert result["summary"]["desktop_gui"]
    rows = [json.loads(row) for row in (output / "target.jsonl").read_text().splitlines()]
    assert [row["state"] for row in rows] == ["TRACKING", "UNSELECTED", "UNSELECTED", "UNSELECTED"]
    assert [row["timestamp"] for row in rows] == [0, 0.1, 0.2, 0.3]
    assert len(json.loads((output / "actions.json").read_text())) == 3


def test_gui_stop_from_initial_paused_frame_saves_partial_session(tmp_path):
    control = GuiControl()
    control.send("stop")
    output = tmp_path / "session"
    summary = run_session(
        str(make_video(tmp_path)), output, Detector(), Tracker(), controller=control
    )
    assert summary["operator_stopped"] and summary["frames"] == 1
    assert not summary["complete_source"]
    assert (output / "annotated.mp4").is_file()


def test_tk_video_picker_click_mapping_and_safe_worker_completion(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("PIL")
    from a1_tracker.tracking_gui import TrackingGUI

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    source = make_video(tmp_path)

    def runner(source, output, save_video, control):
        return run_session(
            str(source), output, Detector(), Tracker(), save_video=save_video, controller=control
        )

    gui = TrackingGUI(root, runner=runner, output_root=tmp_path / "outputs")
    monkeypatch.setattr(
        "a1_tracker.tracking_gui.filedialog.askopenfilename", lambda **kw: str(source)
    )
    monkeypatch.setattr(
        "a1_tracker.tracking_gui.messagebox.showerror", lambda *a, **kw: pytest.fail(str(a))
    )
    try:
        gui.choose_video()
        assert gui.source == source
        assert gui.image.size == (96, 64)
        gui.start()
        item = gui.control.frames.get(timeout=10)
        gui._accept_frame(item)
        gui.viewport = image_viewport(gui.image.size, (1001, 801))
        left, top, width, height = gui.viewport
        gui._click(SimpleNamespace(x=left + width * 10 / 96, y=top + height * 30 / 64))
        assert gui.control.events.get(timeout=10)["success"]
        gui.stop()
        gui.worker.join(timeout=10)
        assert not gui.worker.is_alive()
        gui._poll()
        assert gui.worker is None and not gui.ready
        assert "已保存" in gui.status.get()
        assert (gui.output / "summary.json").is_file()
    finally:
        if gui.worker is not None:
            gui.control.send("stop")
            gui.worker.join(timeout=10)
        root.destroy()
