"""Disk-backed GUI frames and a media clock independent of inference speed."""

from __future__ import annotations

import math
from pathlib import Path
from threading import Event
from time import perf_counter

from a1_tracker.gui_control import GuiControl


class FrameStore:
    """One image in memory; completed frames use an all-keyframe MJPEG file."""

    def __init__(self, path):
        self.path = Path(path)
        self.items = {}
        self.pending = None
        self.writer = None
        self.reader = None
        self.reader_frame = 0
        self.fps = None
        self.finished = False

    def capture(self, canvas, values):
        import cv2

        if self.finished:
            raise RuntimeError("Frame store already closed")
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fps = values["input_fps"]
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.fps,
                (canvas.shape[1], canvas.shape[0]),
            )
            if not self.writer.isOpened():
                raise RuntimeError("无法创建视频回放缓存")
        frame = values["frame"]
        if self.pending is not None and self.pending[1]["frame"] != frame:
            self._flush()
        self.pending = canvas, dict(values)

    def _flush(self):
        if self.pending is not None:
            canvas, values = self.pending
            if values["frame"] != len(self.items) + 1:
                raise RuntimeError("Non-contiguous playback frames")
            self.writer.write(canvas)
            self.items[values["frame"]] = values
            self.pending = None

    def finish(self):
        if not self.finished:
            try:
                self._flush()
            finally:
                if self.writer is not None:
                    self.writer.release()
                self.finished = True

    def read(self, frame, *, paused=True):
        import cv2

        if not self.finished:
            raise RuntimeError("Playback requires a completed buffer")
        if frame not in self.items:
            raise IndexError(frame)
        if self.reader is None:
            self.reader = cv2.VideoCapture(str(self.path))
            if not self.reader.isOpened():
                raise RuntimeError("无法打开视频回放缓存")
        if frame != self.reader_frame + 1:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
        ok, canvas = self.reader.read()
        if not ok:
            raise RuntimeError(f"无法读取回放第{frame}帧")
        self.reader_frame = frame
        return {**self.items[frame], "paused": paused, "canvas": canvas}

    def close_reader(self):
        if self.reader is not None:
            self.reader.release()
            self.reader = None
            self.reader_frame = 0


class BufferedGuiControl(GuiControl):
    """Retain the same causal inference loop; playback consumes its final frames."""

    def __init__(self, path, replay_actions=()):
        super().__init__()
        self.store = FrameStore(path)
        self.preparing = Event()
        self.frame = None
        self.replay_actions = {}
        self.injected = set()
        for action in replay_actions:
            self.replay_actions.setdefault(action["frame"], []).append(dict(action))
        if replay_actions:
            self.preparing.set()

    def send(self, action, **values):
        if action == "play":
            self.preparing.set()
        super().send(action, **values)

    def present(self, canvas, **values):
        self.frame = values["frame"]
        self.store.capture(canvas, values)
        super().present(canvas, **values)

    def poll(self, timeout=0.0):
        commands = super().poll(0.0 if self.preparing.is_set() else timeout)
        if any(c["action"] == "stop" for c in commands):
            return commands
        if self.frame not in self.injected:
            commands = self.replay_actions.get(self.frame, []) + commands
            self.injected.add(self.frame)
        if self.preparing.is_set():
            commands.append({"action": "play"})
        return commands


class PlaybackClock:
    """Original FPS and wall time determine the displayed frame, never model FPS."""

    def __init__(self, fps, count, now=perf_counter):
        if not math.isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError("Invalid playback FPS or frame count")
        self.fps, self.count, self.now = fps, count, now
        self.frame = 1
        self.playing = False
        self.anchor = None
        self.start_frame = 1

    def play(self):
        if not self.playing:
            if self.frame == self.count:
                self.frame = 1
            self.start_frame, self.anchor = self.frame, self.now()
            self.playing = True

    def due(self):
        if self.playing:
            elapsed = max(0.0, self.now() - self.anchor)
            self.frame = min(self.count, self.start_frame + int(elapsed * self.fps + 1e-8))
            if elapsed >= (self.count - self.start_frame + 1) / self.fps:
                self.playing = False
        return self.frame

    def pause(self):
        self.due()
        self.playing = False

    def step(self):
        self.pause()
        self.frame = min(self.count, self.frame + 1)
        return self.frame
