"""Desktop video tester. Tk stays on the main thread; inference owns its state."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from a1_tracker.buffered_playback import BufferedGuiControl, PlaybackClock
from a1_tracker.gui_control import image_point, image_viewport

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
STATES = {
    "UNSELECTED": "未选择人物",
    "TRACKING": "正在跟踪",
    "LOST": "正在搜索原人物",
    "VERIFYING": "正在确认恢复",
    "NEED_RESELECT": "搜索超时，请重新选择",
}
REASONS = {
    "selected": "已选中人物，可以继续播放。",
    "selection_cleared": "已清除选择，请点击人物框重新选择。",
    "ambiguous_click": "点击处有重叠人物，请点击框内不重叠的位置。",
    "no_observed_track_at_click": "点击处没有检测到人物，请点击灰色人物框内。",
    "stale_frame_input": "画面已经更新，请在暂停画面上重新点击。",
}


def run_gui_session(source, output, save_video, control):
    """Use the same detector, runtime and exporters as the formal video entry."""
    import torch

    from a1_tracker.detectors.ultralytics import UltralyticsDetector
    from a1_tracker.gui_analysis_cache import (
        CachedDetector,
        CachedRuntime,
        RecordingRuntime,
        source_token,
    )
    from a1_tracker.joint_runtime import JointRuntime
    from a1_tracker.video_app import file_sha, run_session

    model = ROOT / "models/best.pt"
    tracker = ROOT / "configs/finetuned/tracker.yaml"
    target = ROOT / "configs/finetuned/target.json"
    appearance = ROOT / "models/yolo26n-reid.onnx"
    for path in (model, tracker, target, appearance):
        if not path.is_file():
            raise FileNotFoundError(f"缺少本地模型或配置：{path}")
    device = "0" if torch.cuda.is_available() else "cpu"
    cache = getattr(control, "analysis_cache", None)
    index = None
    if cache is not None:
        # Include code as well as frozen inputs; cosmetic GUI changes do not
        # invalidate the causal analysis, but algorithm changes do.
        sources = sorted((ROOT / "src/a1_tracker").rglob("*.py"))
        sources = [
            p
            for p in sources
            if p.name not in ("tracking_gui.py", "buffered_playback.py", "gui_control.py")
        ]
        signature = [
            source_token(source),
            [file_sha(p) for p in (model, tracker, target, appearance, *sources)],
        ]
        key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        index = ROOT / "outputs/gui_cache_index" / f"{key}.json"
        if not cache.complete and index.is_file():
            cache.restore(json.loads(index.read_text(encoding="utf-8"))["path"])
    reuse = cache is not None and cache.complete
    if control is not None:
        control.feedback({"action": "cache_status", "reused": reuse})
    if reuse:
        if source_token(source) != cache.source:
            raise RuntimeError("视频文件已变化，请重新选择视频建立缓存")
        expected = {**cache.config_hashes, str(model): cache.provenance["model_sha256"]}
        if any(file_sha(path) != digest for path, digest in expected.items()):
            raise RuntimeError("模型或配置已变化，请重新选择视频建立缓存")
        detector = CachedDetector(cache)
    else:
        detector = UltralyticsDetector(str(model), 960, 0.001, 0.7, device=device, nms=True)

    def runtime_factory(fps):
        if reuse:
            if fps != cache.fps:
                raise RuntimeError("视频FPS与缓存不一致")
            return CachedRuntime(cache, target)
        runtime = JointRuntime(fps, tracker, target, appearance)
        return RecordingRuntime(runtime, cache) if cache is not None else runtime

    provenance = {
        "model": str(model),
        "model_sha256": file_sha(model),
        "tracker": str(tracker),
        "tracker_sha256": file_sha(tracker),
        "application_sha256": file_sha(ROOT / "src/a1_tracker/video_app.py"),
        "gui_sha256": file_sha(__file__),
        "imgsz": 960,
        "inference_floor": 0.001,
        "detection_floor": 0.001,
        "strategy": "A2_v7 + ShortLink2",
        "device": device,
        "actions_source": "desktop_gui",
        "actions_sha256": None,
        "profile": "finetuned26s_20261007",
        "playback": "buffered_original_fps",
        "analysis_cache_reused": reuse,
        "analysis_cache_sha256": file_sha(ROOT / "src/a1_tracker/gui_analysis_cache.py"),
    }
    summary = run_session(
        str(source),
        output,
        detector,
        None,
        floor=0.001,
        save_video=save_video,
        provenance=provenance,
        controller=control,
        runtime_factory=runtime_factory,
    )
    if cache is not None:
        cache.finish(summary)
        if cache.complete and index is not None:
            index.parent.mkdir(parents=True, exist_ok=True)
            temporary = index.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"path": str(cache.path)}), encoding="utf-8")
            temporary.replace(index)
    return summary


class TrackingGUI:
    def __init__(self, root, runner=run_gui_session, output_root=None):
        self.root = root
        self.runner = runner
        self.output_root = Path(output_root or ROOT / "outputs/gui_sessions")
        self.source = None
        self.output = None
        self.worker = None
        self.control = None
        self.pending_source = None
        self.closing = False
        self.stopping = False
        self.ready = False
        self.shown = None
        self.image = None
        self.photo = None
        self.viewport = None
        self.buffer = None
        self.clock = None
        self.return_frame = 1
        self.autoplay = False
        self.replay_actions = []
        self.session_ok = False
        self.analysis_cache = None
        self.filename = tk.StringVar(value="尚未选择视频")
        self.status = tk.StringVar(value="请选择本地视频")
        self.progress = tk.StringVar(value="")
        self.hint = tk.StringVar(value="选择视频 → 开始检测 → 暂停后点击人物框 → 播放")
        self.save_video = tk.BooleanVar(value=True)
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", self._space)
        root.bind("<Right>", self._right)
        root.bind("<Escape>", self._escape)
        self.poll_id = root.after(40, self._poll)
        LOGGER.info(
            "GUI ready: best.pt, finetuned26s_20261007, shared analysis cache, original FPS"
        )

    def _build(self):
        self.root.title("人物跟踪 · 微调模型 · 缓存复用")
        self.root.geometry("1180x820")
        self.root.minsize(920, 680)
        self.root.configure(bg="#eef2f6")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f6")
        style.configure("TLabel", background="#eef2f6", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 8))
        style.configure("TCheckbutton", background="#eef2f6", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="人物跟踪视频测试", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="微调26s（best.pt） · ShortLink → 身份交接 → A2_v7 · 微调后参数 · 原帧率播放",
        ).pack(anchor="w", pady=(4, 12))
        source_row = ttk.Frame(outer)
        source_row.pack(fill="x", pady=(0, 10))
        self.choose_button = ttk.Button(source_row, text="选择视频…", command=self.choose_video)
        self.choose_button.pack(side="left")
        ttk.Label(source_row, textvariable=self.filename, width=58).pack(
            side="left", padx=12, fill="x", expand=True
        )
        self.save_check = ttk.Checkbutton(source_row, text="保存带框视频", variable=self.save_video)
        self.save_check.pack(side="right")
        self.canvas = tk.Canvas(outer, bg="#111827", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<Button-1>", self._click)
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(12, 8))
        self.start_button = ttk.Button(
            toolbar, text="开始检测", command=self.start, state="disabled"
        )
        self.start_button.pack(side="left")
        self.play_button = ttk.Button(
            toolbar, text="播放", command=self.toggle_play, state="disabled"
        )
        self.play_button.pack(side="left", padx=(8, 0))
        self.step_button = ttk.Button(toolbar, text="下一帧", command=self.step, state="disabled")
        self.step_button.pack(side="left", padx=(8, 0))
        self.reset_button = ttk.Button(
            toolbar, text="清除选择", command=self.reset, state="disabled"
        )
        self.reset_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(toolbar, text="结束测试", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        self.output_button = ttk.Button(
            toolbar, text="打开结果目录", command=self.open_output, state="disabled"
        )
        self.output_button.pack(side="right")
        status_row = ttk.Frame(outer)
        status_row.pack(fill="x", pady=(4, 6))
        ttk.Label(status_row, textvariable=self.status, style="Status.TLabel").pack(side="left")
        ttk.Label(status_row, textvariable=self.progress).pack(side="right")
        ttk.Label(outer, textvariable=self.hint, wraplength=1080).pack(anchor="w")
        ttk.Label(
            outer,
            text="空格：播放 / 暂停    →：下一帧    Esc：清除选择    首次播放先处理视频；完成后按视频原帧率播放。",
            foreground="#64748b",
        ).pack(anchor="w", pady=(6, 0))
        self._draw()

    def choose_video(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择要测试的视频",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        if self.worker is not None:
            self.pending_source = Path(path)
            self.stop()
        else:
            self.load_video(Path(path))

    def load_video(self, path):
        from a1_tracker.video_app import FrameSource

        source = None
        try:
            source = FrameSource(str(path))
            packet = source.read()
            if packet is None:
                raise ValueError("视频中没有可读取的画面")
            self.image = Image.fromarray(packet[0][:, :, ::-1])
            total = source.expected
            duration = f"  ·  {total / source.fps:.1f} 秒" if total else ""
            self.progress.set(f"{source.fps:g} FPS{duration}")
        except Exception as exc:
            LOGGER.exception("Cannot open video")
            messagebox.showerror("无法打开视频", str(exc), parent=self.root)
            return
        finally:
            if source is not None:
                source.close()
        self.source = Path(path)
        if self.analysis_cache is not None:
            self.analysis_cache.close()
            self.analysis_cache = None
        self._close_playback()
        self.filename.set(self.source.name)
        self.shown = None
        self.ready = False
        self.status.set("视频已就绪，点击开始检测")
        self.hint.set(str(self.source))
        self.start_button.configure(state="normal")
        for button in (self.play_button, self.step_button, self.reset_button, self.stop_button):
            button.configure(state="disabled")
        self._draw()

    def _close_playback(self):
        if self.buffer is not None:
            self.buffer.close_reader()
        self.buffer = self.clock = None

    def start(self, *, replay_actions=(), return_frame=1):
        if self.worker is not None or self.source is None:
            return
        stem = re.sub(r"[^\w\-]", "_", self.source.stem)[:50] or "video"
        stamp = datetime.now(tz=timezone.utc).astimezone()
        self.output = self.output_root / f"{stem}_{stamp:%Y%m%d_%H%M%S_%f}"
        self._close_playback()
        self.control = BufferedGuiControl(self.output / "preview.avi", replay_actions)
        from a1_tracker.gui_analysis_cache import AnalysisCache

        if self.analysis_cache is None or not self.analysis_cache.complete:
            if self.analysis_cache is not None:
                self.analysis_cache.close()
            self.analysis_cache = AnalysisCache(self.output / "analysis.sqlite", self.source)
        self.control.analysis_cache = self.analysis_cache
        self.return_frame = return_frame
        self.replay_actions = [dict(a) for a in replay_actions if a.get("replayed")]
        self.autoplay = False
        self.session_ok = False
        self.ready = False
        self.stopping = False
        self.shown = None
        self.status.set("正在加载模型和检测首帧…")
        self.hint.set("首帧检测完成后会暂停；点击人物框即可选中，再点击播放。")
        self.start_button.configure(state="disabled")
        self.save_check.configure(state="disabled")
        self.output_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        for button in (self.play_button, self.step_button, self.reset_button):
            button.configure(state="disabled")
        if replay_actions:
            self.status.set("正在使用缓存更新选人状态…")
            self.hint.set(
                "复用检测、轨迹、运动与身份交接缓存，只更新A2状态和绘制；完成后返回暂停帧。"
            )
        source, output, save_video, control = (
            self.source,
            self.output,
            self.save_video.get(),
            self.control,
        )

        def work():
            try:
                summary = self.runner(source, output, save_video, control)
                control.store.finish()
                control.feedback({"action": "finished", "summary": summary})
            except Exception as exc:
                LOGGER.exception("Video session failed")
                try:
                    control.store.finish()
                except Exception:
                    LOGGER.exception("Cannot close playback cache")
                control.feedback({"action": "error", "message": str(exc)})

        self.worker = threading.Thread(target=work, name="tracking-inference", daemon=False)
        self.worker.start()

    def toggle_play(self):
        if not self.ready or self.stopping:
            return
        if self.clock is not None:
            if self.clock.playing:
                self.clock.pause()
            else:
                self.clock.play()
            self._show_playback()
        elif self.shown["paused"]:
            self.return_frame = self.shown["frame"]
            self.autoplay = True
            self.ready = False
            self.control.send("play")
            self.status.set("正在处理视频，准备原帧率播放…")
            self.hint.set("逐帧推理完成后自动以1倍速播放。等待处理不会改变跟踪帧数或参数。")
            for button in (self.play_button, self.step_button, self.reset_button):
                button.configure(state="disabled")

    def step(self):
        if self.ready and not self.stopping:
            if self.clock is not None:
                self.clock.step()
                self._show_playback()
            else:
                self.control.send("step")

    def reset(self):
        if not self.ready or self.stopping:
            return
        if not self.shown["paused"]:
            if self.clock is not None:
                self.clock.pause()
                self._show_playback()
            else:
                self.control.send("pause")
            self.hint.set("正在暂停，请停稳后再次清除选择。")
            return
        self._selection_action("reset")

    def _selection_action(self, action, **values):
        command = {"action": action, "frame": self.shown["frame"], **values}
        if self.clock is not None:
            prior = [
                {**a, "replayed": True}
                for a in self.replay_actions
                if a["frame"] <= command["frame"]
            ]
            self.start(replay_actions=[*prior, command], return_frame=command["frame"])
        else:
            self.control.send(action, frame=self.shown["frame"], **values)

    def stop(self):
        if self.clock is not None:
            self.clock.pause()
            self._show_playback()
            self.status.set("回放已停止，结果已保存")
            return
        if self.worker is None or self.stopping:
            return
        self.stopping = True
        self.control.send("stop")
        self.status.set("正在结束并保存结果…")
        for button in (self.play_button, self.step_button, self.reset_button, self.stop_button):
            button.configure(state="disabled")

    def close(self):
        self.closing = True
        self.choose_button.configure(state="disabled")
        if self.worker is not None:
            self.stop()
        else:
            self._close_playback()
            if self.analysis_cache is not None:
                self.analysis_cache.close()
            self.root.after_cancel(self.poll_id)
            self.root.destroy()

    def open_output(self):
        if self.output and self.output.is_dir():
            os.startfile(str(self.output))

    def _click(self, event):
        if not self.ready or self.stopping:
            return
        if not self.shown["paused"]:
            if self.clock is not None:
                self.clock.pause()
                self._show_playback()
            else:
                self.control.send("pause")
            self.hint.set("正在暂停，停稳后请再次点击人物框。")
            return
        point = image_point((event.x, event.y), self.image.size, self.viewport)
        if point is not None:
            self._selection_action("click", x=point[0], y=point[1])

    def _space(self, event):
        self.toggle_play()
        return "break"

    def _right(self, event):
        self.step()
        return "break"

    def _escape(self, event):
        self.reset()
        return "break"

    def _resize(self, event):
        self._draw()

    def _draw(self):
        self.canvas.delete("all")
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self.image is None:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="选择一个视频，开始测试人物跟踪",
                fill="#94a3b8",
                font=("Microsoft YaHei UI", 15),
            )
            return
        self.viewport = image_viewport(self.image.size, (width, height))
        left, top, shown_width, shown_height = self.viewport
        self.photo = ImageTk.PhotoImage(
            self.image.resize((shown_width, shown_height), Image.Resampling.BILINEAR)
        )
        self.canvas.create_image(left, top, image=self.photo, anchor="nw")

    def _accept_frame(self, item):
        if self.control is not None and self.control.preparing.is_set() and self.worker is not None:
            total = f" / {item['expected_frames']}" if item["expected_frames"] else ""
            self.progress.set(
                f"已处理 {item['frame']}{total} 帧 · 原帧率 {item['input_fps']:g} FPS"
            )
            return
        self.shown = item
        self.image = Image.fromarray(item["canvas"][:, :, ::-1])
        self.ready = True
        paused = item["paused"]
        result = item["result"]
        state = result.state.value
        identity = f"  ·  ID {result.tracker_id}" if result.tracker_id is not None else ""
        if not self.stopping:
            self.status.set(
                f"{'已暂停' if paused else '播放中'}  ·  {STATES.get(state, state)}{identity}"
            )
            self.play_button.configure(text="播放" if paused else "暂停", state="normal")
            self.step_button.configure(state="normal")
            self.reset_button.configure(state="normal" if paused else "disabled")
        total = f" / {item['expected_frames']}" if item["expected_frames"] else ""
        self.progress.set(
            f"第 {item['frame']}{total} 帧 · {item['timestamp']:.1f} 秒 · 1× {item['input_fps']:g} FPS"
        )
        if self.output.is_dir():
            self.output_button.configure(state="normal")
        self._draw()

    def _feedback(self, event):
        action = event["action"]
        if action == "cache_status":
            if event["reused"]:
                self.status.set("已复用分析缓存，正在更新A2选人状态…")
                self.hint.set("无需加载检测模型或重跑Tracker；只补算未缓存的A2外观并更新结果。")
            else:
                self.status.set("首次分析视频，正在准备模型和首帧…")
        elif action == "finished":
            self.session_ok = event["summary"]["complete_source"] and not self.stopping
            self.status.set("测试已结束，结果已保存")
            self.hint.set(f"已处理 {event['summary']['frames']} 帧。结果目录：{self.output}")
        elif action == "error":
            self.status.set("测试失败")
            self.hint.set(
                f"错误：{event['message']}。详细日志：{ROOT / 'outputs/tracking_gui.log'}"
            )
            if not self.closing:
                messagebox.showerror("测试失败", event["message"], parent=self.root)
        else:
            if event.get("success") and not event.get("replayed"):
                self.replay_actions.append(
                    {k: event[k] for k in ("action", "frame", "x", "y") if k in event}
                )
            self.hint.set(
                REASONS.get(event["reason"], "未能选中该人物，请尝试更清晰、较大的目标框。")
            )

    def _show_playback(self):
        frame = self.clock.due()
        self._accept_frame(self.buffer.read(frame, paused=not self.clock.playing))

    def _drain_control(self):
        if self.control is not None:
            try:
                self._accept_frame(self.control.frames.get_nowait())
            except Empty:
                pass
            while True:
                try:
                    self._feedback(self.control.events.get_nowait())
                except Empty:
                    break

    def _poll(self):
        self._drain_control()
        if self.worker is not None and not self.worker.is_alive():
            self.worker.join()
            # Completion may have arrived between draining and is_alive().
            self._drain_control()
            self.worker = None
            store = self.control.store
            self.control = None
            self.ready = False
            self.save_check.configure(state="normal")
            self.start_button.configure(state="normal")
            for button in (self.play_button, self.step_button, self.reset_button, self.stop_button):
                button.configure(state="disabled")
            if self.closing:
                store.close_reader()
                if self.analysis_cache is not None:
                    self.analysis_cache.close()
                self.root.destroy()
                return
            if self.pending_source is not None:
                path, self.pending_source = self.pending_source, None
                self.load_video(path)
            elif self.session_ok:
                self.buffer = store
                self.clock = PlaybackClock(store.fps, len(store.items))
                self.clock.frame = min(self.return_frame, self.clock.count)
                if self.autoplay:
                    self.clock.play()
                self.stop_button.configure(state="normal")
                self._show_playback()
                self.hint.set(f"处理完成，按视频原帧率回放。结果目录：{self.output}")
        if self.clock is not None and self.clock.playing:
            frame = self.clock.due()
            if frame != self.shown["frame"] or self.shown["paused"]:
                self._show_playback()
            if not self.clock.playing:
                self._show_playback()
                self.status.set("原速回放已结束，结果已保存")
        self.poll_id = self.root.after(
            5 if self.clock is not None and self.clock.playing else 40, self._poll
        )


def main():
    root = tk.Tk()
    TrackingGUI(root)
    root.mainloop()
