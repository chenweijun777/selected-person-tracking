"""Video application: click, track, recover or reselect, and export a session.

This module never reads GT. Scripted clicks exercise exactly the same handler as
the desktop mouse callback and are explicitly distinguished from human input.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from a1_tracker.geometry import contains_point
from a1_tracker.target.baselines import TargetBaseline
from a1_tracker.types import Detection, FramePacket


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical_detections(detections, floor):
    """Match saved MOT cache precision and reject quantization-degenerate boxes."""
    result = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        width, height = round(x2 - x1, 3), round(y2 - y1, 3)
        score = round(round(det.score, 6), 3)
        if width <= 0 or height <= 0 or score < floor:
            continue
        left, top = round(x1, 3), round(y1, 3)
        result.append(Detection((left, top, left + width, top + height), score, det.class_id))
    return result


def apply_action(target, tracks, image, action):
    """Reject ambiguous clicks atomically; reset/reselect is always explicit."""
    old_session = target.session_target_id
    kind = action["action"]
    if kind == "reset":
        target.reset()
        success, reason = True, "selection_cleared"
    elif kind == "click":
        point = (float(action["x"]), float(action["y"]))
        if not all(math.isfinite(v) for v in point):
            raise ValueError("Click coordinates must be finite")
        candidates = [t for t in tracks if t.observed and contains_point(t.bbox_xyxy, point)]
        success = target.select_by_point(tracks, point, image)
        reason = (
            "selected"
            if success
            else "ambiguous_click"
            if len(candidates) > 1
            else "no_observed_track_at_click"
            if not candidates
            else getattr(target, "selection_failure_reason", None) or "selection_failed"
        )
    else:
        raise ValueError(f"Unknown action: {kind}")
    return {
        **action,
        "success": success,
        "reason": reason,
        "previous_session": old_session,
        "session_target_id": target.session_target_id,
        "tracker_id": target.tracker_id,
    }


class FrameSource:
    def __init__(self, source):
        import cv2

        self.camera = source.startswith("camera:")
        self.directory = Path(source) if not self.camera and Path(source).is_dir() else None
        self.cap = None
        self.index = 0
        self.expected = None
        if self.directory is not None:
            info = configparser.ConfigParser()
            if not info.read(self.directory / "seqinfo.ini"):
                raise ValueError("Image sequence requires seqinfo.ini in its directory")
            self.fps = info.getfloat("Sequence", "frameRate")
            self.expected = info.getint("Sequence", "seqLength")
            self.paths = sorted((self.directory / info.get("Sequence", "imDir")).glob("*.jpg"))
            if [int(p.stem) for p in self.paths] != list(range(1, self.expected + 1)):
                raise ValueError("Incomplete or non-contiguous image sequence")
        else:
            self.cap = cv2.VideoCapture(int(source.split(":")[1]) if self.camera else source)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open {source}")
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            if self.camera and (not math.isfinite(self.fps) or self.fps <= 0):
                self.fps = 30.0  # encoding rate only; timestamps use monotonic capture time
            if not self.camera:
                count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.expected = count if count > 0 else None
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.close()
            raise ValueError("Input has no valid FPS metadata")
        self.started = perf_counter()

    def read(self):
        import cv2

        if self.directory:
            if self.index == len(self.paths):
                return None
            image = cv2.imread(str(self.paths[self.index]))
            if image is None:
                raise ValueError(f"Cannot decode {self.paths[self.index]}")
        else:
            ok, image = self.cap.read()
            if not ok:
                if self.camera or (self.expected is not None and self.index < self.expected):
                    raise ValueError(f"Capture failed at frame {self.index + 1}")
                return None
        self.index += 1
        timestamp = perf_counter() - self.started if self.camera else (self.index - 1) / self.fps
        return image, timestamp

    def close(self):
        if self.cap is not None:
            self.cap.release()


def run_session(
    source,
    output,
    detector,
    tracker,
    *,
    floor=0.1,
    actions=None,
    display=False,
    save_video=True,
    max_frames=None,
    provenance=None,
    runtime_factory=None,
    controller=None,
):
    import cv2

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame_source = FrameSource(source)
    try:
        runtime = runtime_factory(frame_source.fps) if runtime_factory else None
    except Exception:
        frame_source.close()
        raise
    target = runtime.target if runtime else TargetBaseline("A0")
    tracker = runtime.tracker if runtime else tracker
    strategy = getattr(target, "strategy", "A0")
    action_map = defaultdict(list)
    for action in actions or []:
        if action.get("frame", 0) < 1 or action.get("action") not in ("click", "reset"):
            raise ValueError(f"Invalid scripted action: {action}")
        action_map[action["frame"]].append(action)
    events, timings, states = [], [], Counter()
    invalid_boxes = 0
    source_start = perf_counter()
    writer = None
    records = None
    tracks_file = None
    diagnostics_file = None
    motion_file = None
    paused, stop, completed = display or controller is not None, False, False
    queue = []
    scale = 1.0
    title = f"{strategy} | click: select | space: pause | r: reset | q: quit"
    if display:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                queue.append({"action": "click", "x": x / scale, "y": y / scale})

        cv2.setMouseCallback(title, mouse)
    warmup_ms = 0.0
    tracker.reset()
    try:
        records = (output / "target.jsonl").open("x", encoding="utf-8")
        tracks_file = (output / "tracks.txt").open("x", encoding="utf-8")
        if runtime:
            diagnostics_file = (output / "diagnostics.jsonl").open("x", encoding="utf-8")
            motion_file = (output / "camera_motion.csv").open("x", newline="", encoding="utf-8")
            motion_writer = csv.DictWriter(
                motion_file,
                fieldnames=(
                    "frame",
                    "timestamp",
                    "h00",
                    "h01",
                    "h02",
                    "h10",
                    "h11",
                    "h12",
                    "valid",
                    "reason",
                    "matches",
                    "inlier_ratio",
                    "residual_px",
                ),
            )
            motion_writer.writeheader()
            (output / "runtime.json").write_text(
                json.dumps(runtime.provenance(), indent=2, allow_nan=False), encoding="utf-8"
            )
        while not stop and (max_frames is None or frame_source.index < max_frames):
            begin = perf_counter()
            item = frame_source.read()
            decode_ms = (perf_counter() - begin) * 1000
            if item is None:
                completed = True
                break
            image, timestamp = item
            frame = frame_source.index
            packet = FramePacket(source, frame, timestamp, perf_counter(), image)
            this_warmup = 0.0
            if frame == 1:
                t = perf_counter()
                for _ in range(5):
                    detector.detect(packet)
                this_warmup = perf_counter() - t
                warmup_ms = this_warmup * 1000
            t = perf_counter()
            detections = canonical_detections(detector.detect(packet), floor)
            invalid_boxes += getattr(detector, "last_invalid_boxes", 0)
            detect_ms = (perf_counter() - t) * 1000
            t = perf_counter()
            if runtime:
                runtime.begin_frame(image, timestamp)
            gmc_ms = (perf_counter() - t) * 1000 if runtime else 0.0
            t = perf_counter()
            tracks = tracker.update(detections, image)
            track_ms = (perf_counter() - t) * 1000
            t = perf_counter()
            for action in action_map.get(frame, []):
                events.append(
                    apply_action(
                        target,
                        tracks,
                        image,
                        {**action, "timestamp": timestamp, "source": "scripted"},
                    )
                )
            result = target.update(tracks, timestamp, image)
            target_ms = (perf_counter() - t) * 1000
            t = perf_counter()

            def render(image, tracks, result, frame, timestamp):
                canvas = image.copy()
                for tr in tracks:
                    a, b, c, d = map(round, tr.bbox_xyxy)
                    cv2.rectangle(canvas, (a, b), (c, d), (140, 140, 140), 1)
                    cv2.putText(
                        canvas,
                        str(tr.tracker_id),
                        (a, max(15, b)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (220, 220, 220),
                        1,
                    )
                if result.valid:
                    a, b, c, d = map(round, result.bbox_xyxy)
                    cv2.rectangle(canvas, (a, b), (c, d), (0, 230, 0), 3)
                message = f"F{frame} {timestamp:.2f}s {strategy} {result.state.value} ID={result.tracker_id}"
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 55), (20, 20, 20), -1)
                cv2.putText(
                    canvas, message, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
                )
                labels = {
                    "UNSELECTED": ("Click one person to select", (220, 220, 220)),
                    "TRACKING": ("Tracking selected person", (0, 230, 0)),
                    "LOST": (
                        "Searching for selected person"
                        if runtime
                        else "LOST: no substitute identity; click to reselect",
                        (0, 220, 255),
                    ),
                    "VERIFYING": ("Verifying recovery candidate", (0, 160, 255)),
                    "NEED_RESELECT": ("Search expired: click to reselect", (80, 80, 255)),
                }
                text, color = labels[result.state.value]
                cv2.putText(canvas, text, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                return canvas

            canvas = render(image, tracks, result, frame, timestamp)
            render_ms = (perf_counter() - t) * 1000
            paused_seconds = 0.0
            if controller is not None:
                # The GUI sends only commands. Tk widgets and image presentation
                # live on its main thread; model/GMC/target state stay here.
                def present(canvas, result, paused, frame=frame, timestamp=timestamp):
                    controller.present(
                        canvas,
                        frame=frame,
                        timestamp=timestamp,
                        result=result,
                        paused=paused,
                        expected_frames=frame_source.expected,
                        input_fps=frame_source.fps,
                    )

                present(canvas, result, paused)
                while True:
                    wait_started = perf_counter()
                    was_paused = paused
                    commands = controller.poll(timeout=0.05 if paused else 0.0)
                    if was_paused:
                        paused_seconds += perf_counter() - wait_started
                    advance_once = False
                    for command in commands:
                        action = command["action"]
                        if action == "stop":
                            stop = True
                        elif action == "pause":
                            paused = True
                        elif action == "play":
                            paused = False
                        elif action == "step":
                            paused = True
                            advance_once = True
                        elif action in ("click", "reset"):
                            if command.get("frame", frame) != frame:
                                event = {
                                    **command,
                                    "success": False,
                                    "reason": "stale_frame_input",
                                    "source": "desktop_gui",
                                }
                            else:
                                event = apply_action(
                                    target,
                                    tracks,
                                    image,
                                    {
                                        **command,
                                        "frame": frame,
                                        "timestamp": timestamp,
                                        "source": "desktop_gui",
                                    },
                                )
                                result = target.update(tracks, timestamp, image)
                                canvas = render(image, tracks, result, frame, timestamp)
                            events.append(event)
                            controller.feedback(event)
                        else:
                            raise ValueError(f"Unknown GUI action: {action}")
                    if commands:
                        present(canvas, result, paused)
                    if not paused or stop or advance_once:
                        break
            if display:
                scale = min(1.0, 1280 / image.shape[1])
                while True:
                    shown = cv2.resize(
                        canvas, (round(image.shape[1] * scale), round(image.shape[0] * scale))
                    )
                    cv2.imshow(title, shown)
                    wait_started = perf_counter()
                    was_paused = paused
                    key = cv2.waitKey(30 if paused else 1) & 0xFF
                    if was_paused:
                        paused_seconds += perf_counter() - wait_started
                    if key == ord("q") or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        stop = True
                    if key == ord(" "):
                        paused = not paused
                    if key == ord("r"):
                        queue.append({"action": "reset"})
                    if queue:
                        for action in queue:
                            events.append(
                                apply_action(
                                    target,
                                    tracks,
                                    image,
                                    {
                                        **action,
                                        "frame": frame,
                                        "timestamp": timestamp,
                                        "source": "mouse_or_keyboard",
                                    },
                                )
                            )
                        queue.clear()
                        result = target.update(tracks, timestamp, image)
                        canvas = render(image, tracks, result, frame, timestamp)
                    if not paused or stop:
                        break
            t = perf_counter()
            if save_video:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        frame_source.fps,
                        (image.shape[1], image.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot open video writer")
                writer.write(canvas)
            payload = {"frame_id": frame, **asdict(result)}
            records.write(json.dumps(payload) + "\n")
            records.flush()
            if runtime:
                diagnostics_file.write(
                    json.dumps({**payload, **runtime.diagnostics()}, allow_nan=False) + "\n"
                )
                warp = runtime.motion_warp.ravel()
                motion_writer.writerow(
                    dict(
                        frame=frame,
                        timestamp=timestamp,
                        **dict(zip(("h00", "h01", "h02", "h10", "h11", "h12"), warp, strict=True)),
                        **runtime.motion_info,
                    )
                )
            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox_xyxy
                tracks_file.write(
                    f"{frame},{tr.tracker_id},{x1:.3f},{y1:.3f},{x2 - x1:.3f},{y2 - y1:.3f},{tr.score:.6f},-1,-1,-1\n"
                )
            tracks_file.flush()
            write_ms = (perf_counter() - t) * 1000
            total = (perf_counter() - begin - this_warmup - paused_seconds) * 1000
            timings.append(
                {
                    "frame": frame,
                    "decode_ms": decode_ms,
                    "detect_ms": detect_ms,
                    "gmc_ms": gmc_ms,
                    "track_ms": track_ms,
                    "target_ms": target_ms,
                    "render_ms": render_ms,
                    "write_ms": write_ms,
                    "active_total_ms": total,
                    "paused_wait_ms": paused_seconds * 1000,
                }
            )
            states[result.state.value] += 1
            if frame % 250 == 0:
                print(f"APP {source}: {frame} frames", flush=True)
        if frame_source.expected == frame_source.index:
            completed = True
    finally:
        if writer is not None:
            writer.release()
        if records is not None:
            records.close()
        if tracks_file is not None:
            tracks_file.close()
        if diagnostics_file is not None:
            diagnostics_file.close()
        if motion_file is not None:
            motion_file.close()
        frame_source.close()
        if display:
            cv2.destroyWindow(title)
        wall = perf_counter() - source_start
        summary = {
            "frames": len(timings),
            "source": source,
            "expected_source_frames": frame_source.expected,
            "complete_source": completed,
            "requested_max_frames": max_frames,
            "operator_stopped": stop,
            "states": dict(states),
            "warmup_ms": warmup_ms,
            "wall_seconds_including_warmup_pauses_and_encoder_close": wall,
            "display": display,
            "save_video": save_video,
            "input_fps": frame_source.fps,
            "desktop_gui": controller is not None,
            "camera_drop_rate": None,
            "camera_drop_rate_note": "No hardware frame counter; never infer zero drops.",
            "timing_scope": f"Decode + fresh detector + GMC + tracker/appearance + {strategy} + draw + video encode + JSONL/MOT writes; excludes model loading, 5 warmups and paused wait. Display=false excludes physical display latency.",
            "invalid_boxes_skipped": invalid_boxes,
            "effective_predict_config": getattr(detector, "effective_predict_config", None),
            "provenance": provenance or {},
        }
        if runtime:
            summary["runtime"] = runtime.provenance()
        if timings:
            with (output / "timings.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(timings[0]))
                w.writeheader()
                w.writerows(timings)
            summary["latency"] = {
                k: {
                    "p50_ms": float(np.percentile([r[k] for r in timings], 50)),
                    "p95_ms": float(np.percentile([r[k] for r in timings], 95)),
                }
                for k in timings[0]
                if k.endswith("_ms")
            }
        (output / "actions.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        summary["unexecuted_scripted_actions"] = [
            v for frame, values in action_map.items() if frame > len(timings) for v in values
        ]
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not timings:
        raise ValueError("No frame processed")
    if save_video:
        cap = cv2.VideoCapture(str(output / "annotated.mp4"))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count != len(timings):
            raise ValueError(f"Encoded frame count differs: {count} != {len(timings)}")
    return summary


def main(*, joint=False):
    from a1_tracker.detectors.ultralytics import UltralyticsDetector
    from a1_tracker.trackers.ultralytics_mot import UltralyticsMOTTracker

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "YOLO26s O2M/960 + 5s Lost/ShortLink + standalone A2 video application"
            if joint
            else "A0 video application: click, reject on loss, reselect, and export a session."
        )
    )
    parser.add_argument(
        "--source", required=True, help="Video, MOT image-sequence directory or camera:0"
    )
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    parser.add_argument(
        "--model", type=Path, default=root / ("models/best.pt" if joint else "models/yolo26n.pt")
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        default=root
        / ("configs/finetuned/tracker.yaml" if joint else "configs/tracker_t2_botsort_gmc.yaml"),
    )
    if joint:
        parser.add_argument(
            "--target-config", type=Path, default=root / "configs/finetuned/target.json"
        )
        parser.add_argument(
            "--appearance-model", type=Path, default=root / "models/yolo26n-reid.onnx"
        )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--detection-floor", type=float, default=0.001 if joint else 0.1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new session directory")
    if not args.model.is_file() or not args.tracker.is_file():
        parser.error("Local model and tracker configuration must exist")
    if joint and (not args.target_config.is_file() or not args.appearance_model.is_file()):
        parser.error("Local A2 configuration and appearance model must exist")
    if not 0.001 <= args.detection_floor <= 1 or (
        args.max_frames is not None and args.max_frames < 1
    ):
        parser.error("Invalid detection floor or max frames")
    actions = json.loads(args.actions.read_text(encoding="utf-8-sig")) if args.actions else []
    detector = UltralyticsDetector(
        str(args.model.resolve()), args.imgsz, 0.001, 0.7, device=args.device, nms=True
    )
    runtime_factory = None
    if joint:
        from a1_tracker.joint_runtime import JointRuntime

        def runtime_factory(fps):
            return JointRuntime(fps, args.tracker, args.target_config, args.appearance_model)

    tracker = None if joint else UltralyticsMOTTracker(args.tracker)
    provenance = {
        "model": str(args.model.resolve()),
        "model_sha256": file_sha(args.model),
        "tracker": str(args.tracker.resolve()),
        "tracker_sha256": file_sha(args.tracker),
        "application_sha256": file_sha(__file__),
        "imgsz": args.imgsz,
        "inference_floor": 0.001,
        "detection_floor": args.detection_floor,
        "strategy": "A2_v7 + ShortLink2" if joint else "A0",
        "actions_source": str(args.actions) if args.actions else None,
        "actions_sha256": file_sha(args.actions) if args.actions else None,
    }
    print(
        json.dumps(
            run_session(
                args.source,
                args.output,
                detector,
                tracker,
                floor=args.detection_floor,
                actions=actions,
                display=args.display,
                save_video=not args.no_video,
                max_frames=args.max_frames,
                provenance=provenance,
                runtime_factory=runtime_factory,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
