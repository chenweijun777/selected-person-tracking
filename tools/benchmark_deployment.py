"""Measure independent detection and selected-target computation on a video or MOT sequence."""
from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path
from time import perf_counter

from deploy_o2m import digest, save
from run_deployed_joint import MODELS, add_runtime_args, configure_threads, model_path


def statistics(values):
    import numpy as np

    if not values or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("Timing samples must be nonempty, positive and finite")
    mean = float(np.mean(values))
    return {"samples": len(values), "mean_ms": mean,
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "fps": 1000 / mean}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=(*MODELS, "all"), default="all")
    add_runtime_args(parser)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("detector", "pipeline", "both"), default="both")
    parser.add_argument("--select-frame", type=int, default=1)
    parser.add_argument("--select-point", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--skip-frames", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output directory")
    if args.source.startswith("camera:"):
        parser.error("Use a recorded video or MOT sequence for repeated timing")
    if args.backend == "all" and args.model:
        parser.error("--model requires a single --backend")
    if (args.device < 0 or args.warmup < 1 or args.repeats < 1
            or not 0 <= args.skip_frames < args.max_frames
            or not 1 <= args.select_frame <= args.max_frames):
        parser.error("Invalid device, warmup, repeats or frame range")
    if args.mode != "detector" and (not args.select_point
                                    or not all(math.isfinite(x) for x in args.select_point)):
        parser.error("Pipeline timing requires --select-point X Y")
    return args


def measure(args, backend, mode, repeat, path):
    import torch

    from a1_tracker.detectors.deployed import DeployedO2MDetector
    from a1_tracker.joint_runtime import JointRuntime
    from a1_tracker.types import FramePacket
    from a1_tracker.video_app import FrameSource, apply_action, canonical_detections

    source = FrameSource(args.source)
    try:
        first = source.read()
        if first is None:
            raise ValueError("Empty input")
        worker = DeployedO2MDetector(path, device=str(args.device))
        packet = FramePacket("warmup", 0, 0., 0., first[0])
        for _ in range(args.warmup):
            worker.detect(packet)
        runtime = (JointRuntime(source.fps, args.tracker, args.target_config, args.reid)
                   if mode == "pipeline" else None)
        rows, selected = [], False
        item, frame = first, 1
        while item is not None and frame <= args.max_frames:
            image, stamp = item
            torch.cuda.synchronize(args.device)
            start = perf_counter()
            detections = worker.detect(FramePacket("benchmark", frame, stamp, stamp, image))
            torch.cuda.synchronize(args.device)
            detector_ms = (perf_counter() - start) * 1000
            valid = False
            if runtime:
                runtime.begin_frame(image, stamp)
                tracks = runtime.tracker.update(canonical_detections(detections, .001), image)
                if frame == args.select_frame:
                    action = apply_action(runtime.target, tracks, image, {
                        "action": "click", "x": args.select_point[0], "y": args.select_point[1],
                    })
                    if not action["success"]:
                        raise ValueError(f"Selection failed: {action['reason']}")
                    selected = True
                result = runtime.target.update(tracks, stamp, image)
                valid = result.valid
                torch.cuda.synchronize(args.device)
            elapsed = (perf_counter() - start) * 1000 if runtime else detector_ms
            if frame > args.skip_frames and (runtime is None or selected):
                rows.append({"backend": backend, "mode": mode, "repeat": repeat,
                             "frame": frame, "detector_ms": detector_ms, "elapsed_ms": elapsed,
                             "target_valid": bool(valid)})
            frame += 1
            item = source.read() if frame <= args.max_frames else None
        if runtime and not selected:
            raise ValueError("Input ended before the selection frame")
        if not rows:
            raise ValueError("No timing samples remain after warmup/selection")
        return rows, worker.effective_predict_config
    finally:
        source.close()


def main():
    args = parse_args()
    import torch

    configure_threads()
    backends = list(MODELS) if args.backend == "all" else [args.backend]
    models = {b: model_path(b, args.model) for b in backends}
    modes = ("detector", "pipeline") if args.mode == "both" else (args.mode,)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, configurations = [], {}
    for mode in modes:
        for repeat in range(1, args.repeats + 1):
            order = backends if repeat % 2 else list(reversed(backends))
            for backend in order:
                samples, config = measure(args, backend, mode, repeat, models[backend])
                rows.extend(samples)
                configurations[backend] = config
                print(mode, repeat, backend, statistics([r["elapsed_ms"] for r in samples]),
                      flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    summary = []
    for mode in modes:
        for backend in backends:
            subset = [r for r in rows if r["mode"] == mode and r["backend"] == backend]
            summary.append({"backend": backend, "mode": mode,
                            **statistics([r["elapsed_ms"] for r in subset]),
                            "target_valid_samples": sum(r["target_valid"] for r in subset)})
    with (args.output / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    save(args.output / "benchmark.json", {
        "summary": summary, "gpu": torch.cuda.get_device_name(args.device),
        "source": str(Path(args.source).resolve()), "selection_frame": args.select_frame,
        "selection_point": args.select_point, "max_frames": args.max_frames,
        "skip_frames": args.skip_frames, "warmup": args.warmup, "repeats": args.repeats,
        "order": "forward then reverse, independent state for each run",
        "excludes": ["decoding", "model loading", "drawing", "encoding", "file writes"],
        "model_sha256": {b: digest(p) for b, p in models.items()},
        "configs": configurations,
        "tracker_sha256": digest(args.tracker), "target_sha256": digest(args.target_config),
        "reid_sha256": digest(args.reid) if "pipeline" in modes else None,
    })


if __name__ == "__main__":
    main()
