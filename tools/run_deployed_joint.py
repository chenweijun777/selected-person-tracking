"""Run selected-person tracking with an explicit O2M GPU backend."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/.ultralytics"))
os.environ["YOLO_AUTOINSTALL"] = "False"
MODELS = {"pytorch": "models/best.pt", "onnx": "outputs/deployment/best_o2m.onnx",
          "trt-fp32": "outputs/deployment/best_o2m_fp32.engine",
          "trt-fp16": "outputs/deployment/best_o2m_fp16.engine"}


def add_runtime_args(parser):
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tracker", type=Path, default=ROOT / "configs/pretrained/tracker.yaml")
    parser.add_argument("--target-config", type=Path, default=ROOT / "configs/pretrained/target.json")
    parser.add_argument("--reid", type=Path, default=ROOT / "models/yolo26n-reid.onnx")
    parser.add_argument("--device", type=int, default=0)


def model_path(backend, model=None):
    path = Path(model) if model is not None else ROOT / MODELS[backend]
    suffix = {"pytorch": ".pt", "onnx": ".onnx", "trt-fp32": ".engine", "trt-fp16": ".engine"}
    if path.suffix.lower() != suffix[backend]:
        raise ValueError(f"{backend} requires a {suffix[backend]} model")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def configure_threads():
    import cv2
    import onnxruntime as ort
    import torch

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    if not getattr(ort.InferenceSession, "_deployment_threads", False):
        original = ort.InferenceSession

        def bounded_session(path, sess_options=None, **kwargs):
            options = sess_options or ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
            return original(path, options, **kwargs)

        bounded_session._deployment_threads = True
        ort.InferenceSession = bounded_session


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=MODELS, required=True)
    add_runtime_args(parser)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output directory")
    if args.device < 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("--device must be nonnegative and --max-frames positive")
    return args


def main():
    args = parse_args()
    from a1_tracker.detectors.deployed import DeployedO2MDetector
    from a1_tracker.joint_runtime import JointRuntime
    from a1_tracker.video_app import file_sha, run_session

    model = model_path(args.backend, args.model)
    configure_threads()
    worker = DeployedO2MDetector(model, device=str(args.device))
    actions = json.loads(args.actions.read_text(encoding="utf-8-sig")) if args.actions else []
    provenance = {"backend": args.backend, "model_sha256": file_sha(model),
                  "tracker_sha256": file_sha(args.tracker),
                  "target_sha256": file_sha(args.target_config),
                  "reid_sha256": file_sha(args.reid),
                  "application_sha256": file_sha(__file__),
                  "strategy": "O2M/NMS + BoT-SORT/ShortLink + A2"}
    summary = run_session(
        args.source, args.output, worker, None, floor=.001, actions=actions,
        display=args.display, save_video=not args.no_video, max_frames=args.max_frames,
        provenance=provenance,
        runtime_factory=lambda fps: JointRuntime(fps, args.tracker, args.target_config, args.reid),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
