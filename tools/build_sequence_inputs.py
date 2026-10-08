"""Build causal tracker/GMC/appearance inputs from images and local models; no GT."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "outputs/.ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/.ultralytics"))


def main():
    import numpy as np
    import torch

    from a1_tracker.detectors.ultralytics import UltralyticsDetector
    from a1_tracker.joint_runtime import JointRuntime, sha256
    from a1_tracker.types import FramePacket
    from a1_tracker.video_app import FrameSource, canonical_detections

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video or MOT image sequence directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--tracker", type=Path, default=ROOT / "configs/finetuned/tracker.yaml")
    parser.add_argument("--target", type=Path, default=ROOT / "configs/finetuned/target.json")
    parser.add_argument("--appearance", type=Path, default=ROOT / "models/yolo26n-reid.onnx")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    source = FrameSource(args.source)
    detector = UltralyticsDetector(str(args.model), 960, 0.001, 0.7, device=device, nms=True)
    runtime = JointRuntime(source.fps, args.tracker, args.target, args.appearance)
    vectors = []
    present = []
    motion = []
    events = []
    count = 0
    try:
        with (args.output / "tracks.txt").open("x", encoding="utf-8") as stream:
            while True:
                item = source.read()
                if item is None:
                    break
                image, timestamp = item
                frame = source.index
                runtime.begin_frame(image, timestamp)
                raw = detector.detect(FramePacket(args.source, frame, timestamp, timestamp, image))
                tracks = runtime.tracker.update(canonical_detections(raw, 0.001), image)
                runtime.target.appearance.new_frame()
                features = runtime.target.appearance.encode(image, tracks)
                for track, feature in zip(tracks, features, strict=True):
                    x, y, x2, y2 = track.bbox_xyxy
                    stream.write(
                        f"{frame},{track.tracker_id},{x:.3f},{y:.3f},{x2 - x:.3f},{y2 - y:.3f},{track.score:.6f},-1,-1,-1\n"
                    )
                    present.append(feature is not None)
                    vectors.append(feature if feature is not None else np.zeros(512, np.float32))
                warp = runtime.motion_warp.reshape(-1)
                motion.append(
                    dict(
                        frame=frame,
                        valid=runtime.motion_info["valid"],
                        **dict(
                            zip(
                                ("h00", "h01", "h02", "h10", "h11", "h12"),
                                map(float, warp),
                                strict=True,
                            )
                        ),
                    )
                )
                for event in runtime.tracker.identity_frame.remaps:
                    events.append(
                        dict(
                            frame=frame,
                            young_id=event.young_id,
                            old_id=event.old_id,
                            bbox=list(event.bbox_xyxy),
                        )
                    )
                count += 1
                if count % 100 == 0:
                    print("Built frame", count, flush=True)
                if args.max_frames and count >= args.max_frames:
                    break
    finally:
        source.close()
    if not count:
        raise ValueError("No readable frames in the input source")
    np.save(
        args.output / "features.npy",
        np.asarray(vectors, np.float32).reshape(-1, 512),
        allow_pickle=False,
    )
    np.save(args.output / "valid.npy", np.asarray(present, bool), allow_pickle=False)
    with (args.output / "motion.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(motion[0]))
        writer.writeheader()
        writer.writerows(motion)
    (args.output / "remaps.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    metadata = dict(
        frames=count,
        rows=len(vectors),
        source=args.source,
        models={p.name: sha256(p) for p in [args.model, args.appearance]},
        provenance=runtime.provenance(),
        GT_online=False,
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved", count, "frames and", len(vectors), "observations")


if __name__ == "__main__":
    main()
