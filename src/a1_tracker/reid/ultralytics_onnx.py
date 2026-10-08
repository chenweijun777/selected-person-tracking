"""Appearance features from the same frozen ONNX model used in T3 comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from a1_tracker.types import Track


def configure_rgb_reid(encoder, imgsz=448):
    """Keep BGR crops until the backend's single BGR-to-RGB tensor conversion.

    This produces exactly the former flipped-image RGB tensor without copying
    the full image or reversing channels in save_one_box a second time.
    """
    import torch
    from ultralytics.utils.ops import xywh2xyxy
    from ultralytics.utils.plotting import save_one_box

    if encoder.is_pt:
        raise ValueError("The explicit RGB contract requires an exported ReID model")
    encoder.imgsz = imgsz

    def crop_bgr(image, boxes):
        return [
            save_one_box(box, image, BGR=True, save=False)
            for box in xywh2xyxy(torch.from_numpy(boxes[:, :4]))
        ]

    encoder._crop_detections = crop_bgr
    return encoder


class UltralyticsOnnxAppearance:
    def __init__(self, model: str | Path, *, mode="224_BGR"):
        from ultralytics.trackers.utils.reid import ReID

        path = Path(model).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model_path = path
        self.encoder = ReID(str(path), device="cpu")
        if mode == "448_RGB":
            configure_rgb_reid(self.encoder)
        elif mode != "224_BGR":
            raise ValueError(f"Unsupported appearance mode: {mode}")
        self.mode = mode

    def encode(self, image: np.ndarray, tracks: list[Track]) -> list[np.ndarray | None]:
        if not tracks:
            return []
        boxes = np.asarray(
            [
                [
                    (t.bbox_xyxy[0] + t.bbox_xyxy[2]) / 2,
                    (t.bbox_xyxy[1] + t.bbox_xyxy[3]) / 2,
                    t.bbox_xyxy[2] - t.bbox_xyxy[0],
                    t.bbox_xyxy[3] - t.bbox_xyxy[1],
                ]
                for t in tracks
            ],
            dtype=np.float32,
        )
        features = self.encoder(image, boxes)
        if len(features) != len(tracks):
            raise RuntimeError("appearance encoder returned a different number of features")
        normalized = []
        for feature in features:
            if feature is None:
                normalized.append(None)
                continue
            vector = np.asarray(feature, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            normalized.append(vector / norm if np.isfinite(norm) and norm > 0 else None)
        return normalized
