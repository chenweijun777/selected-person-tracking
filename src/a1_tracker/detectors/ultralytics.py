from __future__ import annotations

from math import isfinite
from typing import Any

from a1_tracker.types import Detection, FramePacket


class UltralyticsDetector:
    """延迟导入 Ultralytics，避免核心测试依赖模型或 OpenCV。"""

    def __init__(
        self,
        model: str = "models/yolo26n.pt",
        image_size: int = 960,
        confidence: float = 0.10,
        iou_threshold: float = 0.70,
        person_class_id: int = 0,
        device: str = "0",
        nms: bool | None = True,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("请先安装运行依赖：pip install -e '.[runtime]'") from exc
        self._model: Any = YOLO(model)
        self.image_size = image_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        self.device = device
        self.nms = nms
        self.last_invalid_boxes = 0

    @property
    def uses_end2end(self) -> bool:
        """Return the branch selected by Ultralytics after predictor setup."""
        predictor = getattr(self._model, "predictor", None)
        backend = getattr(predictor, "model", None)
        return bool(getattr(backend, "end2end", False))

    @property
    def effective_predict_config(self) -> dict[str, Any]:
        """Actual predictor settings after setup, including postprocessing limits."""
        predictor = getattr(self._model, "predictor", None)
        args = getattr(predictor, "args", None)
        return {
            "end2end": self.uses_end2end,
            "nms_option": getattr(args, "nms", None),
            "nms_iou": getattr(args, "iou", None),
            "confidence": getattr(args, "conf", None),
            "imgsz": getattr(args, "imgsz", None),
            "device": str(getattr(args, "device", None)),
            "classes": getattr(args, "classes", None),
            "max_det": getattr(args, "max_det", None),
            "agnostic_nms": getattr(args, "agnostic_nms", None),
            "rect": getattr(args, "rect", None),
            "half": getattr(args, "half", None),
        }

    def detect(self, frame: FramePacket) -> list[Detection]:
        self.last_invalid_boxes = 0
        predict_options: dict[str, Any] = {
            "imgsz": self.image_size,
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "classes": [self.person_class_id],
            "device": self.device,
            "verbose": False,
        }
        if self.nms is not None:
            predict_options["nms"] = self.nms
        result = self._model.predict(frame.image, **predict_options)[0]
        if tuple(result.orig_shape) != tuple(frame.image.shape[:2]):
            raise RuntimeError("预测结果未还原到输入原图尺寸")
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        detections = []
        for box, score, class_id in zip(boxes, scores, classes, strict=True):
            if (
                not all(isfinite(v) for v in (*box, score, class_id))
                or box[2] <= box[0]
                or box[3] <= box[1]
                or not 0 <= score <= 1
            ):
                self.last_invalid_boxes += 1
                continue
            detections.append(Detection(tuple(map(float, box)), float(score), int(class_id)))
        return detections
