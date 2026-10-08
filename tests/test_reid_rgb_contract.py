"""A channel cleanup must preserve the calibrated tensor, including crop rounding."""

import numpy as np
import pytest


def test_clean_rgb_tensor_equals_legacy_calibrated448(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.trackers.utils.reid import ReID

    from a1_tracker.reid.ultralytics_onnx import configure_rgb_reid

    old = object.__new__(ReID)
    old.imgsz, old.device, old.fp16, old.is_pt = 448, torch.device("cpu"), False, False
    clean = object.__new__(ReID)
    clean.device, clean.fp16, clean.is_pt = torch.device("cpu"), False, False
    configure_rgb_reid(clean)
    image = np.random.default_rng(42).integers(0, 256, (99, 133, 3), dtype=np.uint8)
    boxes = np.array([[60.25, 48.5, 38.75, 71.25], [1, 2, 20, 30]], np.float32)
    expected = old._crops_to_tensor(
        old._crop_detections(np.ascontiguousarray(image[..., ::-1]), boxes)
    )
    actual = clean._crops_to_tensor(clean._crop_detections(image, boxes))
    assert torch.equal(actual, expected)
    pixel = np.full((32, 32, 3), [10, 20, 30], dtype=np.uint8)
    tensor = clean._crops_to_tensor(clean._crop_detections(pixel, boxes[1:]))
    assert torch.allclose(tensor[0, :, 0, 0], torch.tensor([30, 20, 10]) / 255)
