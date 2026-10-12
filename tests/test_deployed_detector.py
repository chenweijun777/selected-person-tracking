"""Reject wrong detection heads, GPU fallback, and changed normalization."""
from types import SimpleNamespace

import pytest

from a1_tracker.detectors.deployed import DeployedO2MDetector


def worker_for(backend):
    worker = DeployedO2MDetector.__new__(DeployedO2MDetector)
    worker._model = SimpleNamespace(predictor=SimpleNamespace(model=backend))
    worker.ort_io_binding = False
    return worker


def test_cpu_fallback_is_rejected():
    backend = SimpleNamespace(device=SimpleNamespace(type="cpu"), backend=object())
    with pytest.raises(RuntimeError, match="CPU fallback"):
        worker_for(backend)._configure_backend()


def test_o2o_or_embedded_nms_cannot_silently_replace_o2m():
    backend = SimpleNamespace(device=SimpleNamespace(type="cuda"), backend=object(), end2end=True)
    with pytest.raises(RuntimeError, match="O2O/embedded-NMS"):
        worker_for(backend)._configure_backend()


def test_fp16_io_cannot_change_the_original_preprocessing():
    backend = SimpleNamespace(device=SimpleNamespace(type="cuda"), backend=object(), end2end=False, fp16=True)
    with pytest.raises(RuntimeError, match="FP32 model IO"):
        worker_for(backend)._configure_backend()


def test_available_provider_name_without_initialized_cuda_is_rejected():
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    backend = SimpleNamespace(device=SimpleNamespace(type="cuda"), backend=SimpleNamespace(session=session),
        end2end=False, fp16=False, format="onnx")
    with pytest.raises(RuntimeError, match="not initialized"):
        worker_for(backend)._configure_backend()
