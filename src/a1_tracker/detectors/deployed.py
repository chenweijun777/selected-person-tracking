"""O2M deployment adapter with strict GPU selection and dynamic ORT IO binding.

Letterbox, RGB conversion, normalization, NMS and coordinate restoration remain
in the existing Ultralytics predictor. Only its raw network backend changes.
"""
from __future__ import annotations

from pathlib import Path
from types import MethodType

from a1_tracker.detectors.ultralytics import UltralyticsDetector


def _ort_cuda_forward(backend, image):
    """Bind dynamic GPU tensors without staging the image/output through CPU."""
    import numpy as np
    import torch

    if image.device.type != "cuda" or not image.is_contiguous():
        raise ValueError("ORT IO binding requires a contiguous CUDA tensor")
    io = backend.session.io_binding()
    io.bind_input(
        name=backend.session.get_inputs()[0].name,
        device_type="cuda",
        device_id=image.device.index,
        element_type=np.float32,
        shape=tuple(image.shape),
        buffer_ptr=image.data_ptr(),
    )
    for name in backend.output_names:
        io.bind_output(name, "cuda", image.device.index)
    # ORT and PyTorch use the same CUDA stream, including preprocessing and NMS.
    backend.session.run_with_iobinding(io)
    io.synchronize_outputs()
    values = io.get_outputs()
    # Keep the binding/allocation owners alive until the next synchronous call.
    backend._bound_output_owners = (io, values)
    return [torch.utils.dlpack.from_dlpack(value) for value in values]


class DeployedO2MDetector(UltralyticsDetector):
    """Explicit .pt/.onnx/.engine adapter; fail rather than silently using CPU."""

    def __init__(self, model, *, device="0", ort_io_binding=True):
        suffix = Path(model).suffix.lower()
        if suffix not in {".pt", ".onnx", ".engine"}:
            raise ValueError("Deployment requires a local .pt, .onnx or .engine model")
        super().__init__(str(model), image_size=960, confidence=.001,
                         iou_threshold=.70, device=device, nms=True)
        self.model_path = Path(model).resolve()
        self.ort_io_binding = ort_io_binding
        self._verified_backend = False

    def detect(self, frame):
        result = super().detect(frame)
        if not self._verified_backend:
            self._configure_backend()
            self._verified_backend = True
        if self.uses_end2end:
            raise RuntimeError("Deployment must preserve the O2M head and external NMS")
        return result

    def _configure_backend(self):
        import torch

        model = self._model.predictor.model
        backend = model.backend
        if model.device.type != "cuda":
            raise RuntimeError("Deployment requires actual GPU inference; CPU fallback rejected")
        if self.uses_end2end:
            raise RuntimeError("An O2O/embedded-NMS model cannot replace this O2M pipeline")
        if model.fp16:
            raise RuntimeError("Keep FP32 model IO to preserve the original normalization")
        if model.format == "onnx":
            if "CUDAExecutionProvider" not in backend.session.get_providers():
                raise RuntimeError("ONNX Runtime CUDAExecutionProvider was not initialized")
            if self.ort_io_binding:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
                stream = torch.cuda.current_stream(model.device)
                backend.session = ort.InferenceSession(
                    str(self.model_path), options,
                    providers=[("CUDAExecutionProvider", {
                        "device_id": model.device.index,
                        "user_compute_stream": str(stream.cuda_stream),
                        "use_tf32": "0",
                    }), "CPUExecutionProvider"],
                )
                if backend.session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError("ORT IO binding session silently fell back to CPU")
                backend.forward = MethodType(_ort_cuda_forward, backend)
                backend.use_io_binding = True

    @property
    def effective_predict_config(self):
        config = super().effective_predict_config
        predictor = getattr(self._model, "predictor", None)
        model = getattr(predictor, "model", None)
        if model is not None:
            backend = model.backend
            config.update(
                backend_format=model.format,
                actual_device=str(model.device),
                input_fp16=bool(model.fp16),
                dynamic=bool(model.dynamic),
                providers=(backend.session.get_providers() if model.format == "onnx" else None),
                ort_io_binding=(bool(backend.use_io_binding) if model.format == "onnx" else False),
            )
        return config
