"""Export a dynamic O2M ONNX model and build local TensorRT engines."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {"min": (1, 3, 32, 32), "opt": (1, 3, 544, 960), "max": (1, 3, 960, 960)}
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/.ultralytics"))
os.environ["YOLO_AUTOINSTALL"] = "False"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def fresh(path):
    if Path(path).exists():
        raise FileExistsError(f"Output already exists: {path}")


def inspect_onnx(path):
    import onnx

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    metadata = {m.key: m.value for m in graph.metadata_props}
    if ast.literal_eval(metadata.get("end2end", "True")) is not False:
        raise ValueError("Expected raw O2M predictions (end2end=False)")
    if any(n.op_type == "NonMaxSuppression" for n in graph.graph.node):
        raise ValueError("NMS must remain outside the exported graph")
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("Expected one image input and one raw prediction output")
    image, output = graph.graph.input[0], graph.graph.output[0]
    shape = image.type.tensor_type.shape.dim
    if (len(shape) != 4 or shape[1].dim_value != 3
            or not shape[2].dim_param or not shape[3].dim_param):
        raise ValueError("Expected dynamic NCHW input with three channels")
    if any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for v in (image, output)):
        raise ValueError("Model inputs and outputs must use FP32")
    for key in ("names", "args", "end2end", "imgsz"):
        if key in metadata:
            metadata[key] = ast.literal_eval(metadata[key])
    return metadata, image.name


def export_model(model, output, device):
    from ultralytics import YOLO

    model, output = Path(model), Path(output)
    if model.suffix.lower() != ".pt" or not model.is_file():
        raise ValueError("--model must be an existing .pt file")
    output.mkdir(parents=True, exist_ok=True)
    copied, onnx_path = output / "best_o2m.pt", output / "best_o2m.onnx"
    fresh(copied)
    fresh(onnx_path)
    shutil.copyfile(model, copied)
    exported = Path(YOLO(str(copied)).export(
        format="onnx", imgsz=960, batch=1, dynamic=True, nms=None,
        simplify=False, opset=17, device=str(device),
    ))
    if exported.resolve() != onnx_path.resolve():
        raise RuntimeError(f"Unexpected export path: {exported}")
    metadata, name = inspect_onnx(onnx_path)
    save(output / "export.json", {
        "weights_sha256": digest(model), "onnx_sha256": digest(onnx_path),
        "input_name": name, "metadata": metadata, "opset": 17,
        "input_output_precision": "FP32", "nms": "external",
    })
    return onnx_path


def build_engine(onnx_path, output, precision, device):
    import tensorrt as trt
    import torch

    torch.cuda.set_device(int(device))
    metadata, name = inspect_onnx(onnx_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"best_o2m_{precision}.engine"
    fresh(path)
    start = perf_counter()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not support fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape(name, PROFILE["min"], PROFILE["opt"], PROFILE["max"])
    if not profile or tuple(profile.get_shape(name)[1]) != PROFILE["opt"]:
        raise RuntimeError("TensorRT shape profile was not accepted")
    config.add_optimization_profile(profile)
    payload = builder.build_serialized_network(network, config)
    if payload is None:
        raise RuntimeError("TensorRT engine build failed")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(payload)
    if engine is None:
        raise RuntimeError("Built engine cannot be loaded")
    io = [{"name": engine.get_tensor_name(i),
           "dtype": str(engine.get_tensor_dtype(engine.get_tensor_name(i))),
           "shape": list(engine.get_tensor_shape(engine.get_tensor_name(i)))}
          for i in range(engine.num_io_tensors)]
    if any(engine.get_tensor_dtype(x["name"]) != trt.float32 for x in io):
        raise RuntimeError("Engine input/output precision changed from FP32")
    header = json.dumps(metadata).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(len(header).to_bytes(4, "little"))
        stream.write(header)
        stream.write(payload)
    save(output / f"build_{precision}.json", {
        "engine_sha256": digest(path), "onnx_sha256": digest(onnx_path),
        "gpu": torch.cuda.get_device_name(int(device)), "cuda": torch.version.cuda,
        "tensorrt": trt.__version__, "internal_fp16": precision == "fp16",
        "tf32": False, "profile": PROFILE, "workspace_GiB": 2,
        "io": io, "build_seconds": perf_counter() - start,
    })
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("export", "build", "all"), default="all")
    parser.add_argument("--model", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/deployment")
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be a nonnegative CUDA device index")
    if args.onnx and args.stage != "build":
        parser.error("--onnx is only used with --stage build")
    return args


def main():
    args = parse_args()
    onnx_path = args.onnx or args.output / "best_o2m.onnx"
    if args.stage in ("export", "all"):
        onnx_path = export_model(args.model, args.output, args.device)
        print(f"Exported {onnx_path}")
    if args.stage in ("build", "all"):
        for precision in (("fp32", "fp16") if args.precision == "both" else (args.precision,)):
            print(f"Built {build_engine(onnx_path, args.output, precision, args.device)}")


if __name__ == "__main__":
    main()
