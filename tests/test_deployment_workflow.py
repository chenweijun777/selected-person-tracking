"""Public deployment entry points preserve paths, backend contracts and timing scope."""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return tuple(importlib.import_module(name) for name in (
        "deploy_o2m", "run_deployed_joint", "benchmark_deployment"))


def test_deployment_paths_are_independent_of_working_directory(scripts, tmp_path, monkeypatch):
    deploy, run, _ = scripts
    monkeypatch.chdir(tmp_path)
    args = run.parse_args(["--backend", "onnx", "--source", "video.mp4",
                           "--output", str(tmp_path / "new")])
    assert args.tracker == run.ROOT / "configs/pretrained/tracker.yaml"
    assert args.target_config == run.ROOT / "configs/pretrained/target.json"
    assert deploy.parse_args([]).model == run.ROOT / "models/best.pt"
    assert deploy.parse_args([]).precision == "both"


def test_wrong_model_type_and_existing_output_are_rejected(scripts, tmp_path):
    deploy, run, _ = scripts
    model = tmp_path / "model.pt"
    model.write_bytes(b"test")
    assert run.model_path("pytorch", model) == model.resolve()
    with pytest.raises(ValueError, match="requires"):
        run.model_path("onnx", model)
    with pytest.raises(FileNotFoundError):
        run.model_path("onnx", tmp_path / "missing.onnx")
    with pytest.raises(FileExistsError):
        deploy.fresh(model)
    with pytest.raises(SystemExit):
        run.parse_args(["--backend", "pytorch", "--source", "video.mp4",
                        "--output", str(tmp_path)])


def test_pipeline_benchmark_requires_finite_target_selection(scripts, tmp_path):
    _, _, bench = scripts
    base = ["--source", "video.mp4", "--output", str(tmp_path / "new")]
    with pytest.raises(SystemExit):
        bench.parse_args(base)
    with pytest.raises(SystemExit):
        bench.parse_args(base + ["--select-point", "nan", "10"])
    with pytest.raises(SystemExit):
        bench.parse_args(base + ["--mode", "detector", "--max-frames", "5"])
    assert bench.parse_args(base + ["--mode", "detector"]).warmup == 20


def test_statistics_use_reciprocal_mean_latency(scripts):
    _, _, bench = scripts
    stats = bench.statistics([10., 30.])
    assert stats["samples"] == 2
    assert stats["fps"] == 50.
    assert stats["p50_ms"] == 20.
    assert stats["p95_ms"] == 29.
    for bad in ([], [0.], [-1.], [math.nan], [math.inf]):
        with pytest.raises(ValueError):
            bench.statistics(bad)


def test_export_contract_accepts_raw_dynamic_o2m_and_rejects_o2o(scripts, tmp_path):
    onnx = pytest.importorskip("onnx")
    deploy, _, _ = scripts
    image = onnx.helper.make_tensor_value_info(
        "images", onnx.TensorProto.FLOAT, ["batch", 3, "height", "width"])
    output = onnx.helper.make_tensor_value_info(
        "raw", onnx.TensorProto.FLOAT, ["batch", 3, "height", "width"])
    graph = onnx.helper.make_graph([onnx.helper.make_node("Identity", ["images"], ["raw"])],
                                  "contract", [image], [output])
    model = onnx.helper.make_model(graph)
    onnx.helper.set_model_props(model, {"end2end": "False"})
    path = tmp_path / "test.onnx"
    onnx.save(model, path)
    assert deploy.inspect_onnx(path)[1] == "images"
    onnx.helper.set_model_props(model, {"end2end": "True"})
    onnx.save(model, path)
    with pytest.raises(ValueError, match="O2M"):
        deploy.inspect_onnx(path)


def test_custom_onnx_only_used_for_build(scripts):
    deploy, _, _ = scripts
    with pytest.raises(SystemExit):
        deploy.parse_args(["--onnx", "export.onnx"])
    args = deploy.parse_args(["--stage", "build", "--onnx", "export.onnx",
                              "--precision", "fp16"])
    assert args.onnx == Path("export.onnx")


def test_runtime_records_gpu_ort_when_cpu_distribution_is_absent(tmp_path, monkeypatch):
    import a1_tracker.joint_runtime as module

    class Tracker:
        pass

    monkeypatch.setattr(module, "JointTracker", lambda *args: SimpleNamespace(
        base=SimpleNamespace(tracker=Tracker())))
    monkeypatch.setattr(module, "JointTarget", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "FrameBoxAppearance", lambda *args: object())
    monkeypatch.setattr(module, "SparseCameraMotion", lambda: object())

    def version(name):
        if name == "onnxruntime":
            raise module.importlib.metadata.PackageNotFoundError(name)
        return "1.30.0" if name == "onnxruntime-gpu" else "test"

    monkeypatch.setattr(module.importlib.metadata, "version", version)
    paths = [tmp_path / name for name in ("tracker.yaml", "target.json", "reid.onnx")]
    for path in paths:
        path.write_bytes(b"fixture")
    runtime = module.JointRuntime(30., *paths)
    assert runtime.versions["onnxruntime-gpu"] == "1.30.0"
    assert "onnxruntime" not in runtime.versions
