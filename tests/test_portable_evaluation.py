"""Evaluation stays independent of draft code and preserves unknown/ambiguous frames."""

from pathlib import Path
import pytest
from a1_tracker.evaluation import _common
from a1_tracker.evaluation.frame import assess_frame
from a1_tracker.evaluation.metrics import state
from a1_tracker.target.a2 import A2Config, A2Target


def test_a2_has_no_historical_bases_or_imports():
    assert A2Target.__bases__ == (object,)
    assert A2Config.__bases__ == (object,)
    folder = Path(__file__).resolve().parents[1] / "src/a1_tracker/target/a2"
    for path in folder.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in [
            "target.conservative",
            "target.multi_template",
            "target.soft_motion",
            "target.evidence",
        ]:
            assert name not in text


def test_project_assets_never_read_an_absolute_source(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "draft.csv"
    external.write_text("old output")
    monkeypatch.setattr(_common, "ROOT", root)
    assert _common.resolve_asset(str(external), required=False) is None
    with pytest.raises(FileNotFoundError):
        _common.resolve_asset(str(external))


def test_project_assets_resolve_from_repository_not_working_directory(tmp_path, monkeypatch):
    (tmp_path / "tracks.csv").write_text("current output")
    monkeypatch.setattr(_common, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    assert _common.resolve_asset("tracks.csv") == tmp_path / "tracks.csv"


def test_project_assets_reject_parent_traversal_and_windows_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "ROOT", tmp_path)
    windows_absolute = str(_common.PureWindowsPath("Z:") / "old-project" / "tracks.csv")
    for value in ("../outside.csv", windows_absolute):
        with pytest.raises(FileNotFoundError):
            _common.resolve_asset(value)


def test_reports_save_project_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "ROOT", tmp_path)
    report = {"assets": [{"path": str(tmp_path / "data/gt.txt"), "sha256": "hash"}]}
    assert _common.portable_record(report)["assets"][0]["path"] == "data/gt.txt"


def test_regenerated_inputs_rebind_ids_keep_starts_and_failed_cases(tmp_path):
    import numpy as np
    from a1_tracker.evaluation.protocol import rebind_protocol

    folder = tmp_path / "reproduction_inputs/sequence"
    folder.mkdir(parents=True)
    (folder / "tracks.txt").write_text("1,7,0,0,10,20,0.9,-1,-1,-1\n")
    (folder / "motion.csv").write_text("frame\n1\n2\n")
    (folder / "remaps.jsonl").write_text("")
    np.save(folder / "features.npy", np.ones((1, 512), dtype=np.float32))
    np.save(folder / "valid.npy", np.ones(1, dtype=bool))
    base = {
        "sequences": {"sequence": {"t2_tracks": "reproduction_inputs/sequence/tracks.txt"}},
        "files": [],
        "cases": [
            {"sequence": "sequence", "gt_id": 1, "selection_frame": 1,
             "end_frame": 2, "selection_bbox_xyxy": [0, 0, 10, 20], "initial_tracker_id": 99},
            {"sequence": "sequence", "gt_id": 2, "selection_frame": 2,
             "end_frame": 2, "selection_bbox_xyxy": [0, 0, 10, 20], "initial_tracker_id": 99},
        ],
    }
    result = rebind_protocol(base, tmp_path / "reproduction_inputs", tmp_path)
    assert [case["initial_tracker_id"] for case in result["cases"]] == [7, None]
    assert [case["selection_frame"] for case in result["cases"]] == [1, 2]
    assert base["cases"][0]["initial_tracker_id"] == 99
    assert result["initialization"]["failed"] == 1
    (folder / "motion.csv").write_text("frame\n1\n")
    with pytest.raises(ValueError, match="Incomplete sequence"):
        rebind_protocol(base, tmp_path / "reproduction_inputs", tmp_path)


def test_ambiguous_overlap_is_not_forced_to_be_correct():
    gt = {1: {1: ((0, 0, 40, 80), 0.9), 2: ((0, 0, 40, 80), 0.9)}}
    row = assess_frame(
        1,
        1,
        dict(tracker_id=3, state="TRACKING", reason="test", valid=True, bbox_xyxy=(0, 0, 40, 80)),
        gt,
    )
    assert row["ambiguous_overlap"]
    assert state(row) == "A"


def test_frozen_split_alias_is_normalized_only_at_boundary():
    assert _common.canonical_split("historical") == "validation"
    assert _common.canonical_split("test") == "test"
