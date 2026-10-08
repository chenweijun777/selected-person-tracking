"""Evaluation stays independent of draft code and preserves unknown/ambiguous frames."""

import json
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


def test_portable_mapping_never_falls_back_to_an_absolute_source(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "protocols").mkdir(parents=True)
    (root / "protocols/path_map.json").write_text("{}")
    external = tmp_path / "draft.csv"
    external.write_text("old output")
    monkeypatch.setattr(_common, "ROOT", root)
    assert _common.resolve_asset(str(external), required=False) is None
    with pytest.raises(FileNotFoundError):
        _common.resolve_asset(str(external))


def test_portable_mapping_uses_the_copied_asset(tmp_path, monkeypatch):
    (tmp_path / "protocols").mkdir()
    (tmp_path / "tracks.csv").write_text("current output")
    (tmp_path / "protocols/path_map.json").write_text(
        json.dumps({"frozen/original.csv": "tracks.csv"})
    )
    monkeypatch.setattr(_common, "ROOT", tmp_path)
    assert _common.resolve_asset("frozen/original.csv") == tmp_path / "tracks.csv"


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
