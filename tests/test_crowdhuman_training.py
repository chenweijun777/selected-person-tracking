"""Contract tests for ignore supervision, head preservation and official matching."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(ROOT / "outputs/crowdhuman_yolo26s_20261006/ultralytics_config")
)

import pytest
import torch
from crowdhuman_training import CrowdHumanDataset, IgnoreDetectionLoss, ignore_prediction_mask
from run_crowdhuman_yolo26s import normalized_box
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from vendor.crowdhuman_eval.database import Database

from ultralytics import YOLO


def test_full_training_refuses_a_smoke_only_dataset(tmp_path, monkeypatch):
    import json
    import run_crowdhuman_yolo26s as runner

    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset/audit.json").write_text(json.dumps({"smoke_only": True}))
    monkeypatch.setattr(runner, "RUN", tmp_path)
    with pytest.raises(RuntimeError, match="complete dataset"):
        runner.train(smoke=False)


def test_ignore_mask_uses_prediction_area_and_preserves_positives():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]])
    ignored = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert ignore_prediction_mask(boxes, ignored, torch.zeros(3, dtype=torch.bool)).tolist() == [
        True,
        False,
        False,
    ]
    assert ignore_prediction_mask(boxes, ignored, torch.tensor([True, False, False])).tolist() == [
        False,
        False,
        False,
    ]
    assert not ignore_prediction_mask(boxes, ignored[:0], torch.zeros(3, dtype=torch.bool)).any()


def test_collation_removes_ignores_from_assignment_but_preserves_alignment():
    def sample(classes):
        return {
            "img": torch.zeros(3, 32, 32),
            "cls": torch.tensor(classes).view(-1, 1),
            "bboxes": torch.arange(len(classes) * 4).reshape(-1, 4).float(),
            "batch_idx": torch.zeros(len(classes)),
        }

    batch = CrowdHumanDataset.collate_fn([sample([0, -1]), sample([-1, 0, -1])])
    assert batch["cls"].flatten().tolist() == [0, 0]
    assert batch["batch_idx"].tolist() == [0, 1]
    assert batch["ignore_batch_idx"].tolist() == [0, 1, 1]
    assert batch["ignore_bboxes"].shape == (3, 4)


def test_clipping_preserves_valid_box_and_rejects_outside_box():
    assert normalized_box([-10, -20, 50, 80], 100, 100) == [0.2, 0.3, 0.4, 0.6]
    assert normalized_box([200, 200, 10, 10], 100, 100) is None


@pytest.fixture(scope="module")
def predictions():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = YOLO(str(ROOT / "models/yolo26s.pt")).model.float().train()
    model.args = get_cfg()
    with torch.no_grad():
        preds = model(torch.randn(2, 3, 96, 96))
    batch = {
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.5], [0.5, 0.5, 0.25, 0.5]]),
        "ignore_batch_idx": torch.empty(0),
        "ignore_bboxes": torch.empty(0, 4),
    }
    return model, preds, batch


def test_no_ignores_matches_stock_progressive_loss_in_both_branches(predictions):
    model, preds, batch = predictions
    stock = E2ELoss(model)
    custom = E2ELoss(model, loss_fn=IgnoreDetectionLoss)
    expected, _ = stock(preds, batch)
    actual, _ = custom(preds, batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stock.update()
    custom.update()
    assert (stock.o2m, stock.o2o) == (custom.o2m, custom.o2o)


@pytest.mark.parametrize("branch,topk,topk2", [("one2many", 10, None), ("one2one", 7, 1)])
def test_ignored_negatives_have_zero_gradient_and_box_losses_are_unchanged(
    predictions, branch, topk, topk2
):
    model, preds, original = predictions
    batch = dict(
        original,
        ignore_batch_idx=torch.tensor([0.0, 1.0]),
        ignore_bboxes=torch.tensor([[0.5, 0.5, 20.0, 20.0], [0.5, 0.5, 20.0, 20.0]]),
    )
    current = dict(preds[branch])
    current["scores"] = current["scores"].detach().clone().requires_grad_(True)
    stock = v8DetectionLoss(model, tal_topk=topk, tal_topk2=topk2)
    custom = IgnoreDetectionLoss(model, tal_topk=topk, tal_topk2=topk2)
    assigned, expected, _ = stock.get_assigned_targets_and_loss(current, batch)
    _, actual, _ = custom.get_assigned_targets_and_loss(current, batch)
    torch.testing.assert_close(actual[[0, 2]], expected[[0, 2]], rtol=0, atol=0)
    assert actual[1] < expected[1]
    actual[1].backward()
    gradient = current["scores"].grad.permute(0, 2, 1)
    assert assigned[0].any(), "Fixture must exercise valid positive assignments"
    assert gradient[assigned[0]].abs().sum() > 0
    assert gradient[~assigned[0]].abs().sum() == 0
    assert custom.masked_predictions > 0


def test_official_evaluator_ignores_predictions_and_counts_duplicate_false_positives(tmp_path):
    import json

    positive = {
        "tag": "person",
        "fbox": [10, 10, 20, 40],
        "hbox": [10, 10, 20, 10],
        "extra": {"ignore": 0},
    }
    ignored = {
        "tag": "mask",
        "fbox": [60, 10, 20, 40],
        "hbox": [60, 10, 20, 10],
        "extra": {"ignore": 1},
    }
    gt = tmp_path / "gt.odgt"
    dt = tmp_path / "dt.odgt"
    gt.write_text(json.dumps({"ID": "sample", "gtboxes": [positive, ignored]}) + "\n")
    dt.write_text(
        json.dumps(
            {
                "ID": "sample",
                "width": 100,
                "height": 100,
                "dtboxes": [
                    {"box": [60, 10, 20, 40], "score": 0.99},
                    {"box": [10, 10, 20, 40], "score": 0.9},
                    {"box": [10, 10, 20, 40], "score": 0.8},
                ],
            }
        )
        + "\n"
    )
    database = Database(str(gt), str(dt), "box", None, 0)
    database.compare(0.5)
    assert database._gtNum - database._ignNum == 1
    assert [r[1] for r in database.scorelist] == [1, 0]
