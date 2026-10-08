"""CrowdHuman ignore-aware YOLO26 training, pinned to Ultralytics 8.4.158.

Ignore boxes share every geometric augmentation with positive boxes, then are
removed before assignment. Only unmatched predictions with IoA >= .5 have their
classification loss masked. Both YOLO26 branches retain the stock box loss and
progressive branch weighting.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("CROWDHUMAN_RUN", ROOT / "outputs/crowdhuman_yolo26s_20261006"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUN / "ultralytics_config"))

import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER, nms, ops
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model
from vendor.crowdhuman_eval.database import Database


def ignore_prediction_mask(boxes, ignored, foreground, threshold=0.5):
    """Return unmatched predictions covered by ignore regions, in bounded chunks."""
    mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    boxes, ignored = boxes.float(), ignored.float()
    for start in range(0, len(boxes), 2048):
        current = boxes[start : start + 2048]
        if not len(ignored):
            break
        lo = torch.maximum(current[:, None, :2], ignored[None, :, :2])
        hi = torch.minimum(current[:, None, 2:], ignored[None, :, 2:])
        intersection = (hi - lo).clamp(min=0).prod(-1)
        area = (current[:, 2:] - current[:, :2]).clamp(min=0).prod(-1).clamp(min=1e-7)
        mask[start : start + len(current)] = (intersection / area[:, None]).amax(1) >= threshold
    return mask & ~foreground


class CrowdHumanDataset(YOLODataset):
    def get_labels(self):
        labels = super().get_labels()
        self.ignored = {}
        for label in labels:
            image = Path(label["im_file"])
            sidecar = image.parents[2] / "ignore" / image.parent.name / (image.stem + ".txt")
            if not sidecar.exists():
                raise FileNotFoundError(sidecar)
            content = sidecar.read_text().strip()
            self.ignored[str(image)] = np.array(
                [list(map(float, row.split())) for row in content.splitlines()] if content else [],
                dtype=np.float32,
            ).reshape(-1, 4)
        return labels

    def update_labels_info(self, label):
        ignored = self.ignored[label["im_file"]]
        label["bboxes"] = np.concatenate((label["bboxes"], ignored))
        label["cls"] = np.concatenate((label["cls"], -np.ones((len(ignored), 1), dtype=np.float32)))
        return super().update_labels_info(label)

    @staticmethod
    def collate_fn(batch):
        result = YOLODataset.collate_fn(batch)
        ignored = result["cls"].flatten() < 0
        result["ignore_bboxes"] = result["bboxes"][ignored]
        result["ignore_batch_idx"] = result["batch_idx"][ignored]
        for key in ("cls", "bboxes", "batch_idx"):
            result[key] = result[key][~ignored]
        return result


class IgnoreDetectionLoss(v8DetectionLoss):
    masked_predictions = 0
    # get_assigned_targets_and_loss is appended from the pinned upstream source
    # below; the sole change is masking negative BCE terms after assignment.

    def mask_classification(self, bce_loss, pred_bboxes, stride_tensor, foreground, batch, imgsz):
        ignored = batch.get("ignore_bboxes")
        if ignored is None or not len(ignored):
            return bce_loss
        from ultralytics.utils.ops import xywh2xyxy

        ignored = xywh2xyxy(ignored.to(self.device).float()) * imgsz[[1, 0, 1, 0]].float()
        image_indices = batch["ignore_batch_idx"].to(self.device)
        decoded = pred_bboxes.detach().float() * stride_tensor.float()
        masks = torch.stack(
            [
                ignore_prediction_mask(decoded[i], ignored[image_indices == i], foreground[i])
                for i in range(len(decoded))
            ]
        )
        self.masked_predictions += int(masks.sum().item())
        return bce_loss * (~masks).unsqueeze(-1)

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp_(min=1)  # on the device: no host sync

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        bce_loss = self.mask_classification(
            bce_loss, pred_bboxes, stride_tensor, fg_mask, batch, imgsz
        )
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss: zero on an empty foreground, and pred_distri stays in the graph either way
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )  # loss(box, cls, dfl)


def build_dataset(args, path, data, batch, mode, stride=32):
    return CrowdHumanDataset(
        img_path=path,
        imgsz=args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=args,
        rect=mode == "val",
        cache=False,
        single_cls=False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        task="detect",
        classes=args.classes,
        data=data,
        fraction=1.0,
    )


class CrowdHumanValidator(DetectionValidator):
    def postprocess(self, preds):
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            classes=[0],
            multi_label=True,
            agnostic=False,
            max_det=self.args.max_det,
            end2end=False,
        )
        return [
            {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
            for x in outputs
        ]

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_dataset(
            self.args, img_path, self.data, batch or self.args.batch, mode, self.stride
        )

    def init_metrics(self, model):
        super().init_metrics(model)
        self.crowd_predictions = []
        self.crowd_epoch = getattr(getattr(self, "owner", None), "epoch", -1) + 1
        self.crowd_gt = RUN / "dataset/annotation_validation.odgt"
        if "smoke" in str(self.args.data):
            self.crowd_gt = RUN / "dataset/annotation_smoke_validation.odgt"
        self.end2end = False
        if self.training:
            model.criterion = E2ELoss(model, loss_fn=IgnoreDetectionLoss)
            epoch = self.crowd_epoch
            model.criterion.o2m = model.criterion.decay(max(epoch - 1, 0))
            model.criterion.o2o = 1 - model.criterion.o2m

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            keep = pred["cls"] == 0
            boxes = ops.scale_boxes(
                batch["img"].shape[2:],
                pred["bboxes"][keep].clone(),
                batch["ori_shape"][si],
                batch["ratio_pad"][si],
            )
            boxes = ops.xyxy2ltwh(boxes).float().cpu().tolist()
            scores = pred["conf"][keep].float().cpu().tolist()
            self.crowd_predictions.append(
                {
                    "ID": Path(batch["im_file"][si]).stem,
                    "height": int(batch["ori_shape"][si][0]),
                    "width": int(batch["ori_shape"][si][1]),
                    "dtboxes": [{"box": b, "score": s, "tag": 1} for b, s in zip(boxes, scores)],
                }
            )

    def get_stats(self):
        stats = super().get_stats()
        stage = "epoch" if self.training else "standalone"
        directory = self.save_dir / "crowdhuman"
        directory.mkdir(parents=True, exist_ok=True)
        detections = directory / f"{stage}_{self.crowd_epoch:03d}.odgt"
        detections.write_text(
            "".join(json.dumps(p) + "\n" for p in self.crowd_predictions), encoding="utf-8"
        )
        # Keep the original annotation bytes for official matching. Smoke uses an
        # explicit subset so every annotated image has a corresponding prediction.
        expected = {
            json.loads(line)["ID"]
            for line in self.crowd_gt.read_text(encoding="utf-8").splitlines()
        }
        actual = [p["ID"] for p in self.crowd_predictions]
        if set(actual) != expected or len(actual) != len(expected):
            raise RuntimeError("Validation image cohort does not match the frozen ODGT")
        database = Database(str(self.crowd_gt), str(detections), "box", None, 0)
        database.compare(thres=0.5)
        ap, _ = database.eval_AP()
        mr, _ = database.eval_MR()
        recall = sum(match == 1 for _, match, _ in database.scorelist) / (
            database._gtNum - database._ignNum
        )
        if not np.isfinite([ap, mr, recall]).all():
            raise FloatingPointError("Nonfinite CrowdHuman evaluation")
        stats.update(
            {
                "crowdhuman/AP": float(ap),
                "crowdhuman/MR_minus2": float(mr),
                "crowdhuman/Recall": float(recall),
                "fitness": float(1 - mr),
            }
        )
        result = {
            "epoch": self.crowd_epoch,
            "stage": stage,
            "images": len(actual),
            "AP": float(ap),
            "MR_minus2": float(mr),
            "Recall": float(recall),
            "inference": {
                "imgsz": self.args.imgsz,
                "conf": self.args.conf,
                "iou": self.args.iou,
                "max_det": self.args.max_det,
                "nms": True,
            },
            "ultralytics_metrics_ignore_naive_diagnostic_only": {
                k: v for k, v in stats.items() if k.startswith("metrics/")
            },
        }
        (directory / f"{stage}_{self.crowd_epoch:03d}.json").write_text(
            json.dumps(result, indent=2)
        )
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        LOGGER.info(f"CrowdHuman AP={ap:.5f} MR^-2={mr:.5f} Recall={recall:.5f}")
        return stats


class CrowdHumanTrainer(DetectionTrainer):
    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        record = self.save_dir / "best_selection.json"
        if self.resume:
            if not record.exists():
                raise RuntimeError("Resume requires the full-precision best-selection record")
            best = json.loads(record.read_text())
            self.best_crowdhuman = (best["MR_minus2"], best["AP"])

    def save_model(self):
        saved = super().save_model()
        if self.best_fitness == self.fitness:
            record = {
                "epoch": self.epoch + 1,
                "MR_minus2": self.best_crowdhuman[0],
                "AP": self.best_crowdhuman[1],
                "fitness_counter": self.best_fitness,
            }
            (self.save_dir / "best_selection.json").write_text(json.dumps(record, indent=2))
        return saved

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None:
            raise RuntimeError("A preserved pretrained person head is required")
        assert_person_head(weights)
        model = copy.deepcopy(weights).float()
        for name, tensor in weights.state_dict().items():
            if not torch.equal(model.state_dict()[name], tensor.float()):
                raise RuntimeError(f"Initialization changed: {name}")
        model.requires_grad_(True)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_dataset(self.args, img_path, self.data, batch or self.args.batch, mode, stride)

    def get_validator(self):
        validator = CrowdHumanValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy.copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.owner = self
        return validator

    def validate(self):
        metrics = self.validator(self)
        # BaseValidator rounds returned floats; checkpoint selection uses the
        # unrounded evaluation result saved by get_stats instead.
        result = json.loads(
            (self.save_dir / "crowdhuman" / f"epoch_{self.epoch + 1:03d}.json").read_text()
        )
        mr, ap = result["MR_minus2"], result["AP"]
        best = getattr(self, "best_crowdhuman", None)
        better = best is None or mr < best[0] or (mr == best[0] and ap > best[1])
        if better:
            self.best_crowdhuman = (mr, ap)
            # Fitness must compare monotonically; the exact MR/AP decision is
            # made above rather than a weighted approximation of the tie break.
            self.best_fitness = (self.best_fitness or 0.0) + 1.0
            fitness = self.best_fitness
        else:
            fitness = self.best_fitness - 1.0
        metrics.pop("fitness", None)
        return metrics, fitness


def assert_person_head(model):
    assert model.nc == model.model[-1].nc == 1
    assert model.names == {0: "person"}
    for branch in ("cv3", "one2one_cv3"):
        assert [
            (s[-1].in_channels, s[-1].out_channels) for s in getattr(model.model[-1], branch)
        ] == [(128, 1)] * 3


def install_criterion(trainer):
    model = unwrap_model(trainer.model)
    assert_person_head(model)
    model.criterion = E2ELoss(model, loss_fn=IgnoreDetectionLoss)
    if trainer.start_epoch > 0:
        model.criterion.updates = trainer.start_epoch
        model.criterion.o2m = model.criterion.decay(trainer.start_epoch)
        model.criterion.o2o = 1 - model.criterion.o2m


def check_batch(trainer):
    if not torch.isfinite(trainer.loss).all():
        raise FloatingPointError(f"Nonfinite loss at epoch {trainer.epoch + 1}")


def log_epoch(trainer):
    criterion = unwrap_model(trainer.model).criterion
    row = {
        "epoch": trainer.epoch + 1,
        "batch": trainer.batch_size,
        "accumulate": trainer.accumulate,
        "lr": [g["lr"] for g in trainer.optimizer.param_groups],
        "ignore_masked_o2m": criterion.one2many.masked_predictions,
        "ignore_masked_o2o": criterion.one2one.masked_predictions,
        "cuda_peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    with (trainer.save_dir / "training_audit.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def log_progress(trainer):
    """Expose epoch progress and official metrics without a monitoring service."""
    if trainer.save_dir.name != "train":
        return
    row = {
        "stage": "training",
        "pid": os.getpid(),
        "completed_epochs": trainer.epoch + 1,
        "total_epochs": trainer.epochs,
        "metrics": trainer.metrics,
        "best": json.loads((trainer.save_dir / "best_selection.json").read_text()),
    }
    path = RUN / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2))
    temporary.replace(path)
