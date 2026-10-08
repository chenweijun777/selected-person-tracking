"""Prepare, verify, baseline-evaluate and fine-tune YOLO26s on CrowdHuman."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("CROWDHUMAN_RUN", ROOT / "outputs/crowdhuman_training"))
SOURCE = Path(os.environ.get("CROWDHUMAN_SOURCE", ROOT / "data/CrowdHuman"))
os.environ.setdefault("CROWDHUMAN_RUN", str(RUN))
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUN / "ultralytics_config"))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLCONFIGDIR", str(RUN / "matplotlib"))

import cv2
import torch
import yaml
from crowdhuman_training import (
    CrowdHumanTrainer,
    CrowdHumanValidator,
    assert_person_head,
    check_batch,
    install_criterion,
    log_epoch,
    log_progress,
)

import ultralytics
from ultralytics import YOLO

SETTINGS = {
    "epochs": 100,
    "patience": 0,
    "imgsz": 960,
    "batch": 8,
    "nbs": 64,
    "device": 0,
    "workers": 4,
    "seed": 42,
    "deterministic": True,
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "lrf": 0.01,
    "cos_lr": True,
    "warmup_epochs": 3,
    "warmup_bias_lr": 0.0,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "amp": "bf16",
    "freeze": 0,
    "multi_scale": 0.0,
    "mosaic": 0.5,
    "close_mosaic": 10,
    "fliplr": 0.5,
    "flipud": 0.0,
    "scale": 0.3,
    "translate": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.5,
    "hsv_v": 0.3,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "cache": False,
    "val": True,
    "save": True,
    "save_period": 5,
    "plots": True,
    "classes": [0],
    "conf": 0.001,
    "iou": 0.7,
    "nms": True,
    "max_det": 1000,
    "verbose": True,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def status(stage, **details):
    row = {
        "stage": stage,
        "pid": os.getpid(),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        **details,
    }
    save(RUN / "status.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def normalized_box(box, width, height):
    x, y, w, h = map(float, box)
    lo_x, lo_y = max(0.0, x), max(0.0, y)
    hi_x, hi_y = min(float(width), x + w), min(float(height), y + h)
    if hi_x <= lo_x or hi_y <= lo_y:
        return None
    return [
        (lo_x + hi_x) / (2 * width),
        (lo_y + hi_y) / (2 * height),
        (hi_x - lo_x) / width,
        (hi_y - lo_y) / height,
    ]


def prepare(smoke_only=False):
    data_root = RUN / "dataset"
    if (data_root / "audit.json").exists():
        audit = json.loads((data_root / "audit.json").read_text())
        if bool(audit.get("smoke_only", False)) != smoke_only:
            raise RuntimeError("Dataset scope differs; choose a separate output directory")
        for filename, digest in audit["source_hashes"].items():
            if sha256(SOURCE / filename) != digest:
                raise RuntimeError(f"Original manifest changed: {filename}")
        status("data_ready", audit=audit["splits"])
        return
    data_root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    source_hashes = {
        f"annotation_{s}.odgt": sha256(SOURCE / f"annotation_{s}.odgt") for s in ("train", "val")
    }
    audit = {"source": str(SOURCE), "source_hashes": source_hashes, "splits": {}}
    id_sets = []
    for source_split, split in (("train", "train"), ("val", "validation")):
        status("preparing_data", split=split)
        records = [
            json.loads(line)
            for line in (SOURCE / f"annotation_{source_split}.odgt")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if smoke_only:
            records = records[: 64 if split == "train" else 32]
        paths = (
            list(SOURCE.glob("CrowdHuman_train*/Images/*.jpg"))
            if split == "train"
            else list(SOURCE.glob("CrowdHuman_val/Images/*.jpg"))
        )
        lookup = {p.stem: p for p in paths}
        ids = [r["ID"] for r in records]
        if (
            len(set(ids)) != len(ids)
            or len(lookup) != len(paths)
            or any(i not in lookup for i in ids)
        ):
            raise RuntimeError("Duplicate ID or missing image")
        id_sets.append(set(ids))
        for kind in ("images", "labels", "ignore"):
            (data_root / kind / split).mkdir(parents=True, exist_ok=True)

        def convert(record, lookup=lookup, split=split):
            source = lookup[record["ID"]]
            image = cv2.imread(str(source))
            if image is None:
                raise RuntimeError(f"Image decode failed: {source}")
            height, width = image.shape[:2]
            del image
            # Reject damaged JPEGs so Ultralytics never tries to repair an image
            # sharing a hard link with a frozen original file.
            with source.open("rb") as stream:
                stream.seek(-2, 2)
                if stream.read() != b"\xff\xd9":
                    raise RuntimeError(f"JPEG missing end marker: {source}")
            counts = Counter(images=1)
            positive, ignored = [], []
            for box in record["gtboxes"]:
                is_ignore = box.get("tag") != "person" or box.get("extra", {}).get("ignore", 0)
                counts["raw_ignore" if is_ignore else "raw_person"] += 1
                x, y, w, h = box["fbox"]
                counts["out_of_bounds"] += int(x < 0 or y < 0 or x + w > width or y + h > height)
                converted = normalized_box(box["fbox"], width, height)
                if converted is None:
                    counts["dropped_ignore" if is_ignore else "dropped_person"] += 1
                    continue
                (ignored if is_ignore else positive).append(converted)
            target = data_root / "images" / split / source.name
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)

            def formatted(boxes):
                return "".join(" ".join(f"{v:.9f}" for v in row) + "\n" for row in boxes)

            (data_root / "labels" / split / (source.stem + ".txt")).write_text(
                formatted([[0, *b] for b in positive])
            )
            (data_root / "ignore" / split / (source.stem + ".txt")).write_text(formatted(ignored))
            counts["person"] = len(positive)
            counts["ignore"] = len(ignored)
            return counts

        total = Counter()
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, counts in enumerate(pool.map(convert, records), 1):
                total.update(counts)
                if i % 500 == 0:
                    status(
                        "preparing_data",
                        split=split,
                        images=i,
                        total=len(records),
                        seconds=round(time.monotonic() - started),
                    )
        shutil.copyfile(
            SOURCE / f"annotation_{source_split}.odgt", data_root / f"annotation_{split}.odgt"
        )
        audit["splits"][split] = dict(total)
        # Smoke combines the most crowded examples with a deterministic random
        # sample, exercising dense labels as well as ordinary augmentation.
        dense = sorted(
            records,
            key=lambda r: sum(
                g.get("tag") == "person" and not g.get("extra", {}).get("ignore", 0)
                for g in r["gtboxes"]
            ),
            reverse=True,
        )[:8]
        import random

        rng = random.Random(42)
        remaining = [r for r in records if r["ID"] not in {r["ID"] for r in dense}]
        chosen = dense + rng.sample(remaining, min(len(remaining), 56 if split == "train" else 24))
        (data_root / f"smoke_{split}.txt").write_text(
            "".join(str(data_root / "images" / split / (r["ID"] + ".jpg")) + "\n" for r in chosen)
        )
        (data_root / f"annotation_smoke_{split}.odgt").write_text(
            "".join(json.dumps(r) + "\n" for r in chosen)
        )
    if id_sets[0] & id_sets[1]:
        raise RuntimeError("Train/validation ID overlap")
    for smoke in (False, True):
        config = {
            "path": str(data_root),
            "train": "smoke_train.txt" if smoke else "images/train",
            "val": "smoke_validation.txt" if smoke else "images/validation",
            "names": {0: "person"},
        }
        (data_root / ("smoke.yaml" if smoke else "data.yaml")).write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
    for filename, digest in source_hashes.items():
        if sha256(SOURCE / filename) != digest:
            raise RuntimeError("Original annotations changed during conversion")
    audit["smoke_only"] = smoke_only
    save(data_root / "audit.json", audit)
    status("data_ready", audit=audit["splits"])


def initialize():
    target = RUN / "initializations/yolo26s_person_preserved.pt"
    if target.exists():
        assert_person_head(YOLO(str(target)).model)
        return target
    status("initializing")
    source_path = ROOT / "models/yolo26s.pt"
    model = YOLO(str(source_path)).model
    original = {k: v.clone() for k, v in model.state_dict().items()}
    head = model.model[-1]
    changed = []
    for branch in ("cv3", "one2one_cv3"):
        for i, sequence in enumerate(getattr(head, branch)):
            final = sequence[-1]
            assert final.in_channels == 128 and final.out_channels == 80
            replacement = torch.nn.Conv2d(128, 1, 1, bias=True)
            with torch.no_grad():
                replacement.weight.copy_(final.weight[:1])
                replacement.bias.copy_(final.bias[:1])
            sequence[-1] = replacement
            stem = f"model.{len(model.model) - 1}.{branch}.{i}.{len(sequence) - 1}"
            changed.extend([stem + ".weight", stem + ".bias"])
    head.nc = model.nc = 1
    head.no = 1 + head.reg_max * 4
    model.names = {0: "person"}
    model.yaml = copy.deepcopy(model.yaml)
    model.yaml.update(nc=1, person_cls_hidden=128)
    assert_person_head(model)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key][:1] if key in changed else original[key]), key
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model,
            "ema": None,
            "epoch": -1,
            "optimizer": None,
            "train_args": {"task": "detect"},
            "version": ultralytics.__version__,
        },
        target,
    )
    loaded = YOLO(str(target)).model
    assert_person_head(loaded)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
    save(
        RUN / "initializations/provenance.json",
        {
            "source": str(source_path),
            "source_sha256": sha256(source_path),
            "initialization": str(target),
            "initialization_sha256": sha256(target),
            "sliced_tensors": changed,
            "preserved_tensor_count": len(original) - len(changed),
            "reinitialized_tensors": [],
            "save_reload_exact": True,
            "hidden_channels": 128,
        },
    )
    return target


def train(smoke=False, batch=8, resume=False):
    audit = json.loads((RUN / "dataset/audit.json").read_text(encoding="utf-8"))
    if not smoke and audit.get("smoke_only", False):
        raise RuntimeError(
            "Full training requires the complete dataset; prepare a new output directory"
        )
    name = "smoke" if smoke else "train"
    settings = dict(
        SETTINGS,
        batch=batch,
        data=str(RUN / "dataset" / ("smoke.yaml" if smoke else "data.yaml")),
        project=str(RUN),
        name=name,
        exist_ok=False,
    )
    if smoke:
        settings.update(epochs=1, workers=0, close_mosaic=0, save_period=-1)
    status("smoke_training" if smoke else "training", settings=settings)
    model = YOLO(str(RUN / "train/weights/last.pt") if resume else str(initialize()))
    model.add_callback("on_train_start", install_criterion)
    model.add_callback("on_train_batch_end", check_batch)
    model.add_callback("on_train_epoch_end", log_epoch)
    model.add_callback("on_fit_epoch_end", log_progress)
    if resume:
        settings["resume"] = True
    model.train(trainer=CrowdHumanTrainer, **settings)
    last = RUN / name / "weights/last.pt"
    best = RUN / name / "weights/best.pt"
    for path in (last, best):
        assert_person_head(YOLO(str(path)).model)
    status(
        "smoke_complete" if smoke else "complete",
        best=str(best),
        last=str(last),
        best_sha256=sha256(best),
        last_sha256=sha256(last),
    )


def baseline():
    status("baseline_evaluation")
    validator = CrowdHumanValidator(
        args={
            "model": str(ROOT / "models/yolo26s.pt"),
            "data": str(RUN / "dataset/data.yaml"),
            "imgsz": 960,
            "batch": 8,
            "device": SETTINGS["device"],
            "workers": 4,
            "classes": [0],
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 1000,
            "nms": True,
            "quantize": 16,
            "plots": False,
            "save_json": False,
            "project": str(RUN),
            "name": "baseline",
            "exist_ok": False,
        }
    )
    metrics = validator()
    save(RUN / "baseline/summary.json", metrics)
    status("baseline_complete", metrics=metrics)


def main():
    global RUN, SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "initialize", "smoke", "baseline", "train", "pipeline")
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Prepare 64 train / 32 validation images for the one-epoch installation check",
    )
    args = parser.parse_args()
    if args.smoke_only and args.stage != "prepare":
        parser.error("--smoke-only is only supported by the prepare stage")
    RUN, SOURCE = args.output.resolve(), args.source.resolve()
    import crowdhuman_training as training

    training.RUN = RUN
    SETTINGS["device"] = args.device
    if args.device == "cpu":
        SETTINGS["amp"] = False
    RUN.mkdir(parents=True, exist_ok=True)
    if ultralytics.__version__ != "8.4.158":
        raise RuntimeError("Training integration requires Ultralytics 8.4.158")
    save(
        RUN / "approved_config.json",
        {
            "source": str(SOURCE),
            "initial_model": str(ROOT / "models/yolo26s.pt"),
            "settings": SETTINGS,
            "selection": "lowest full-precision CrowdHuman MR^-2; ties higher AP",
            "ignore": "IoA>=0.5 masks unmatched classification terms in both branches; positives take precedence",
            "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__},
            "source_hashes": {
                p.name: sha256(p) for p in (Path(__file__), ROOT / "tools/crowdhuman_training.py")
            },
        },
    )
    try:
        if args.stage == "pipeline":
            prepare()
            initialize()
            if not (RUN / "smoke/weights/best.pt").exists():
                train(smoke=True, batch=args.batch)
            if not (RUN / "baseline/summary.json").exists():
                baseline()
            train(batch=args.batch, resume=args.resume)
        elif args.stage == "prepare":
            prepare(smoke_only=args.smoke_only)
        elif args.stage == "initialize":
            initialize()
        elif args.stage == "smoke":
            train(smoke=True, batch=args.batch)
        elif args.stage == "baseline":
            baseline()
        else:
            train(batch=args.batch, resume=args.resume)
    except BaseException as error:
        status("failed", failed_stage=args.stage, error=repr(error))
        raise


if __name__ == "__main__":
    main()
