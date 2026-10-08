"""Verify the exact two model files used by the selected online configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def models(profile="full"):
    specs = json.loads((ROOT / "manifests/models.json").read_text(encoding="utf-8"))["models"]
    return [s for s in specs if profile == "full" or s["file"] != "best.pt"]


def verify(folder, profile="full"):
    records = []
    for spec in models(profile):
        path = Path(folder) / spec["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run tools/prepare_models.py --download")
        actual = digest(path)
        if actual != spec["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path}")
        records.append({"file": spec["file"], "sha256": actual})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--profile", choices=["full", "pretrained"], default="full")
    args = parser.parse_args()
    print(json.dumps({"verified": verify(args.models_dir, args.profile)}, indent=2))
