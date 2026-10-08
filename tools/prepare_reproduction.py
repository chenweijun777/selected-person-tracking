"""Copy current, hash-verified offline assets from a local experiment source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((ROOT / "manifests/reproduction_assets.json").read_text(encoding="utf-8"))[
        "assets"
    ]
    for row in rows:
        destination = ROOT / row["destination"]
        if destination.is_file():
            source = destination
        else:
            portable = args.source_dir / row["destination"]
            source = portable if portable.is_file() else args.source_dir / row["source"]
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing {row['destination']}; use a complete local input bundle or rebuild "
                "it with tools/build_sequence_inputs.py (see docs/EVALUATION.md)"
            )
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {source}")
        if source != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    print("Verified", len(rows), "frozen assets")


if __name__ == "__main__":
    main()
