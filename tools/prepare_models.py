"""Copy already verified model files or download the pinned official assets."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from verify_models import ROOT, digest, models, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--source-dir", type=Path)
    parser.add_argument("--profile", choices=["full", "pretrained"], default="full")
    args = parser.parse_args()
    folder = ROOT / "models"
    folder.mkdir(parents=True, exist_ok=True)
    for spec in models(args.profile):
        path = folder / spec["file"]
        if path.exists():
            if digest(path) != spec["sha256"]:
                raise ValueError(f"Refusing to overwrite unrecognized model: {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            if args.source_dir:
                shutil.copyfile(args.source_dir / spec["file"], temporary)
            else:
                if not spec.get("url"):
                    raise ValueError(
                        "Fine-tuned best.pt has no published download yet; provide --source-dir or train it"
                    )
                print(f"Downloading {spec['file']}", flush=True)
                with (
                    urllib.request.urlopen(spec["url"], timeout=120) as src,
                    temporary.open("wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
            if digest(temporary) != spec["sha256"]:
                raise ValueError(f"Download/copy checksum mismatch: {spec['file']}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
    print(f"Verified {len(verify(folder, args.profile))} models in {folder}")


if __name__ == "__main__":
    main()
