"""Prepare evaluation bindings for newly generated tracker inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from a1_tracker.evaluation._common import project_path
from a1_tracker.evaluation.protocol import rebind_protocol

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "validation", "test"], required=True)
    parser.add_argument("--inputs", type=Path, default=ROOT / "reproduction_inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "protocols/current" / f"{args.split}.json"
    result = rebind_protocol(
        json.loads(source.read_text(encoding="utf-8")), project_path(args.inputs), ROOT
    )
    result["source_protocol_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    output = project_path(args.output)
    if not output.is_relative_to(ROOT):
        raise ValueError("Protocol output must be inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(result["initialization"], indent=2))


if __name__ == "__main__":
    main()
