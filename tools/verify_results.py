"""Verify curated result integrity, denominators and current cross-table consistency."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "results" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    frozen = json.loads((ROOT / "manifests/public_integrity.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        path = ROOT / item["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Curated asset changed: {item['path']}")
    policies = rows("policy_reproduction.csv")
    index = {(r["split"], r["mode"]): r for r in policies}
    expected = {
        (s, m)
        for s in ("development", "validation", "test")
        for m in ("A0", "A1", "A2", "NoBank", "NoEMA")
    }
    if set(index) != expected or len(policies) != len(expected):
        raise ValueError("Expected one row per current split and policy")
    for row in policies:
        count = sum(int(row[k]) for k in ("C", "W", "R", "U", "A"))
        if count != int(row["person_frames"]):
            raise ValueError("Frame categories do not partition the denominator")
        if abs(float(row["coverage"]) - 100 * int(row["C"]) / count) > 0.011:
            raise ValueError("Coverage does not match frame counts")
        if not 0 <= int(row["common_recovered"]) <= int(row["common_events"]):
            raise ValueError("Invalid event counts")
        if row["formal_events"] == "0" and row["formal_rate"]:
            raise ValueError("A zero formal denominator must have an unavailable rate")
    for split in ("development", "validation"):
        final = next(
            r
            for r in rows("devval_system.csv")
            if r["split"] == split and r["group"] == "Finetuned_Joint"
        )
        replay = index[split, "A2"]
        for key in ("C", "W", "R", "U", "A"):
            if final[key] != replay[key]:
                raise ValueError(f"System/replay differ: {split}/{key}")
        if final["recovered"] != replay["common_recovered"]:
            raise ValueError(f"System/replay events differ: {split}")
    test = next(r for r in rows("test_system.csv") if r["group"] == "Finetuned_Joint")
    replay = index["test", "A2"]
    for key in ("C", "W", "R", "U", "A", "coverage", "targets", "person_frames"):
        if float(test[key]) != float(replay[key]):
            raise ValueError(f"Test system/replay differ: {key}")
    if test["fixed_cohort_success"] != replay["common_recovered"]:
        raise ValueError("Test common recovery count differs")
    print(
        json.dumps(
            {
                "passed": True,
                "verified_files": len(frozen["files"]),
                "replayed_policy_rows": len(policies),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
