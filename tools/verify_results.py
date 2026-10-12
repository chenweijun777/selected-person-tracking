"""Verify curated result integrity, denominators and current cross-table consistency."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "results" / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_deployment():
    expected = {"PyTorch", "ORT_FP32", "TRT_FP32", "TRT_FP16"}
    count = 0
    for name in ("deployment_detector.csv", "deployment_pipeline.csv", "deployment_quality.csv"):
        data = rows(name)
        if len(data) != 4 or {r["backend"] for r in data} != expected:
            raise ValueError(f"Expected four unique deployment backends: {name}")
        for row in data:
            if row["split"] != "validation":
                raise ValueError(f"Deployment split changed: {name}")
            if "mean_ms" in row:
                values = [float(row[k]) for k in ("mean_ms", "p50_ms", "p95_ms", "fps")]
                if not all(math.isfinite(v) and v > 0 for v in values):
                    raise ValueError("Deployment timings must be positive and finite")
                if abs(values[0] * values[3] - 1000) > .01 or values[2] < values[1]:
                    raise ValueError("Deployment FPS or timing percentiles are inconsistent")
            if name == "deployment_detector.csv":
                if int(row["frames"]) != 1179:
                    raise ValueError("Expected 1179 detector timing frames")
                tp, fp, fn = (int(row[k]) for k in (
                    "true_positives", "false_positives", "false_negatives"))
                if (abs(float(row["precision_percent"]) - 100 * tp / (tp + fp)) > .001
                        or abs(float(row["recall_percent"]) - 100 * tp / (tp + fn)) > .001):
                    raise ValueError("Deployment precision/recall differ from counts")
            elif name == "deployment_pipeline.csv":
                if int(row["samples"]) != 320 or int(row["target_valid_samples"]) != 288:
                    raise ValueError("Deployment pipeline sample counts changed")
            else:
                total = sum(int(row[k]) for k in ("C", "W", "U", "A", "R"))
                if (total != int(row["person_frames"]) or total != 9640
                        or int(row["targets"]) != 47 or int(row["frames"]) != 1179
                        or int(row["events"]) != 14):
                    raise ValueError("Deployment evaluation scope changed")
                if abs(float(row["coverage_percent"]) - 100 * int(row["C"]) / total) > .011:
                    raise ValueError("Deployment coverage differs from frame counts")
                if not 0 <= int(row["recovered"]) <= int(row["events"]):
                    raise ValueError("Invalid deployment recovery count")
            count += 1
    return count


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
    deployment_rows = verify_deployment()
    print(
        json.dumps(
            {
                "passed": True,
                "verified_files": len(frozen["files"]),
                "replayed_policy_rows": len(policies),
                "deployment_rows": deployment_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
