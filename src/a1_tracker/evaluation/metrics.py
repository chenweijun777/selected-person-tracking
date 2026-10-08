"""Common metric formulas for frozen target-recovery comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median


def split(sequence: str) -> str:
    number = sequence.split("-")[1]
    if number in {"02", "04", "05"}:
        return "development"
    if number in {"09", "10"}:
        return "validation"
    if number == "11":
        return "test"
    raise ValueError(f"Unexpected sequence: {sequence}")


def state(row: dict) -> str:
    if str(row["valid"]).lower() != "true":
        return "R"
    if str(row["ambiguous_overlap"]).lower() == "true":
        return "A"
    return {"correct": "C", "other": "W", "unmatched": "U"}[row["category"]]


def pct(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 2) if denominator else None


def initialization_metrics(rows: list[dict], modes: tuple[str, ...], detector: str) -> dict:
    groups = defaultdict(dict)
    for row in rows:
        if row["detector"] != detector or row["strategy"] not in modes:
            continue
        key = (row["sequence"], row["gt_id"])
        group = groups[(split(row["sequence"]), row["strategy"])]
        if key in group:
            raise ValueError(f"Duplicate selected person: {key}, {row['strategy']}")
        group[key] = str(row["initialization_success"]).lower() == "true"
    result = {}
    for subset in ("development", "validation", "test"):
        keys = [set(groups[subset, mode]) for mode in modes]
        if any(key != keys[0] for key in keys[1:]):
            raise ValueError(f"Different selected people between methods: {subset}")
        result[subset] = {}
        for mode in modes:
            success = sum(groups[subset, mode].values())
            result[subset][mode] = {
                "eligible_people": len(keys[0]),
                "success": success,
                "success_pct": pct(success, len(keys[0])),
            }
    return result


def frame_metrics(rows: list[dict], modes: tuple[str, ...], detector: str) -> dict:
    groups = defaultdict(dict)
    for row in rows:
        if row["detector"] != detector or row["strategy"] not in modes:
            continue
        key = (row["sequence"], row["gt_id"], row["frame"])
        group = groups[(split(row["sequence"]), row["strategy"])]
        if key in group:
            raise ValueError(f"Duplicate person-frame: {key}, {row['strategy']}")
        if float(row["visibility"]) >= 0.3:
            group[key] = state(row)
    result = {}
    for subset in ("development", "validation", "test"):
        keys = [set(groups[subset, mode]) for mode in modes]
        if any(key != keys[0] for key in keys[1:]):
            raise ValueError(f"Different evaluated person-frames between methods: {subset}")
        result[subset] = {}
        for mode in modes:
            counts = Counter(groups[subset, mode].values())
            denominator = len(keys[0])
            result[subset][mode] = {
                "person_frames": denominator,
                "C": counts["C"],
                "W": counts["W"],
                "R": counts["R"],
                "U": counts["U"],
                "A": counts["A"],
                "correct_pct": pct(counts["C"], denominator),
                "wrong_pct": pct(counts["W"], denominator),
                "reject_pct": pct(counts["R"], denominator),
            }
    return result


def summarize_event_cohort(cohort: list[dict], modes: tuple[str, ...], baseline: str) -> dict:
    base_stable = {r["event_id"] for r in cohort if r[f"{baseline}_stable_correct_start"]}
    result = {
        "events": len(cohort),
        "unreviewed_events": sum(r["event_review_status"] != "reviewed" for r in cohort),
        "methods": {},
    }
    for mode in modes:
        stable = {r["event_id"] for r in cohort if r[f"{mode}_stable_correct_start"]}
        wrong = sum(bool(r[f"{mode}_first_wrong_frame"]) for r in cohort)
        delays = [
            float(r[f"{mode}_delay_s"])
            for r in cohort
            if r[f"{mode}_stable_correct_start"] and r[f"{mode}_delay_s"] not in (None, "")
        ]
        result["methods"][mode] = {
            "stable_events": len(stable),
            "stable_pct": pct(len(stable), len(cohort)),
            "wrong_events": wrong,
            "wrong_pct": pct(wrong, len(cohort)),
            "gain_over_baseline": len(stable - base_stable),
            "loss_from_baseline": len(base_stable - stable),
            "preserved_baseline": len(stable & base_stable),
            "preservation_pct": pct(len(stable & base_stable), len(base_stable)),
            "median_delay_s": round(median(delays), 3) if delays else None,
        }
    return result


def event_metrics(rows: list[dict], modes: tuple[str, ...], baseline: str) -> dict:
    result = {}
    for subset in ("development", "validation", "test"):
        cohort = [
            r
            for r in rows
            if ("validation" if r["split"] == "historical" else r["split"]) == subset
            and str(r["common_cohort"]).lower() == "true"
        ]
        if len({r["event_id"] for r in cohort}) != len(cohort):
            raise ValueError(f"Duplicate event ID in {subset}")
        result[subset] = summarize_event_cohort(cohort, modes, baseline)
        outage = [
            r
            for r in cohort
            if str(r.get(f"{baseline}_ever_rejected_before_stable", "")).lower() == "true"
        ]
        result[subset]["a0_outage"] = summarize_event_cohort(outage, modes, baseline)
    return result
