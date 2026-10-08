"""Fixed-window stable recovery assessment."""

import math
from collections import Counter
from itertools import pairwise

RECOVERY_KINDS = {"full_occlusion_reappearance", "partial_visibility_recovery"}


def maybe_int(value):
    return int(value) if value not in (None, "", "None") else None


def status(row):
    if row is None:
        return "N"
    if row["valid"] != "True":
        return "R"
    if row["ambiguous_overlap"] == "True":
        return "A"
    return {"correct": "C", "other": "W", "unmatched": "U"}[row["category"]]


def stable_start(frames, required):
    run_start = last = None
    count = 0
    for frame in sorted(frames):
        if last is None or frame != last + 1:
            run_start, count = (frame, 1)
        else:
            count += 1
        if count >= required:
            return (run_start, frame)
        last = frame
    return (None, None)


def assess_mode(event, frames, selection_frame, mode):
    start, end, fps = (event["start_frame"], event["context_end_frame"], event["fps"])
    anchor = event["reappearance_frame"]
    before = [
        r
        for f, r in frames.items()
        if max(1, start - math.ceil(fps)) <= f < start and float(r["visibility"]) > 0
    ]
    before.sort(key=lambda row: int(row["frame"]))
    pre = status(before[-1]) if before else "N"
    window = [frames[f] for f in range(start, end + 1) if f in frames]
    post = [r for r in window if anchor is not None and int(r["frame"]) >= anchor]
    main = [r for r in post if float(r["visibility"]) >= 0.3]
    visible = [r for r in post if float(r["visibility"]) > 0]
    stage = [r for r in window if anchor is not None and int(r["frame"]) <= anchor]
    correct = [int(r["frame"]) for r in visible if status(r) == "C"]
    sustain_start, sustain_confirm = stable_start(correct, math.ceil(0.2 * fps))
    wrong = [int(r["frame"]) for r in visible if status(r) == "W"]
    before_success = [r for r in window if sustain_start is None or int(r["frame"]) < sustain_start]
    lost = [int(r["frame"]) for r in before_success if status(r) == "R"]
    counts = Counter(status(r) for r in main)
    post_count = len(main)
    assert sum(counts.values()) == post_count
    first = next((int(r["frame"]) for r in post if status(r) == "C"), None)
    ids = [maybe_int(r["tracker_id"]) for r in visible if r["valid"] == "True"]
    switches = sum((a != b for a, b in pairwise(ids)))
    return {
        f"{mode}_pre_status": pre,
        f"{mode}_post_main_frames": post_count,
        f"{mode}_C": counts["C"],
        f"{mode}_W": counts["W"],
        f"{mode}_U": counts["U"],
        f"{mode}_R": counts["R"],
        f"{mode}_A": counts["A"],
        f"{mode}_first_correct_frame": first,
        f"{mode}_stable_correct_start": sustain_start,
        f"{mode}_stable_correct_confirm": sustain_confirm,
        f"{mode}_delay_s": round((sustain_start - anchor) / fps, 6)
        if sustain_start is not None and anchor is not None
        else None,
        f"{mode}_first_wrong_frame": wrong[0] if wrong else None,
        f"{mode}_wrong_before_stable": bool(
            wrong and (sustain_start is None or wrong[0] < sustain_start)
        ),
        f"{mode}_first_reject_frame": lost[0] if lost else None,
        f"{mode}_ever_rejected_before_stable": bool(lost),
        f"{mode}_stage_uncertain": any(status(r) in ("R", "W", "U", "A") for r in stage),
        f"{mode}_observed_id_changes": switches,
        f"{mode}_scoreable": selection_frame is not None
        and selection_frame <= start
        and (anchor is not None)
        and bool(main),
    }
