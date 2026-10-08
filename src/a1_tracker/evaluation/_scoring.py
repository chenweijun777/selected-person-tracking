"""Scoring through fixed matching and target recovery definitions."""

from . import events as event_scoring
from . import frame as frame_scoring
from . import matching, mot
from . import metrics as metric_scoring

import shutil
from collections import Counter

import numpy as np

from ._common import EVENT_FIELDS, GLOBAL_FIELDS


def global_mot(files, output, gt_root):
    staging = output / "trackeval" / "inputs" / "run"
    staging.mkdir(parents=True)
    for seq, source in files.items():
        shutil.copyfile(source, staging / f"{seq}.txt")
    payload = mot.run(
        [seq.split("-")[1] for seq in files],
        ["run"],
        "unified",
        tracks_root=staging.parent,
        output_root=output / "trackeval",
        gt_root=gt_root,
    )
    result = payload["results"]["MotChallenge2DBox"]["run"]
    if not result or "COMBINED_SEQ" not in result:
        raise RuntimeError(f"TrackEval failed: {payload['messages']}")

    def extract(values):
        flat = {
            key: val
            for section in values["pedestrian"].values()
            if isinstance(section, dict)
            for key, val in section.items()
        }
        return {
            key: int(flat[key]) if key in ("IDSW", "Frag") else float(np.mean(flat[key])) * 100
            for key in GLOBAL_FIELDS
        }

    return extract(result["COMBINED_SEQ"]), {seq: extract(result[seq]) for seq in files}


def tracker_matches(predictions, truth):
    evaluator = matching
    from a1_tracker.types import Detection

    assigned = {}
    for seq, frames in truth.items():
        assigned[seq] = {}
        for frame, people in frames.items():
            preds = list(predictions[seq].get(frame, {}).values())
            gt = [evaluator.GroundTruth(pid, b, vis) for pid, (b, vis) in people.items()]
            matches = evaluator.match_detections(
                [Detection(p["bbox_xyxy"], min(1.0, max(0.0, p["score"]))) for p in preds],
                gt,
                [],
                0.5,
            )
            assigned[seq][frame] = {
                gt[g].track_id: preds[p]["pred_id"] for p, g, _ in matches.matches
            }
    return assigned


def target_assessment(cases, predictions, truth):
    assess = frame_scoring.assess_frame
    details, people, by_case = [], [], {}
    for c in cases:
        seq, pid, first = c["sequence"], c["gt_id"], c["selection_frame"]
        initial = predictions[seq, pid, first]
        people.append(
            {
                "sequence": seq,
                "gt_id": pid,
                "initialization_success": initial["valid"]
                and initial["tracker_id"] == c["initial_tracker_id"],
            }
        )
        frames = {}
        for f in range(first, c["end_frame"] + 1):
            pred = predictions[seq, pid, f]
            if pid not in truth[seq].get(f, {}):
                details.append(
                    dict(
                        sequence=seq,
                        gt_id=pid,
                        frame=f,
                        category="unknown",
                        visibility=None,
                        **pred,
                    )
                )
                continue
            row = dict(
                sequence=seq,
                **assess(f, pid, pred, truth[seq]),
                session_target_id=pred["session_target_id"],
                episode_id=pred["episode_id"],
            )
            details.append(row)
            frames[f] = {k: str(v) if v is not None else "" for k, v in row.items()}
        by_case[seq, pid] = frames
    return details, people, by_case


def event_details(
    events,
    protocol,
    sequences,
    strict,
    kind,
    truth,
    intervals,
    assignments=None,
    target_frames=None,
    predictions=None,
):
    evaluator = event_scoring
    fixed = {eid for group in protocol["events"].values() for eid in group["common_event_ids"]}
    rows = []
    for event in events:
        seq, pid = event["sequence"], event["gt_id"]
        if seq not in sequences:
            continue
        start, anchor, end = (
            event["start_frame"],
            event["reappearance_frame"],
            event["context_end_frame"],
        )
        last = event.get("last_visible_frame")
        pre_id = None
        if kind == "tracker":
            pre_id = assignments[seq].get(last, {}).get(pid)
            frames = {}
            for f in range(event["context_start_frame"], end + 1):
                if pid not in truth[seq].get(f, {}):
                    continue
                matched = assignments[seq].get(f, {})
                tid = matched.get(pid)
                bound_other = pre_id is not None and any(
                    p != pid and t == pre_id for p, t in matched.items()
                )
                bound_output = (
                    pre_id in predictions[seq].get(f, {}) if pre_id is not None else False
                )
                category = (
                    "correct"
                    if tid is not None
                    else "other"
                    if bound_other
                    else "unmatched"
                    if bound_output
                    else "reject"
                )
                frames[f] = {
                    "frame": str(f),
                    "visibility": str(truth[seq][f][pid][1]),
                    "valid": str(category != "reject"),
                    "ambiguous_overlap": "False",
                    "category": category,
                    "tracker_id": str(tid if tid is not None else pre_id)
                    if category != "reject"
                    else "",
                }
        else:
            frames = target_frames.get((seq, pid), {})
            pre = frames.get(last)
            if pre and pre["valid"] == "True":
                pre_id = int(pre["tracker_id"])
        # Missing GT is unknown. A manifest gap terminates the recovery observation window.
        gaps = [
            a
            for a, b, state in intervals.get((seq, pid), [])
            if state == "unannotated" and b >= start and a <= end
        ]
        bounded = dict(event, context_end_frame=min([end, *[g - 1 for g in gaps]]))
        measured = evaluator.assess_mode(bounded, frames, min(frames) if frames else None, "run")
        success_frame = measured["run_stable_correct_start"]
        post_id = int(frames[success_frame]["tracker_id"]) if success_frame is not None else None
        reasons = []
        if event["event_type"] not in evaluator.RECOVERY_KINDS:
            reasons.append("not_recovery_event")
        if not event.get("recovery_candidate"):
            reasons.append("not_candidate")
        if not measured["run_scoreable"]:
            reasons.append("no_scoreable_reappearance")
        if kind == "target_manager":
            if event["event_id"] not in fixed:
                reasons.append("outside_frozen_cohort")
        elif pre_id is None:
            reasons.append("no_correct_pre_event_binding")
        if strict:
            if event["review_status"] != "reviewed":
                reasons.append("needs_review")
            if measured["run_pre_status"] != "C":
                reasons.append("pre_event_not_correct")
        post = [
            r
            for f, r in frames.items()
            if anchor is not None
            and anchor <= f <= bounded["context_end_frame"]
            and float(r["visibility"]) > 0
        ]
        wrong = measured["run_first_wrong_frame"] is not None
        if kind == "tracker" and pre_id is not None:
            wrong = any(
                any(
                    p != pid and t == pre_id
                    for p, t in assignments[seq].get(int(r["frame"]), {}).items()
                )
                for r in post
            )
        row = {k: event.get(k) for k in EVENT_FIELDS[:7]}
        row.update(
            pre_event_pred_id=pre_id,
            post_event_pred_id=post_id,
            same_id_recovered=success_frame is not None
            and pre_id is not None
            and post_id == pre_id,
            new_id=success_frame is not None and pre_id is not None and post_id != pre_id,
            wrong_id=wrong,
            no_output=bool(post) and all(r["valid"] == "False" for r in post),
            recovery_success=success_frame is not None,
            recovery_delay_frames=success_frame - anchor if success_frame is not None else None,
            recovery_delay_seconds=measured["run_delay_s"],
            eligible=not reasons,
            exclusion_reason=";".join(reasons),
            provisional=not strict,
            review_status=event["review_status"],
            assessment_end=bounded["context_end_frame"],
            **measured,
        )
        rows.append(row)
    return rows


def recovery_summary(rows):
    eligible = [r for r in rows if r["eligible"]]
    n = len(eligible)
    fields = {
        "same_id_recovery_rate": "same_id_recovered",
        "new_id_rate": "new_id",
        "wrong_id_rate": "wrong_id",
        "no_output_rate": "no_output",
        "recovery_success_rate": "recovery_success",
    }
    result = {
        key: 100 * sum(r[field] for r in eligible) / n if n else None
        for key, field in fields.items()
    }
    delays = [r["recovery_delay_seconds"] for r in eligible if r["recovery_success"]]
    result.update(
        n_events=n,
        n_events_total=len(rows),
        mean_recovery_delay_s=float(np.mean(delays)) if delays else None,
        median_recovery_delay_s=float(np.median(delays)) if delays else None,
        p95_recovery_delay_s=float(np.percentile(delays, 95)) if delays else None,
    )
    return result


def target_summary(details, people, recovery):
    metric = metric_scoring
    counts = Counter(
        metric.state(r)
        for r in details
        if r.get("visibility") is not None and float(r["visibility"]) >= 0.3
    )
    n = sum(counts.values())
    return {
        "initialization_success_rate": metric.pct(
            sum(p["initialization_success"] for p in people), len(people)
        ),
        "coverage_C": metric.pct(counts["C"], n),
        "wrong_identity_W": metric.pct(counts["W"], n),
        "unmatched_U": metric.pct(counts["U"], n),
        "absent_false_output_rate": None,
        "absent_false_output_reason": "No confirmed absence denominator; missing GT is unknown",
        "recovery_R": recovery["recovery_success_rate"],
        "recovery_time_s": recovery["mean_recovery_delay_s"],
        "n_targets": len(people),
        "n_person_frames": n,
        "counts": dict(counts),
    }
