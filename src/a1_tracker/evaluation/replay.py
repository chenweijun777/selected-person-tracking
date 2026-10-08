"""Causal policy replay over fixed observed tracks and image-only descriptors."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from a1_tracker.identity_events import IdentityRemap
from a1_tracker.target.a2 import A2Config, A2Target
from a1_tracker.target.baselines import TargetBaseline
from a1_tracker.types import Track


class FixedAppearance:
    """Pure descriptors; a selected target's memory is never cached here."""

    def __init__(self):
        self.by_id = {}

    def encode(self, image, tracks):
        result = []
        for track in tracks:
            observed, feature = self.by_id[track.tracker_id]
            if track.bbox_xyxy != observed.bbox_xyxy:
                raise ValueError("Descriptor does not belong to this observed box")
            result.append(feature)
        return result


class ReplaySequence:
    def __init__(self, folder):
        self.folder = Path(folder)
        data = np.loadtxt(self.folder / "tracks.txt", delimiter=",", ndmin=2)
        vectors = np.load(self.folder / "features.npy", mmap_mode="r")
        present = np.load(self.folder / "valid.npy", mmap_mode="r")
        if len(data) != len(vectors) or len(data) != len(present):
            raise ValueError("Track and descriptor row counts differ")
        self.frames, self.features = defaultdict(list), defaultdict(dict)
        for index, row in enumerate(data):
            frame, tid = int(row[0]), int(row[1])
            track = Track(tid, (row[2], row[3], row[2] + row[4], row[3] + row[5]), True, row[6])
            if tid in self.features[frame]:
                raise ValueError("Duplicate observed tracker ID")
            self.frames[frame].append(track)
            self.features[frame][tid] = track, vectors[index] if present[index] else None
        self.motion = {}
        with (self.folder / "motion.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                warp = np.array(
                    [float(row[k]) for k in ("h00", "h01", "h02", "h10", "h11", "h12")]
                ).reshape(2, 3)
                self.motion[int(row["frame"])] = warp, row["valid"] == "True"
        self.remaps = defaultdict(list)
        for line in (self.folder / "remaps.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            survivor = next(
                t for t in self.frames[event["frame"]] if t.tracker_id == event["old_id"]
            )
            self.remaps[event["frame"]].append(
                IdentityRemap(
                    event["frame"], event["young_id"], event["old_id"], survivor.bbox_xyxy
                )
            )
        self.encoder = FixedAppearance()

    def observations(self, frame):
        self.encoder.by_id = self.features.get(frame, {})
        return self.frames.get(frame, [])


def replay(sequence, cases, config, modes, output, *, observer=None):
    """GT IDs label offline sessions only; update receives observations and time."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    changes = {
        "A2": {},
        "NoBank": {"bank_enabled": False},
        "NoEMA": {"memory_alpha": 0.0},
        "NoHandoff": {},
        "NoMotion": {"motion_enabled": False},
    }
    policies = {}
    # The fixed encoder supplies this frame's descriptors. A1 still requires
    # a non-None frame token, while no pixels are read during cached replay.
    frame_token = object()
    for mode in modes:
        policies[mode] = {}
        for case in cases:
            if mode in ("A0", "A1"):
                policy = TargetBaseline(mode, sequence.encoder if mode == "A1" else None)
            else:
                policy = A2Target(sequence.encoder, A2Config(**(config | changes[mode])))
            policies[mode][case["gt_id"]] = policy
    streams, writers = {}, {}
    try:
        for mode in modes:
            streams[mode] = (output / f"{mode}.csv").open("x", encoding="utf-8", newline="")
            writers[mode] = csv.DictWriter(
                streams[mode], fieldnames=["sequence", "gt_id", "frame", "tracker_id", "valid"]
            )
            writers[mode].writeheader()
        first = min(c["selection_frame"] for c in cases)
        last = max(c["end_frame"] for c in cases)
        for frame in range(first, last + 1):
            tracks = sequence.observations(frame)
            by_id = {t.tracker_id: t for t in tracks}
            warp, valid = sequence.motion[frame]
            for case in cases:
                if not case["selection_frame"] <= frame <= case["end_frame"]:
                    continue
                timestamp = (frame - 1) / case["fps"]
                for mode in modes:
                    policy = policies[mode][case["gt_id"]]
                    if frame == case["selection_frame"] and case["initial_tracker_id"] is not None:
                        selected = by_id[case["initial_tracker_id"]]
                        if mode in ("A0", "A1"):
                            policy.select_track(selected, frame_token)
                        else:
                            policy.select_track(selected, None, timestamp)
                    if mode in ("A0", "A1"):
                        result = policy.update(tracks, timestamp, frame_token)
                    else:
                        result = policy.update(
                            tracks,
                            timestamp,
                            None,
                            motion_warp=warp,
                            motion_valid=valid,
                            identity_remaps=()
                            if mode == "NoHandoff"
                            else sequence.remaps.get(frame, ()),
                            identity_frame_id=frame,
                        )
                    writers[mode].writerow(
                        dict(
                            sequence=case["sequence"],
                            gt_id=case["gt_id"],
                            frame=frame,
                            tracker_id=result.tracker_id,
                            valid=result.valid,
                        )
                    )
                    if observer is not None:
                        observer(
                            case,
                            frame,
                            mode,
                            policy,
                            result,
                            tracks,
                            timestamp,
                            warp,
                            valid,
                            sequence,
                        )
    finally:
        for stream in streams.values():
            stream.close()
    (output / "config.json").write_text(
        json.dumps(
            {
                "effective_config": config,
                "modes": modes,
                "shared_observed_tracks": True,
                "appearance": "448_RGB",
                "GT_online": False,
                "cached_replay_time_is_online_FPS": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
