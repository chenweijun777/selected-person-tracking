"""ShortLink renames retain identity evidence without bypassing A2 confirmation."""

import json
from pathlib import Path

import numpy as np
import pytest

from a1_tracker.identity_events import IdentityFrame, IdentityRemap
from a1_tracker.joint_runtime import FrameBoxAppearance, JointRuntime, JointTarget
from a1_tracker.target.a2 import A2Config, A2Target
from a1_tracker.types import Detection, TargetState, Track

ROOT = Path(__file__).resolve().parents[1]
BOX = (10.0, 10.0, 40.0, 90.0)


class Encoder:
    similarity = 1.0

    def encode(self, image, tracks):
        feature = np.zeros(512, np.float32)
        feature[:2] = self.similarity, np.sqrt(1 - self.similarity**2)
        return [feature.copy() for _ in tracks]


def subject(identifier=282):
    raw = json.loads((ROOT / "configs/pretrained/target.json").read_text())
    raw.pop("status", None)
    encoder = Encoder()
    policy = A2Target(encoder, A2Config(**raw))
    track = Track(identifier, BOX, True, 0.9)
    assert policy.select_track(track, None, 0.0)
    for stamp in (0.0, 1 / 30, 2 / 30):
        policy.update([track], stamp, None, motion_warp=np.eye(2, 3), motion_valid=True)
    return policy, encoder


def step(policy, identifier, stamp, events=(), frame=4, box=BOX, valid=True):
    return policy.update(
        [Track(identifier, box, True, 0.9)],
        stamp,
        None,
        motion_warp=np.eye(2, 3),
        motion_valid=valid,
        identity_remaps=events,
        identity_frame_id=frame,
    )


def test_confirmed_identity_is_renamed_without_new_recovery_or_memory_reset():
    policy, _ = subject()
    session, anchor, memory, bank = (
        policy.session_target_id,
        policy.anchor,
        policy.memory,
        policy.bank,
    )
    updates = policy.memory_updates
    bank.templates = [(anchor.copy(), 1, 282)]
    result = step(policy, 238, 0.1, (IdentityRemap(4, 282, 238, BOX),))
    assert result.valid and result.tracker_id == 238 and result.reason == "bound_verified"
    assert policy.recoveries == 0 and policy.session_target_id == session
    assert policy.anchor is anchor and policy.bank is bank
    assert np.array_equal(policy.memory, memory) and policy.memory_updates == updates + 1
    assert bank.templates[0][1:] == (1, 238)
    assert len(policy.history) == 4
    assert policy.last_debug["identity_handoffs"][0]["accepted"]


def test_pending_candidate_keeps_elapsed_time_and_cannot_take_same_id_shortcut():
    policy, _ = subject(238)
    policy.update([], 0.1, None, motion_warp=np.eye(2, 3), motion_valid=True)
    result = step(policy, 282, 0.2, frame=7)
    assert not result.valid and policy.candidate_observations == 1
    first = policy.candidate_first_timestamp
    result = step(policy, 238, 0.25, (IdentityRemap(8, 282, 238, BOX),), frame=8)
    assert not result.valid and result.reason == "candidate_confirming"
    assert policy.candidate_first_timestamp == first and policy.candidate_observations == 2
    result = step(policy, 238, 0.30, frame=9)
    assert not result.valid  # 3 observations still do not satisfy .15 seconds
    result = step(policy, 238, 0.36, frame=10)
    assert result.valid and result.reason == "candidate_confirmed" and policy.recoveries == 1


@pytest.mark.parametrize(
    "case", ["stale", "conflict", "wrong_box", "unrelated", "weak", "broken_gmc"]
)
def test_unqualified_handoff_cannot_transfer_verified_binding(case):
    policy, encoder = subject()
    events = [
        IdentityRemap(3 if case == "stale" else 4, 999 if case == "unrelated" else 282, 238, BOX)
    ]
    if case == "conflict":
        events.append(IdentityRemap(4, 282, 239, BOX))
    if case == "weak":
        encoder.similarity = 0.05
    result = step(
        policy,
        238,
        0.1,
        events,
        box=(100.0, 10.0, 130.0, 90.0) if case == "wrong_box" else BOX,
        valid=case != "broken_gmc",
    )
    assert policy.tracker_id == 282
    assert not any(r["accepted"] for r in policy.last_debug["identity_handoffs"])
    assert not result.valid


def test_adapter_checks_frame_generation_and_only_updates_once():
    encoder = Encoder()
    batch = IdentityFrame(1, 1)
    target = JointTarget(
        ROOT / "configs/pretrained/target.json",
        FrameBoxAppearance(None, "448_RGB", encoder=encoder),
        identity_source=lambda: batch,
    )
    target.new_frame(0.0, np.eye(2, 3), True, frame_id=1, tracker_generation=1)
    assert target.select_track(Track(282, BOX, True, 0.9), None)
    first = target.update([Track(282, BOX, True, 0.9)], 0.0, None)
    assert target.update([], 0.0, None) is first
    batch = IdentityFrame(0, 2, (IdentityRemap(2, 282, 238, BOX),))
    target.new_frame(0.1, np.eye(2, 3), True, frame_id=2, tracker_generation=1)
    with pytest.raises(RuntimeError, match="frame/generation"):
        target.update([Track(238, BOX, True, 0.9)], 0.1, None)
    assert target.tracker_id == 282
    batch = IdentityFrame(1, 2, (IdentityRemap(2, 282, 238, BOX),))
    result = target.update([Track(238, BOX, True, 0.9)], 0.1, None)
    count = target.policy._bank_frame
    assert result.valid and target.tracker_id == 238
    assert target.update([], 0.1, None) is result and target.policy._bank_frame == count
    target.reset()
    assert target.update([Track(238, BOX, True, 0.9)], 0.1, None).state == TargetState.UNSELECTED


def test_runtime_delivers_actual_stitch_before_target_update(tmp_path, monkeypatch):
    import a1_tracker.reid.ultralytics_onnx as reid

    class ConfiguredEncoder(Encoder):
        def __init__(self, model, *, mode):
            pass

    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(reid, "UltralyticsOnnxAppearance", ConfiguredEncoder)
    runtime = JointRuntime(
        30,
        ROOT / "configs/pretrained/tracker.yaml",
        ROOT / "configs/pretrained/target.json",
        ROOT / "models/yolo26n-reid.onnx",
    )
    runtime.motion.update = lambda image: (np.eye(2, 3), {"valid": True})
    image = np.zeros((100, 100, 3), np.uint8)
    first_id = selected_id = session = None
    for frame in range(1, 20):
        runtime.begin_frame(image, (frame - 1) / 30)
        detections = (
            [Detection(BOX, 0.9)]
            if frame <= 4
            else [Detection((32.0, 10.0, 62.0, 90.0), 0.9)]
            if frame >= 9
            else []
        )
        tracks = runtime.tracker.update(detections, image)
        if frame == 1:
            first_id = tracks[0].tracker_id
        if frame == 10:
            assert len(tracks) == 1 and tracks[0].tracker_id != first_id
            selected_id = tracks[0].tracker_id
            assert runtime.target.select_track(tracks[0], image)
            session = runtime.target.session_target_id
        result = runtime.target.update(tracks, (frame - 1) / 30, image)
    batch = runtime.tracker.identity_frame
    assert len(batch.remaps) == 1 and batch.remaps[0].young_id == selected_id
    assert batch.remaps[0].old_id == first_id
    assert result.valid and result.reason == "bound_verified" and result.tracker_id == first_id
    assert runtime.target.session_target_id == session and runtime.target.policy.recoveries == 0
