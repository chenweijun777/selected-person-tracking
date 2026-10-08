"""Unadopted experiments remain optional and independent of earlier versions."""

import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from a1_tracker.target.a2 import A2Config, A2Target
from a1_tracker.types import Track

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.temporal_evidence import A2TemporalConfig, TemporalEvidenceTarget
from research.accumulated_review import A2AccumulatedConfig, AccumulatedReviewTarget
from research.joint_review import A2JointReviewConfig, JointReviewTarget


class Encoder:
    def encode(self, image, tracks):
        return [np.array([1.0, 0.0], np.float32) for _ in tracks]


@pytest.mark.parametrize(
    "config_type,policy_type",
    [
        (A2TemporalConfig, TemporalEvidenceTarget),
        (A2AccumulatedConfig, AccumulatedReviewTarget),
        (A2JointReviewConfig, JointReviewTarget),
    ],
)
def test_optional_variant_defaults_preserve_production(config_type, policy_type):
    raw = json.loads((ROOT / "configs/finetuned/target.json").read_text(encoding="utf-8"))
    encoder = Encoder()
    current = A2Target(encoder, A2Config(**raw))
    variant = policy_type(encoder, config_type(**raw))
    assert policy_type.__bases__ == (A2Target,)
    selected = Track(1, (0, 0, 40, 80), True, 0.9)
    for policy in [current, variant]:
        assert policy.select_track(selected, None, 0.0)
    for timestamp, tracks in [
        (0.0, [selected]),
        (0.1, [selected]),
        (0.2, []),
        (0.3, [Track(2, selected.bbox_xyxy, True, 0.9)]),
        (0.4, [Track(2, selected.bbox_xyxy, True, 0.9)]),
        (0.5, [Track(2, selected.bbox_xyxy, True, 0.9)]),
    ]:
        values = []
        for policy in [current, variant]:
            value = asdict(
                policy.update(tracks, timestamp, None, motion_warp=np.eye(2, 3), motion_valid=True)
            )
            value.pop("session_target_id")
            values.append(value)
        assert values[0] == values[1]
