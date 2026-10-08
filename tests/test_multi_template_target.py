import numpy as np

from a1_tracker.target.a2 import A2Config, A2Target
from a1_tracker.types import Track


class Encoder:
    feature = np.array([1.0, 0.0])

    def encode(self, image, tracks):
        return [self.feature.copy() for _ in tracks]


def track(identifier=1):
    return Track(identifier, (0, 0, 60, 100), True, 0.9)


def update(target, tracks, timestamp):
    return target.update(tracks, timestamp, None, motion_warp=np.eye(2, 3), motion_valid=True)


def test_bank_recognizes_new_view_but_keeps_confirmation_and_bound_verification():
    encoder = Encoder()
    target = A2Target(encoder, A2Config(min_tracking_similarity=0.2, min_recovery_similarity=0.7))
    assert target.select_track(track(), None, 0.0)
    encoder.feature = np.array([0.6, 0.8])
    for t in (0.0, 0.1, 0.2):
        result = update(target, [track()], t)
        assert result.reason == "bound_verified"
        assert abs(target.last_debug["bound_similarity"] - 0.6) < 1e-6
    assert len(target.bank.templates) == 1
    assert target.memory_updates == 0
    assert not update(target, [], 0.3).valid
    assert not update(target, [track(2)], 0.4).valid
    assert target.last_debug["best_similarity"] > 0.99
    assert not update(target, [track(2)], 0.5).valid
    assert update(target, [track(2)], 0.61).reason == "candidate_confirmed"
    assert len(target.bank.templates) == 1  # Unconfirmed/just-confirmed crops never collected.
    update(target, [track(2)], 0.7)
    assert abs(target.last_debug["bound_similarity"] - 0.6) < 1e-6
    assert not target._bank_scoring
    assert target.select_track(track(3), None, 0.8)
    assert not target.bank.templates  # Reusing a manager must not retain another person's views.
    target.bank.templates.append((np.array([1.0, 0.0]), 1, 3))
    target.reset()
    assert not target.bank.templates


def test_disabled_bank_does_not_change_recovery_score():
    encoder = Encoder()
    target = A2Target(
        encoder,
        A2Config(min_tracking_similarity=0.2, min_recovery_similarity=0.7, bank_enabled=False),
    )
    assert target.select_track(track(), None, 0.0)
    encoder.feature = np.array([0.6, 0.8])
    for t in (0.0, 0.1, 0.2):
        update(target, [track()], t)
    assert not target.bank.templates
    update(target, [], 0.3)
    result = update(target, [track(2)], 0.4)
    assert result.reason == "appearance_below_threshold"
    assert not target._bank_scoring
