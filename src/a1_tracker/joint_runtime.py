"""Online adapter for the selected 26s / ShortLink2 / A2_v7 experiment.

No dataset, GT, frozen tracks or experiment directories are read at runtime.
Tracker and target share the 448 RGB preprocessing contract.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter

import numpy as np

from a1_tracker.geometry import contains_point
from a1_tracker.identity_events import IdentityFrame
from a1_tracker.motion import SparseCameraMotion
from a1_tracker.target.a2 import A2Config, A2Target
from a1_tracker.types import Track


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrameBoxAppearance:
    """Lazy frame-local crop cache, separate for 224_BGR and 448_RGB."""

    def __init__(self, model, mode, *, encoder=None):
        if mode not in ("224_BGR", "448_RGB"):
            raise ValueError(f"Unsupported appearance mode: {mode}")
        if encoder is None:
            from a1_tracker.reid.ultralytics_onnx import UltralyticsOnnxAppearance

            encoder = UltralyticsOnnxAppearance(model, mode=mode)
        self.encoder = encoder
        self.mode = mode
        self.cache = {}
        self.calls = self.boxes = 0
        self.elapsed_ms = 0.0

    def new_frame(self):
        self.cache.clear()

    def encode(self, image, tracks):
        # Crop geometry, rather than tracker ID, is the frozen experiment's key.
        missing = {tuple(t.bbox_xyxy): t for t in tracks if tuple(t.bbox_xyxy) not in self.cache}
        if missing:
            start = perf_counter()
            features = self.encoder.encode(image, list(missing.values()))
            self.elapsed_ms += (perf_counter() - start) * 1000
            for box, feature in zip(missing, features, strict=True):
                if feature is not None:
                    feature = np.asarray(feature)
                    if feature.shape != (512,) or not np.isfinite(feature).all():
                        raise RuntimeError(
                            "Appearance model must return finite 512-dimensional features"
                        )
                self.cache[box] = feature
            self.calls += 1
            self.boxes += len(missing)
        return [self.cache[tuple(t.bbox_xyxy)] for t in tracks]

    def stats(self):
        return {
            "mode": self.mode,
            "calls": self.calls,
            "encoded_boxes": self.boxes,
            "elapsed_ms": self.elapsed_ms,
        }


class JointTracker:
    """BoT-SORT with five-second Lost retention and ShortLink hooks."""

    def __init__(self, config, model, fps, *, appearance=None):
        from a1_tracker.trackers.identity.short_identity import ShortIdentity
        from a1_tracker.trackers.ultralytics_mot import UltralyticsMOTTracker

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Joint Tracker requires the input video's actual positive FPS")
        self.fps = fps
        self.buffer_frames = math.ceil(5 * fps)
        self.base = UltralyticsMOTTracker(config)
        if self.base.settings["tracker_type"] != "botsort":
            raise ValueError("Joint requires BoT-SORT")
        if self.base.settings.get("gmc_method") != "sparseOptFlow":
            raise ValueError("Joint requires sparseOptFlow GMC")
        # The frozen adapter sets buffer frames directly. Do not scale by FPS twice.
        self.base.settings["track_buffer"] = self.buffer_frames
        self.base.tracker.args.track_buffer = self.buffer_frames
        self.base.tracker.max_frames_lost = self.buffer_frames
        self.appearance = appearance or FrameBoxAppearance(model, "448_RGB")
        self.extension_class = ShortIdentity
        self.extension = None
        self.reset()

    def reset(self):
        self.identity_generation = getattr(self, "identity_generation", -1) + 1
        self.identity_frame = IdentityFrame(self.identity_generation, 0)
        if self.extension is not None:
            # Unhook before reattaching; repeated reset must not nest hooks.
            self.base.tracker._first_association = self.extension.original_first
            self.base.tracker._init_new_tracks = self.extension.original_init
            self.base.tracker._unconfirmed_association = self.extension.original_unconfirmed
        self.base.reset()
        self.base.tracker.max_frames_lost = self.buffer_frames
        self.appearance.new_frame()
        self.extension = self.extension_class(
            self.base.tracker,
            {"mode": "448_RGB", "seed_low": False},
            self.provide,
            self.fps,
            confirm=self.base.settings.get("shortlink_confirm", 1),
            short_cosine_min=self.base.settings.get("shortlink_cosine_min", 0.50),
            margin_min=self.base.settings.get("shortlink_margin_min", 0.03),
            max_young_age_seconds=self.base.settings.get("shortlink_max_young_age_seconds", 0.5),
        )
        self.last_logs = []
        self.last_gallery = []
        self.stage_counts = Counter()
        self.detections = []
        self.image = None

    def provide(self, indices):
        tracks = [
            Track(i + 1, self.detections[i].bbox_xyxy, True, self.detections[i].score)
            for i in indices
        ]
        features = self.appearance.encode(self.image, tracks)
        return {i: v for i, v in zip(indices, features, strict=True) if v is not None}

    def update(self, detections, image):
        self.detections, self.image = detections, image
        self.appearance.new_frame()
        self.extension.frame_input(
            {}, [{"box": list(d.bbox_xyxy), "score": d.score} for d in detections]
        )
        tracks = self.base.update(detections, image)
        self.extension.update_gallery()
        if len({t.tracker_id for t in tracks}) != len(tracks):
            raise RuntimeError("Identity extension emitted duplicate tracker IDs")
        self.last_logs = list(self.extension.logs)
        self.last_gallery = list(self.extension.gallery_records)
        self.extension.gallery_records.clear()
        self.stage_counts.update(r["stage"] for r in self.last_logs)
        self.identity_frame = IdentityFrame(
            self.identity_generation,
            self.base.tracker.frame_id,
            tuple(self.extension.identity_remaps),
        )
        # A2 was evaluated on saved MOT rows, with x/y/w/h at 0.001 px and
        # scores at 1e-6. Use that same boundary before selecting/cropping.
        canonical = []
        for track in tracks:
            x, y, x2, y2 = track.bbox_xyxy
            left, top = round(x, 3), round(y, 3)
            width, height = round(x2 - x, 3), round(y2 - y, 3)
            if width > 0 and height > 0:
                canonical.append(
                    Track(
                        track.tracker_id,
                        (left, top, left + width, top + height),
                        track.observed,
                        round(track.score, 6),
                        track.age,
                    )
                )
        return canonical


class JointTarget:
    """Click adapter with exactly one A2 observation per input frame/session."""

    strategy = "A2 + ShortLink"

    def __init__(self, config, appearance, *, identity_source=None):
        values = json.loads(Path(config).read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(A2Config)}
        unknown = set(values) - allowed - {"status"}
        if unknown:
            raise ValueError(f"Unknown A2 configuration fields: {sorted(unknown)}")
        self.config = A2Config(**{f.name: values.get(f.name, f.default) for f in fields(A2Config)})
        self.appearance = appearance
        self.identity_source = identity_source
        self.identity_frame_id = self.identity_generation = None
        self.timestamp = None
        self.warp = None
        self.motion_valid = False
        self.reset()

    @property
    def session_target_id(self):
        return self.policy.session_target_id

    @property
    def tracker_id(self):
        return self.policy.tracker_id

    def reset(self):
        self.policy = A2Target(self.appearance, self.config)
        self.cached_result = None
        self.selection_failure_reason = None

    def new_frame(
        self, timestamp, warp=None, motion_valid=False, *, frame_id=None, tracker_generation=None
    ):
        if not math.isfinite(timestamp) or (
            self.timestamp is not None and timestamp <= self.timestamp
        ):
            raise ValueError("Input frame timestamps must be finite and strictly increasing")
        self.timestamp, self.warp, self.motion_valid = timestamp, warp, motion_valid
        self.identity_frame_id, self.identity_generation = frame_id, tracker_generation
        self.appearance.new_frame()
        self.cached_result = None

    def select_by_point(self, tracks, point, image):
        candidates = [t for t in tracks if t.observed and contains_point(t.bbox_xyxy, point)]
        if len(candidates) != 1:
            return False
        return self.select_track(candidates[0], image)

    def select_track(self, track, image):
        if self.timestamp is None:
            raise RuntimeError("Selection requires a current input frame")
        candidate = A2Target(self.appearance, self.config)
        if not candidate.select_track(track, image, self.timestamp):
            self.selection_failure_reason = "selection_crop_ineligible_or_no_appearance"
            return False
        self.policy = candidate
        self.cached_result = None
        self.selection_failure_reason = None
        return True

    def update(self, tracks, timestamp, image=None):
        if timestamp != self.timestamp:
            raise ValueError("Target update must use the current video's timestamp")
        if self.cached_result is None:
            remaps = ()
            if self.identity_source is not None:
                batch = self.identity_source()
                if (
                    self.identity_frame_id is None
                    or self.identity_generation is None
                    or (batch.frame_id, batch.generation)
                    != (self.identity_frame_id, self.identity_generation)
                ):
                    raise RuntimeError(
                        "Tracker identity events must belong to this input frame/generation"
                    )
                remaps = batch.remaps
            self.cached_result = self.policy.update(
                tracks,
                timestamp,
                image,
                motion_warp=self.warp,
                motion_valid=self.motion_valid,
                identity_remaps=remaps,
                identity_frame_id=self.identity_frame_id,
            )
        return self.cached_result

    def diagnostics(self):
        reliable = self.policy.last_reliable_timestamp
        return dict(
            **self.policy.last_debug,
            recoveries=self.policy.recoveries,
            last_reliable_timestamp=reliable,
            search_age_seconds=None if reliable is None else self.timestamp - reliable,
        )


class JointRuntime:
    def __init__(self, fps, tracker_config, target_config, appearance_model):
        self.tracker_config = Path(tracker_config).resolve()
        self.target_config = Path(target_config).resolve()
        self.appearance_model = Path(appearance_model).resolve()
        self.tracker = JointTracker(self.tracker_config, self.appearance_model, fps)
        self.target = JointTarget(
            self.target_config,
            FrameBoxAppearance(self.appearance_model, "448_RGB"),
            identity_source=lambda: self.tracker.identity_frame,
        )
        self.motion = SparseCameraMotion()
        self.motion_info = {"valid": False, "reason": "first_frame"}
        self.motion_warp = np.eye(2, 3)
        package = Path(__file__).parent
        sources = [
            Path(__file__),
            package / "video_app.py",
            package / "types.py",
            package / "identity_events.py",
            package / "geometry.py",
            package / "detectors/ultralytics.py",
            package / "motion.py",
            package / "trackers/ultralytics_mot.py",
            package / "reid/ultralytics_onnx.py",
            *sorted((package / "trackers/identity").glob("*.py")),
            *sorted((package / "target").rglob("*.py")),
        ]
        backend = Path(inspect.getfile(type(self.tracker.base.tracker))).parent
        sources.extend(
            backend / name
            for name in (
                "bot_sort.py",
                "byte_tracker.py",
                "basetrack.py",
                "utils/matching.py",
                "utils/reid.py",
                "utils/gmc.py",
                "utils/stracks.py",
            )
        )
        self.source_hashes = {str(p): sha256(p) for p in sources if p.is_file()}
        self.config_hashes = {
            str(p): sha256(p)
            for p in (self.tracker_config, self.target_config, self.appearance_model)
        }
        self.versions = {
            name: importlib.metadata.version(name)
            for name in ("ultralytics", "onnxruntime", "torch", "opencv-python", "numpy")
        }

    def begin_frame(self, image, timestamp):
        self.motion_warp, self.motion_info = self.motion.update(image)
        self.target.new_frame(
            timestamp,
            self.motion_warp,
            self.motion_info["valid"],
            frame_id=self.tracker.base.tracker.frame_id + 1,
            tracker_generation=self.tracker.identity_generation,
        )

    def diagnostics(self):
        return {
            "target": self.target.diagnostics(),
            "camera_motion": dict(**self.motion_info, warp=self.motion_warp.tolist()),
            "tracker_events": self.tracker.last_logs,
            "gallery_updates": self.tracker.last_gallery,
        }

    def provenance(self):
        return {
            "strategy": self.target.strategy,
            "fps": self.tracker.fps,
            "tracker": {
                "extension": "ShortIdentity",
                "appearance": "448_RGB",
                "residual_matching": False,
                "identity_handoff": "Frame-local ShortLink remaps consumed before A2; generation isolated",
                "confirm_observations": self.tracker.extension.confirm,
                "shortlink_cosine_min": self.tracker.extension.short_cosine_min,
                "shortlink_margin_min": self.tracker.extension.margin_min,
                "shortlink_max_young_age_seconds": self.tracker.extension.max_young_age_seconds,
                "retention_seconds": 5.0,
                "buffer_frames": self.tracker.buffer_frames,
                "effective_config": self.tracker.base.settings,
                "stage_counts": dict(self.tracker.stage_counts),
            },
            "target": {
                "algorithm": "A2_v7",
                "appearance": "448_RGB",
                "effective_config": asdict(self.target.config),
            },
            "config_files": self.config_hashes,
            "source_sha256": self.source_hashes,
            "dependency_versions": self.versions,
            "appearance": {
                "tracker": self.tracker.appearance.stats(),
                "target": self.target.appearance.stats(),
            },
            "gmc": "Causal SparseCameraMotion quality gate for A2; native sparseOptFlow for BoT-SORT",
        }
