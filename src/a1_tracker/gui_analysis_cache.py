"""Selection-independent frame inputs, with exact lazy A2 features on disk."""

from __future__ import annotations

import copy
import io
import json
import sqlite3
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np

from a1_tracker.identity_events import IdentityFrame, IdentityRemap
from a1_tracker.joint_runtime import FrameBoxAppearance, JointTarget
from a1_tracker.types import Detection, Track


def source_token(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


class AnalysisCache:
    def __init__(self, path, source):
        self.path = Path(path)
        self.source = source_token(source)
        self.connection = None
        self.runtime = None
        self.complete = False
        self.count = 0
        self.encoder = None
        self.predict_config = None
        self.provenance = None
        self.original_runtime = None
        self.fps = None

    @property
    def db(self):
        if self.connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # GUI jobs are serialized, but successive jobs use different workers.
            self.connection = sqlite3.connect(self.path, check_same_thread=False)
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS frames (frame INTEGER PRIMARY KEY, data TEXT)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS features (frame INTEGER, box BLOB, value BLOB, PRIMARY KEY(frame, box))"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
            )
        return self.connection

    def write(self, frame, data):
        self.db.execute(
            "INSERT INTO frames VALUES (?,?)", (frame, json.dumps(data, allow_nan=False))
        )
        self.db.commit()
        self.count = frame

    def read(self, frame):
        row = self.db.execute("SELECT data FROM frames WHERE frame=?", (frame,)).fetchone()
        if row is None:
            raise RuntimeError(f"分析缓存缺少第{frame}帧")
        return json.loads(row[0])

    def finish(self, summary):
        self.complete = summary["complete_source"]
        self.predict_config = copy.deepcopy(summary.get("effective_predict_config"))
        self.provenance = copy.deepcopy(summary.get("provenance"))
        self.fps = summary["input_fps"]
        if self.original_runtime is None and self.runtime is not None:
            self.original_runtime = copy.deepcopy(self.runtime.provenance())
        if self.connection is not None:
            data = {
                "complete": self.complete,
                "count": self.count,
                "fps": self.fps,
                "source": self.source,
                "provenance": self.provenance,
                "predict_config": self.predict_config,
                "runtime": self.original_runtime,
            }
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('session',?)", (json.dumps(data),)
            )
            self.connection.commit()

    def restore(self, path):
        """Only complete caches can be reopened; incomplete sessions stay separate."""
        path = Path(path)
        if not path.is_file():
            return False
        original = self.path
        self.close()
        self.path = path
        row = self.db.execute("SELECT value FROM metadata WHERE key='session'").fetchone()
        data = json.loads(row[0]) if row else None
        if not data or not data["complete"] or tuple(data["source"]) != self.source:
            self.close()
            self.path = original
            return False
        self.complete, self.count, self.fps = True, data["count"], data["fps"]
        self.provenance, self.predict_config = data["provenance"], data["predict_config"]
        self.original_runtime = data["runtime"]
        return True

    @property
    def config_hashes(self):
        return self.original_runtime["config_files"]

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class PersistentAppearance:
    """Reuse only the same frame/crop/model input; tracker IDs are not keys."""

    def __init__(self, cache, encoder):
        self.cache, self.encoder = cache, encoder
        self.frame = None
        self.hits = self.misses = 0

    def encode(self, image, tracks):
        result, missing, positions = [], [], []
        for track in tracks:
            key = struct.pack("!4d", *track.bbox_xyxy)
            row = self.cache.db.execute(
                "SELECT value FROM features WHERE frame=? AND box=?", (self.frame, key)
            ).fetchone()
            if row is None:
                positions.append((len(result), key))
                result.append(None)
                missing.append(track)
            else:
                self.hits += 1
                result.append(
                    None if row[0] is None else np.load(io.BytesIO(row[0]), allow_pickle=False)
                )
        if missing:
            values = self.encoder.encode(image, missing)
            for (position, key), value in zip(positions, values, strict=True):
                blob = None
                if value is not None:
                    stream = io.BytesIO()
                    np.save(stream, value, allow_pickle=False)
                    blob = stream.getvalue()
                self.cache.db.execute(
                    "INSERT OR REPLACE INTO features VALUES (?,?,?)", (self.frame, key, blob)
                )
                self.misses += 1
                result[position] = value
            self.cache.db.commit()
        return result


class RecordingRuntime:
    def __init__(self, runtime, cache):
        self.runtime, self.cache = runtime, cache
        self.target = runtime.target
        cache.runtime = runtime
        cache.encoder = PersistentAppearance(cache, self.target.appearance.encoder)
        self.target.appearance.encoder = cache.encoder
        self.tracker = RecordingTracker(runtime.tracker, self)
        self.frame = 0

    def begin_frame(self, image, timestamp):
        self.frame += 1
        self.timestamp = timestamp
        self.cache.encoder.frame = self.frame
        self.runtime.begin_frame(image, timestamp)

    @property
    def motion_warp(self):
        return self.runtime.motion_warp

    @property
    def motion_info(self):
        return self.runtime.motion_info

    def diagnostics(self):
        return self.runtime.diagnostics()

    def provenance(self):
        return {
            **self.runtime.provenance(),
            "analysis_cache": {"path": str(self.cache.path), "reused": False},
        }


class RecordingTracker:
    def __init__(self, tracker, runtime):
        self.tracker, self.runtime = tracker, runtime

    def reset(self):
        self.tracker.reset()

    def update(self, detections, image):
        tracks = self.tracker.update(detections, image)
        runtime = self.runtime
        runtime.cache.write(
            runtime.frame,
            {
                "timestamp": runtime.timestamp,
                "detections": [asdict(d) for d in detections],
                "tracks": [asdict(t) for t in tracks],
                "motion_warp": runtime.motion_warp.tolist(),
                "motion_info": runtime.motion_info,
                "identity": asdict(self.tracker.identity_frame),
                "tracker_events": self.tracker.last_logs,
                "gallery_updates": self.tracker.last_gallery,
            },
        )
        return tracks


class CachedDetector:
    def __init__(self, cache):
        self.cache = cache
        self.effective_predict_config = cache.predict_config

    def detect(self, packet):
        return [
            Detection(tuple(d["bbox_xyxy"]), d["score"], d["class_id"])
            for d in self.cache.read(packet.frame_id)["detections"]
        ]


class CachedRuntime:
    """Replay immutable tracker/motion/handoff inputs; only A2 changes."""

    def __init__(self, cache, config):
        self.cache = cache
        self.frame = 0
        self.row = None
        self.tracker = self
        backend = (
            cache.encoder.encoder
            if cache.encoder is not None
            else LazyAppearance(
                next(path for path in cache.config_hashes if path.endswith("yolo26n-reid.onnx"))
            )
        )
        self.encoder = PersistentAppearance(cache, backend)
        self.target = JointTarget(
            config,
            FrameBoxAppearance(None, "448_RGB", encoder=self.encoder),
            identity_source=lambda: self.identity,
        )

    def reset(self):
        self.frame = 0

    def begin_frame(self, image, timestamp):
        self.frame += 1
        self.row = self.cache.read(self.frame)
        if timestamp != self.row["timestamp"]:
            raise RuntimeError("缓存视频时间不匹配")
        batch = self.row["identity"]
        self.identity = IdentityFrame(
            batch["generation"],
            batch["frame_id"],
            tuple(
                IdentityRemap(r["frame_id"], r["young_id"], r["old_id"], tuple(r["bbox_xyxy"]))
                for r in batch["remaps"]
            ),
        )
        self.motion_warp = np.asarray(self.row["motion_warp"])
        self.motion_info = self.row["motion_info"]
        self.encoder.frame = self.frame
        self.target.new_frame(
            timestamp,
            self.motion_warp,
            self.motion_info["valid"],
            frame_id=self.identity.frame_id,
            tracker_generation=self.identity.generation,
        )

    def update(self, detections, image):
        return [
            Track(t["tracker_id"], tuple(t["bbox_xyxy"]), t["observed"], t["score"], t["age"])
            for t in self.row["tracks"]
        ]

    def diagnostics(self):
        return {
            "target": self.target.diagnostics(),
            "camera_motion": {**self.motion_info, "warp": self.motion_warp.tolist()},
            "tracker_events": self.row["tracker_events"],
            "gallery_updates": self.row["gallery_updates"],
        }

    def provenance(self):
        original = self.cache.original_runtime
        return {
            **original,
            "appearance": {**original["appearance"], "target": self.target.appearance.stats()},
            "analysis_cache": {
                "path": str(self.cache.path),
                "reused": True,
                "detection_calls": 0,
                "tracker_calls": 0,
                "gmc_calls": 0,
                "appearance_hits": self.encoder.hits,
                "appearance_misses": self.encoder.misses,
            },
        }


class LazyAppearance:
    """Reopened caches need no ONNX model until a new crop is requested."""

    def __init__(self, model):
        self.model = model
        self.encoder = None

    def encode(self, image, tracks):
        if self.encoder is None:
            from a1_tracker.reid.ultralytics_onnx import UltralyticsOnnxAppearance

            self.encoder = UltralyticsOnnxAppearance(self.model, mode="448_RGB")
        return self.encoder.encode(image, tracks)
