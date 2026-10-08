"""Feed saved person detections to Ultralytics' stateful MOT trackers."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import yaml

from a1_tracker.types import Detection, Track


class _Boxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros(len(conf), dtype=np.float32)
        self.xywh = xyxy.copy()
        self.xywh[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        self.xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, key):
        return _Boxes(self.xyxy[key], self.conf[key])


class UltralyticsMOTTracker:
    def __init__(self, config: str | Path):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.deep_oc_sort import DeepOCSORT
        from ultralytics.trackers.fast_tracker import FASTTracker
        from ultralytics.trackers.oc_sort import OCSORT
        from ultralytics.trackers.track_tracker import TRACKTRACK

        self.config_path = Path(config)
        settings = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.settings = settings
        tracker_types = {
            "bytetrack": BYTETracker,
            "botsort": BOTSORT,
            "tracktrack": TRACKTRACK,
            "ocsort": OCSORT,
            "deepocsort": DeepOCSORT,
            "fasttrack": FASTTracker,
        }
        self.tracker = tracker_types[settings["tracker_type"]](SimpleNamespace(**settings))
        self.encoder_calls = 0
        self.encoded_boxes = 0
        self.encoder_ms = 0.0
        self.appearance_association_calls = 0
        if settings.get("with_reid"):
            if self.tracker.encoder is None or settings.get("model") == "auto":
                raise RuntimeError("ReID requires a loaded external appearance encoder")
            encoder = self.tracker.encoder
            if settings.get("reid_color") == "RGB":
                from a1_tracker.reid.ultralytics_onnx import configure_rgb_reid

                configure_rgb_reid(encoder, settings.get("reid_imgsz", 448))

            def counted_encoder(image, boxes):
                start = perf_counter()
                features = encoder(image, boxes)
                self.encoder_ms += (perf_counter() - start) * 1000
                self.encoder_calls += 1
                self.encoded_boxes += sum(feature is not None for feature in features)
                return features

            self.tracker.encoder = counted_encoder
            # Count actual association passes that receive appearance-bearing detections.
            method_name = (
                "_cost_matrix"
                if settings["tracker_type"] == "tracktrack"
                else (
                    "_fuse_appearance" if settings["tracker_type"] == "deepocsort" else "get_dists"
                )
            )
            original = getattr(self.tracker, method_name)

            def counted_association(*args, **kwargs):
                tracks, detections = (
                    (args[1], args[2]) if method_name == "_fuse_appearance" else (args[0], args[1])
                )
                if (
                    tracks
                    and detections
                    and any(getattr(d, "curr_feat", None) is not None for d in detections)
                ):
                    self.appearance_association_calls += 1
                return original(*args, **kwargs)

            setattr(self.tracker, method_name, counted_association)

    def reset(self) -> None:
        self.tracker.reset()
        self.encoder_calls = self.encoded_boxes = 0
        self.encoder_ms = 0.0
        self.appearance_association_calls = 0

    def update(
        self,
        detections: list[Detection],
        image: np.ndarray | None = None,
        dets_del: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> list[Track]:
        if image is None:
            raise ValueError("original image is required for the shared MOT adapter")
        tracks, _ = self.update_with_stats(detections, image, dets_del=dets_del)
        return tracks

    def update_with_stats(
        self,
        detections: list[Detection],
        image: np.ndarray,
        dets_del: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[list[Track], dict]:
        from ultralytics.trackers.basetrack import BaseTrack

        boxes = np.asarray([d.bbox_xyxy for d in detections], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray([d.score for d in detections], dtype=np.float32)
        args = self.tracker.args
        high = int(np.count_nonzero(scores >= args.track_high_thresh))
        low = int(
            np.count_nonzero((scores > args.track_low_thresh) & (scores < args.track_high_thresh))
        )
        before = BaseTrack._count
        if dets_del is not None and self.settings["tracker_type"] != "tracktrack":
            raise ValueError("dets_del is supported only by TrackTrack")
        result = (
            self.tracker.update(_Boxes(boxes, scores), image, dets_del=dets_del)
            if dets_del is not None
            else (self.tracker.update(_Boxes(boxes, scores), image))
        )
        created = BaseTrack._count - before
        tracks = [
            Track(int(row[4]), tuple(float(v) for v in row[:4]), True, float(row[5]))
            for row in result
            if row[2] > row[0] and row[3] > row[1]
        ]
        return tracks, {
            "high": high,
            "low": low,
            "ignored": len(scores) - high - low,
            "new_tracks": created,
            "output_tracks": len(tracks),
            "recovered_candidates": 0
            if dets_del is None
            else int(np.count_nonzero(dets_del[1] > args.track_high_thresh)),
            "encoder_calls": self.encoder_calls,
            "encoded_boxes": self.encoded_boxes,
            "encoder_ms": round(self.encoder_ms, 4),
            "appearance_association_calls": self.appearance_association_calls,
        }
