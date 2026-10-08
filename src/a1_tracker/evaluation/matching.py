"""Greedy one-to-one IoU matching at the frozen evaluation boundary."""

from dataclasses import dataclass

from a1_tracker.geometry import iou
from a1_tracker.types import BBox, Detection


@dataclass(slots=True, frozen=True)
class GroundTruth:
    track_id: int
    bbox_xyxy: BBox
    visibility: float


@dataclass(slots=True)
class MatchResult:
    matches: list[tuple[int, int, float]]
    false_positive_indices: list[int]
    false_negative_indices: list[int]
    ignored_prediction_indices: list[int]


def match_detections(
    detections: list[Detection],
    ground_truth: list[GroundTruth],
    ignore_boxes: list[BBox],
    iou_threshold: float,
) -> MatchResult:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, detection in enumerate(detections):
        for gt_index, target in enumerate(ground_truth):
            overlap = iou(detection.bbox_xyxy, target.bbox_xyxy)
            if overlap >= iou_threshold:
                candidates.append((overlap, prediction_index, gt_index))
    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, prediction_index, gt_index in sorted(candidates, reverse=True):
        if prediction_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(prediction_index)
        matched_ground_truth.add(gt_index)
        matches.append((prediction_index, gt_index, overlap))
    ignored_predictions: list[int] = []
    false_positives: list[int] = []
    for prediction_index, detection in enumerate(detections):
        if prediction_index in matched_predictions:
            continue
        ignored = any(
            iou(detection.bbox_xyxy, ignore_box) >= iou_threshold for ignore_box in ignore_boxes
        )
        if ignored:
            ignored_predictions.append(prediction_index)
        else:
            false_positives.append(prediction_index)
    false_negatives = [
        gt_index for gt_index in range(len(ground_truth)) if gt_index not in matched_ground_truth
    ]
    return MatchResult(matches, false_positives, false_negatives, ignored_predictions)
