from __future__ import annotations

from .types import BBox


def area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = area((max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)))
    union = area(a) + area(b) - intersection
    return intersection / union if union > 0 else 0.0


def contains_point(box: BBox, point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2
