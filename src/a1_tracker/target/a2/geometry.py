"""Camera-compensated box geometry."""

import numpy as np


def corners(box):
    x1, y1, x2, y2 = box
    return np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=float)


def geometry(points):
    return (
        points.mean(axis=0),
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[3] - points[0]),
    )
