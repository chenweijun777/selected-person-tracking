"""Shared causal sparseOptFlow GMC with explicit estimation quality.

Reconstructs camera motion from saved images; these are not matrices saved by
the historical T2 run. Feature settings match the local Ultralytics GMC.
"""

import copy

import cv2
import numpy as np


class SparseCameraMotion:
    def __init__(self):
        self.feature_params = {
            "maxCorners": 400,
            "qualityLevel": 0.01,
            "minDistance": 0,
            "blockSize": 3,
            "useHarrisDetector": False,
            "k": 0.04,
        }
        self.downscale = 2
        self.previous = None
        self.previous_points = None

    def update(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (gray.shape[1] // self.downscale, gray.shape[0] // self.downscale))
        points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        warp = np.eye(2, 3)
        info = {
            "valid": False,
            "reason": "first_frame",
            "matches": 0,
            "inlier_ratio": 0.0,
            "residual_px": None,
        }
        try:
            if self.previous is not None and self.previous_points is not None:
                matched, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous, gray, self.previous_points, None
                )
                if matched is None or status is None:
                    info["reason"] = "flow_failed"
                else:
                    good = status.ravel().astype(bool)
                    previous = self.previous_points[good].reshape(-1, 2)
                    current = matched[good].reshape(-1, 2)
                    info["matches"] = len(previous)
                    info["reason"] = "insufficient_matches"
                    if len(previous) > 4:
                        estimate, inliers = cv2.estimateAffinePartial2D(
                            previous, current, method=cv2.RANSAC
                        )
                        info["reason"] = "affine_failed"
                        if estimate is not None and inliers is not None:
                            inlier = inliers.ravel().astype(bool)
                            ratio = float(np.mean(inlier))
                            residual = np.linalg.norm(
                                previous @ estimate[:, :2].T + estimate[:, 2] - current, axis=1
                            )
                            error = (
                                float(np.median(residual[inlier])) if inlier.any() else float("inf")
                            )
                            scale = float(np.sqrt(abs(np.linalg.det(estimate[:, :2]))))
                            angle = abs(float(np.arctan2(estimate[1, 0], estimate[0, 0])))
                            translation = np.linalg.norm(estimate[:, 2])
                            valid = (
                                np.all(np.isfinite(estimate))
                                and ratio >= 0.5
                                and error <= 3.0
                                and 0.8 <= scale <= 1.25
                                and angle <= np.deg2rad(15)
                                and translation <= 0.25 * np.hypot(*gray.shape)
                            )
                            info.update(
                                valid=bool(valid),
                                reason="ok" if valid else "quality_rejected",
                                inlier_ratio=ratio,
                                residual_px=error * self.downscale if np.isfinite(error) else None,
                            )
                            if valid:
                                warp = estimate.copy()
                                warp[:, 2] *= self.downscale
            elif self.previous is not None:
                info["reason"] = "no_previous_points"
        except cv2.error:
            info["reason"] = "opencv_failed"
        self.previous = gray.copy()
        self.previous_points = copy.copy(points)
        return warp, info
