"""Image-to-ground-plane mapping via a planar homography."""

import json
from pathlib import Path

import cv2
import numpy as np


class GroundPlane:
    """Maps pixel coordinates to metric road-plane coordinates (and back).

    Needs >= 4 point correspondences between the image and the road plane, e.g.
    corners of lane markings whose real-world spacing is known. Optional
    `max_range_m` (distance from the world origin) marks where the mapping is
    still accurate; for an oblique camera, 1 px spans meters far away.
    """

    def __init__(self, image_points, world_points, max_range_m=None):
        img = np.asarray(image_points, dtype=np.float64)
        wld = np.asarray(world_points, dtype=np.float64)
        if img.shape != wld.shape or img.ndim != 2 or img.shape[1] != 2 or len(img) < 4:
            raise ValueError("need >= 4 matching (x, y) image and world points")
        self.image_points, self.world_points = img, wld
        self.max_range_m = max_range_m
        self.H, _ = cv2.findHomography(img, wld)
        if self.H is None:
            raise ValueError("degenerate calibration points (are 3 of them collinear?)")
        self.H_inv = np.linalg.inv(self.H)

    @classmethod
    def from_dict(cls, d):
        return cls(d["image_points"], d["world_points"], d.get("max_range_m"))

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @staticmethod
    def _apply(H, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    def to_world(self, pixel_points):
        return self._apply(self.H, pixel_points)

    def in_range(self, world_points):
        if self.max_range_m is None:
            return np.ones(len(world_points), dtype=bool)
        return np.hypot(*np.asarray(world_points).T) <= self.max_range_m

    def to_image(self, world_points):
        return self._apply(self.H_inv, world_points)

    def draw_grid(self, frame, step=5.0, extent=None):
        """Overlay a metric grid (default 5 m) on the frame to sanity-check the calibration."""
        out = frame.copy()
        (x0, y0), (x1, y1) = extent or (self.world_points.min(0) - 10, self.world_points.max(0) + 10)
        for x in np.arange(np.floor(x0 / step) * step, x1 + 1e-6, step):
            self._polyline(out, [(x, y) for y in np.linspace(y0, y1, 50)], (0, 255, 255))
        for y in np.arange(np.floor(y0 / step) * step, y1 + 1e-6, step):
            self._polyline(out, [(x, y) for x in np.linspace(x0, x1, 50)], (0, 255, 255))
        for (u, v), (wx, wy) in zip(self.image_points, self.world_points, strict=True):
            cv2.circle(out, (int(u), int(v)), 6, (0, 0, 255), -1)
            cv2.putText(out, f"({wx:g},{wy:g})m", (int(u) + 8, int(v) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return out

    def _polyline(self, img, world_pts, color):
        # Drop points beyond the horizon, where the projective scale w flips sign
        # relative to the calibration points (H is only defined up to scale).
        p = np.asarray(world_pts, dtype=np.float64)
        w = (np.c_[p, np.ones(len(p))] @ self.H_inv.T)[:, 2]
        ref = np.sign(np.r_[self.world_points[0], 1] @ self.H_inv[2])
        pts = self.to_image(p)[w * ref > 0]
        if len(pts) > 1:
            cv2.polylines(img, [pts.astype(np.int32)], False, color, 1, cv2.LINE_AA)
