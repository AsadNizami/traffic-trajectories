import numpy as np
import pytest

from traj.geometry import GroundPlane

# A synthetic perspective: world square 10 x 50 m seen as a trapezoid.
IMG = [[400, 700], [800, 700], [650, 300], [550, 300]]
WLD = [[0, 0], [10, 0], [10, 50], [0, 50]]


def test_calibration_points_map_exactly():
    plane = GroundPlane(IMG, WLD)
    np.testing.assert_allclose(plane.to_world(IMG), WLD, atol=1e-6)


def test_round_trip():
    plane = GroundPlane(IMG, WLD)
    pts = np.array([[5, 5], [2.5, 40], [9, 20]], dtype=float)
    np.testing.assert_allclose(plane.to_world(plane.to_image(pts)), pts, atol=1e-6)


def test_perspective_compresses_far_distances():
    plane = GroundPlane(IMG, WLD)
    near = plane.to_world([[600, 700], [600, 650]])
    far = plane.to_world([[600, 350], [600, 300]])
    # Same 50 px step covers far more road near the horizon.
    assert abs(far[1, 1] - far[0, 1]) > 3 * abs(near[1, 1] - near[0, 1])


def test_max_range():
    plane = GroundPlane(IMG, WLD, max_range_m=30)
    assert plane.in_range([[0, 10], [0, 40]]).tolist() == [True, False]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        GroundPlane(IMG[:3], WLD[:3])


def test_draw_grid_keeps_shape():
    frame = np.zeros((720, 1280, 3), np.uint8)
    out = GroundPlane(IMG, WLD).draw_grid(frame)
    assert out.shape == frame.shape and out.any()
