"""Camera models agree with COLMAP and OpenCV, and invert consistently."""

import cv2
import numpy as np
import pycolmap
import pytest

from cct_calibration.camera_models import DEFAULT_CAMERA_MODEL, build_camera_model
from cct_calibration.run import project_point

CASES = [
    ("pinhole", ["opencv"]),
    ("pinhole", ["k1", "k2", "k3", "p1", "p2"]),
    ("pinhole", ["full_opencv"]),
    ("fisheye", ["opencv_fisheye"]),
    ("fisheye", ["k1", "k2", "k3", "k4", "p1", "p2"]),
    ("fisheye", ["thin_prism_fisheye"]),
    ("fisheye", ["rad_tan_thin_prism_fisheye"]),
]


def _points(count=200, seed=0):
    rng = np.random.default_rng(seed)
    directions = np.column_stack([rng.uniform(-0.8, 0.8, count), rng.uniform(-0.6, 0.6, count), np.ones(count)])
    return directions * rng.uniform(0.5, 3.0, (count, 1))


def _intrinsics(model, seed=1):
    rng = np.random.default_rng(seed)
    return np.concatenate([[1000.0, 1010.0, 640.0, 480.0], rng.normal(0.0, 0.02, len(model.coefficients))])


@pytest.mark.parametrize("projection,dist", CASES)
def test_projection_matches_colmap(projection, dist):
    model = build_camera_model(projection, dist)
    intrinsics = _intrinsics(model)
    camera = pycolmap.Camera(model=model.colmap_model_name, width=1280, height=960,
                             params=model.colmap_params(intrinsics).tolist())
    points = _points()
    ours, valid = model.project(intrinsics, points)
    assert valid.all()
    np.testing.assert_allclose(ours, camera.img_from_cam(points), atol=1e-9)
    np.testing.assert_allclose(
        model.intrinsics_from_colmap(model.colmap_model_name, model.colmap_params(intrinsics)), intrinsics,
    )


@pytest.mark.parametrize("projection,dist", CASES)
def test_undistortion_inverts_projection(projection, dist):
    model = build_camera_model(projection, dist)
    intrinsics = _intrinsics(model)
    points = _points()
    pixels, _ = model.project(intrinsics, points)
    rays = points / np.linalg.norm(points, axis=1, keepdims=True)
    np.testing.assert_allclose(model.bearings(intrinsics, pixels), rays, atol=1e-9)
    np.testing.assert_allclose(model.normalize(intrinsics, pixels), points[:, :2] / points[:, 2:], atol=1e-9)
    ideal = model.undistort_pixels(intrinsics, pixels)
    np.testing.assert_allclose(model.distort_pixels(intrinsics, ideal), pixels, atol=1e-6)


def test_opencv_compatibility():
    points = _points()
    matrix = np.array([[1000.0, 0.0, 640.0], [0.0, 1010.0, 480.0], [0.0, 0.0, 1.0]])
    for dist in (["opencv"], ["k1", "k2", "k3", "p1", "p2"], ["full_opencv"]):
        model = build_camera_model("pinhole", dist)
        intrinsics = _intrinsics(model)
        expected = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix,
                                     model.opencv_distortion(intrinsics))[0].reshape(-1, 2)
        np.testing.assert_allclose(model.project(intrinsics, points)[0], expected, atol=1e-9)
    model = build_camera_model("fisheye", None)
    intrinsics = _intrinsics(model)
    expected = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), matrix,
                                         intrinsics[4:])[0].reshape(-1, 2)
    np.testing.assert_allclose(model.project(intrinsics, points)[0], expected, atol=1e-9)


def test_default_model_matches_legacy_projection():
    intrinsics = np.array([3350.0, 3351.0, 3000.0, 1700.0, -0.1, 0.05, -2e-4, 3e-4])
    pose = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 2.0])
    for point in np.random.default_rng(2).normal(0.0, 1.0, (20, 3)):
        np.testing.assert_allclose(
            DEFAULT_CAMERA_MODEL.project_point(intrinsics, pose, point),
            project_point(intrinsics, pose, point), atol=1e-9,
        )


def test_cheirality_switch_for_wide_fisheye():
    intrinsics = np.array([300.0, 300.0, 640.0, 480.0, 0.01, -0.002, 0.0, 0.0])
    behind = np.array([[1.0, 0.2, -0.3], [0.0, 1.0, -0.1]])
    assert not build_camera_model("fisheye", None).project(intrinsics, behind)[1].any()
    wide = build_camera_model("fisheye", None, check_cheirality=False)
    pixels, valid = wide.project(intrinsics, behind)
    assert valid.all()
    np.testing.assert_allclose(wide.bearings(intrinsics, pixels),
                               behind / np.linalg.norm(behind, axis=1, keepdims=True), atol=1e-9)
    assert not wide.project(intrinsics, np.array([[0.0, 0.0, -1.0]]))[1].any()
    with pytest.raises(ValueError):
        build_camera_model("pinhole", None, check_cheirality=False)


def test_distortion_argument_parsing():
    assert DEFAULT_CAMERA_MODEL == build_camera_model("pinhole", None)
    model = build_camera_model("pinhole", ["k1,k2,k3", "p1", "p2"])
    assert model.param_names == ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    assert model.colmap_model_name == "FULL_OPENCV"
    assert model.opencv_distortion(np.arange(9.0)).tolist() == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert build_camera_model("fisheye", None).colmap_model_name == "OPENCV_FISHEYE"
    assert build_camera_model("fisheye", ["k1", "k2", "p1", "p2"]).colmap_model_name == "THIN_PRISM_FISHEYE"
    for projection, dist in (("pinhole", ["k1", "sx1"]), ("pinhole", ["opencv", "k3"]), ("fisheye", ["k9"])):
        with pytest.raises(ValueError):
            build_camera_model(projection, dist)
