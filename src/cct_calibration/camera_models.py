"""Camera projection and distortion models.

An intrinsics vector is ``[fx, fy, cx, cy, *coefficients]``, where the
coefficients are the *active* distortion terms of a :class:`CameraModel`.
Every model belongs to a distortion family whose full coefficient list (the
"superset") follows the parameter order of the corresponding COLMAP model;
inactive coefficients are identically zero and are not estimated.

Families (see ``colmap/src/colmap/sensor/models``):

``pinhole`` + ``opencv``
    k1 k2 p1 p2 k3 k4 k5 k6 -- COLMAP FULL_OPENCV / OpenCV rational model.
    Default ``k1 k2 p1 p2`` == COLMAP OPENCV (the historical 8-vector).
``fisheye`` + ``thin_prism``
    k1 k2 p1 p2 k3 k4 sx1 sy1 -- COLMAP THIN_PRISM_FISHEYE.
    Default ``k1 k2 k3 k4`` == COLMAP OPENCV_FISHEYE.
``fisheye`` + ``rad_tan_thin_prism``
    k0 k1 k2 k3 k4 k5 p0 p1 s0 s1 s2 s3 -- COLMAP RAD_TAN_THIN_PRISM_FISHEYE.

Fisheye angle and cheirality.  COLMAP's fisheye models use
``theta = atan(r / z)`` with ``z > 0``.  The same projection is written here
as ``theta = atan2(r, z)``: identical in front of the camera, and still
defined for directions beyond 90 degrees.  With ``check_cheirality`` (the
default) points with ``z <= 0`` are rejected exactly as in COLMAP.  For lenses
whose field of view exceeds 180 degrees the check can be switched off; then
only the direction straight behind the camera (theta = pi) is unprojectable.
The pinhole projection always requires ``z > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np

PROJECTIONS = ("pinhole", "fisheye")

_SUPERSETS: Dict[str, Tuple[str, ...]] = {
    "opencv": ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
    "thin_prism": ("k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"),
    "rad_tan_thin_prism": ("k0", "k1", "k2", "k3", "k4", "k5", "p0", "p1", "s0", "s1", "s2", "s3"),
}
_FAMILIES_BY_PROJECTION = {
    "pinhole": ("opencv",),
    "fisheye": ("thin_prism", "rad_tan_thin_prism"),
}
# Named distortion presets: (family, coefficients).
_PRESETS: Dict[str, Dict[str, Tuple[str, Tuple[str, ...]]]] = {
    "pinhole": {
        "opencv": ("opencv", ("k1", "k2", "p1", "p2")),
        "full_opencv": ("opencv", _SUPERSETS["opencv"]),
        "none": ("opencv", ()),
    },
    "fisheye": {
        "opencv_fisheye": ("thin_prism", ("k1", "k2", "k3", "k4")),
        "thin_prism_fisheye": ("thin_prism", _SUPERSETS["thin_prism"]),
        "rad_tan_thin_prism_fisheye": ("rad_tan_thin_prism", _SUPERSETS["rad_tan_thin_prism"]),
        "none": ("thin_prism", ()),
    },
}
DEFAULT_DISTORTION = {"pinhole": "opencv", "fisheye": "opencv_fisheye"}

_DEPTH_EPS = 1e-8
_UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)
# Beyond this angle a fisheye direction has no finite pinhole-plane image.
_MAX_PINHOLE_THETA = 0.5 * np.pi - 1e-6
_BEHIND_THETA = np.pi - 1e-6


@dataclass(frozen=True)
class CameraModel:
    """Projection type, distortion family and the estimated coefficients."""

    projection: str = "pinhole"
    family: str = "opencv"
    coefficients: Tuple[str, ...] = ("k1", "k2", "p1", "p2")
    check_cheirality: bool = True

    def __post_init__(self) -> None:
        if self.projection not in PROJECTIONS:
            raise ValueError(f"unknown camera model {self.projection!r}; choose from {PROJECTIONS}")
        if self.family not in _FAMILIES_BY_PROJECTION[self.projection]:
            raise ValueError(f"distortion family {self.family!r} is not available for {self.projection}")
        superset = _SUPERSETS[self.family]
        unknown = [name for name in self.coefficients if name not in superset]
        if unknown:
            raise ValueError(f"{self.family} distortion has no coefficient(s) {unknown}; available: {superset}")
        if len(set(self.coefficients)) != len(self.coefficients):
            raise ValueError("distortion coefficients must not repeat")
        ordered = tuple(name for name in superset if name in self.coefficients)
        if ordered != tuple(self.coefficients):
            object.__setattr__(self, "coefficients", ordered)
        if self.projection == "pinhole" and not self.check_cheirality:
            raise ValueError("the behind-camera check can only be disabled for the fisheye model")

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------
    @property
    def superset(self) -> Tuple[str, ...]:
        return _SUPERSETS[self.family]

    @property
    def num_params(self) -> int:
        return 4 + len(self.coefficients)

    @property
    def param_names(self) -> Tuple[str, ...]:
        return ("fx", "fy", "cx", "cy", *self.coefficients)

    @property
    def is_fisheye(self) -> bool:
        return self.projection == "fisheye"

    @property
    def is_opencv_pinhole(self) -> bool:
        """True when OpenCV's (non-fisheye) distortion functions apply."""
        return self.projection == "pinhole" and self.family == "opencv"

    @property
    def is_legacy_opencv8(self) -> bool:
        """The historical ``[fx,fy,cx,cy,k1,k2,p1,p2]`` COLMAP OPENCV model."""
        return self.is_opencv_pinhole and self.coefficients == ("k1", "k2", "p1", "p2")

    @property
    def key(self) -> str:
        """Stable identifier stored with cached, model-dependent data."""
        flag = "" if self.check_cheirality else "/no-cheirality"
        return f"{self.projection}/{self.family}/{','.join(self.coefficients) or 'none'}{flag}"

    @property
    def colmap_model_name(self) -> str:
        active = set(self.coefficients)
        if self.family == "opencv":
            return "OPENCV" if active <= {"k1", "k2", "p1", "p2"} else "FULL_OPENCV"
        if self.family == "thin_prism":
            return "OPENCV_FISHEYE" if active <= {"k1", "k2", "k3", "k4"} else "THIN_PRISM_FISHEYE"
        return "RAD_TAN_THIN_PRISM_FISHEYE"

    def describe(self) -> str:
        coefficients = " ".join(self.coefficients) or "no distortion"
        text = f"{self.projection} ({coefficients}; COLMAP {self.colmap_model_name})"
        if self.is_fisheye and not self.check_cheirality:
            text += ", behind-camera check off"
        return text

    def to_dict(self) -> dict:
        return {
            "projection": self.projection,
            "distortion_family": self.family,
            "coefficients": list(self.coefficients),
            "parameter_names": list(self.param_names),
            "colmap_model": self.colmap_model_name,
            "check_cheirality": self.check_cheirality,
        }

    def with_coefficients(self, coefficients: Iterable[str]) -> "CameraModel":
        return CameraModel(self.projection, self.family, tuple(coefficients), self.check_cheirality)

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def initial_intrinsics(self, focal: float, width: float, height: float) -> np.ndarray:
        values = np.zeros(self.num_params, dtype=np.float64)
        values[:4] = [focal, focal, width / 2.0, height / 2.0]
        return values

    def full_coefficients(self, intrinsics: np.ndarray) -> np.ndarray:
        """All superset coefficients (inactive ones are zero)."""
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        if intrinsics.shape[-1] != self.num_params:
            raise ValueError(
                f"{self.describe()} expects {self.num_params} intrinsics, got {intrinsics.shape[-1]}"
            )
        full = np.zeros(len(self.superset), dtype=np.float64)
        for value, name in zip(intrinsics[4:], self.coefficients):
            full[self.superset.index(name)] = value
        return full

    def coefficient_dict(self, intrinsics: np.ndarray) -> Dict[str, float]:
        return {name: float(value) for name, value in zip(self.param_names, intrinsics)}

    def coefficient_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Box bounds for the active coefficients (shared by all solvers).

        The leading terms keep the historical [-1, 1] range; higher orders,
        whose basis functions are small near the axis, get a wider range.
        """
        wide = {"k3", "k4", "k5", "k6"} if self.family == "opencv" else (
            {"k3", "k4"} if self.family == "thin_prism" else {"k2", "k3", "k4", "k5", "s1", "s3"}
        )
        limits = np.array([10.0 if name in wide else 1.0 for name in self.coefficients], dtype=np.float64)
        return -limits, limits

    def opencv_distortion(self, intrinsics: np.ndarray) -> np.ndarray | None:
        """OpenCV distortion vector (k1,k2,p1,p2[,k3[,k4,k5,k6]]) or None."""
        if not self.is_opencv_pinhole:
            return None
        full = self.full_coefficients(intrinsics)
        active = set(self.coefficients)
        if active & {"k4", "k5", "k6"}:
            return full
        if "k3" in active:
            return full[:5].copy()
        return full[:4].copy()

    def colmap_params(self, intrinsics: np.ndarray) -> np.ndarray:
        """Parameters in the order of :attr:`colmap_model_name`."""
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        full = self.full_coefficients(intrinsics)
        names = _COLMAP_PARAM_NAMES[self.colmap_model_name]
        lookup = dict(zip(self.superset, full))
        return np.concatenate([intrinsics[:4], [lookup[name] for name in names]])

    def intrinsics_from_colmap(self, model_name: str, params: Sequence[float]) -> np.ndarray:
        """Convert COLMAP camera parameters to this model's intrinsics.

        Coefficients are carried over only when the COLMAP model belongs to
        the same distortion family (so a name has the same meaning); terms
        this model does not estimate are dropped, missing ones start at zero.
        """
        params = np.asarray(params, dtype=np.float64)
        names = _COLMAP_PARAM_NAMES.get(model_name)
        if names is None or len(params) != 4 + len(names):
            raise ValueError(f"unsupported COLMAP camera model {model_name} with {len(params)} parameters")
        values = np.zeros(self.num_params, dtype=np.float64)
        values[:4] = params[:4]
        if _COLMAP_FAMILY[model_name] == (self.projection, self.family):
            lookup = dict(zip(names, params[4:]))
            for index, name in enumerate(self.coefficients, start=4):
                values[index] = lookup.get(name, 0.0)
        return values

    def kalibr_entry(self, intrinsics: np.ndarray) -> dict | None:
        """Kalibr camera/distortion description, or None if Kalibr cannot represent it."""
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        full = self.full_coefficients(intrinsics)
        active = set(self.coefficients)
        if self.family == "opencv" and active <= {"k1", "k2", "p1", "p2"}:
            return {"camera_model": "pinhole", "distortion_model": "radtan",
                    "distortion_coeffs": [float(value) for value in full[:4]]}
        if self.family == "thin_prism" and active <= {"k1", "k2", "k3", "k4"}:
            lookup = dict(zip(self.superset, full))
            return {"camera_model": "pinhole", "distortion_model": "equidistant",
                    "distortion_coeffs": [float(lookup[name]) for name in ("k1", "k2", "k3", "k4")]}
        return None

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    def _family_from_camera(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.projection == "pinhole":
                valid = z > _DEPTH_EPS
                safe_z = np.where(valid, z, 1.0)
                return x / safe_z, y / safe_z, valid
            radius = np.hypot(x, y)
            theta = np.arctan2(radius, z)
            if self.check_cheirality:
                valid = z > _DEPTH_EPS
            else:
                valid = (theta < _BEHIND_THETA) & ((radius > 0.0) | (z > 0.0))
            tiny = radius <= 1e-12 * np.maximum(np.abs(z), 1e-300)
            scale = np.where(tiny, 1.0 / np.where(z > 0.0, z, 1.0), theta / np.where(tiny, 1.0, radius))
            return x * scale, y * scale, valid

    def _distort(self, full: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Distort undistorted family coordinates (normalized plane or theta-scaled)."""
        if self.family == "opencv":
            k1, k2, p1, p2, k3, k4, k5, k6 = full
            a2, b2, ab = a * a, b * b, a * b
            r2 = a2 + b2
            r4 = r2 * r2
            r6 = r4 * r2
            radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
            return (a * radial + 2.0 * p1 * ab + p2 * (r2 + 2.0 * a2),
                    b * radial + p1 * (r2 + 2.0 * b2) + 2.0 * p2 * ab)
        if self.family == "thin_prism":
            k1, k2, p1, p2, k3, k4, sx1, sy1 = full
            a2, b2, ab = a * a, b * b, a * b
            t2 = a2 + b2
            t4 = t2 * t2
            radial = k1 * t2 + k2 * t4 + k3 * t4 * t2 + k4 * t4 * t4
            return (a + a * radial + 2.0 * p1 * ab + p2 * (t2 + 2.0 * a2) + sx1 * t2,
                    b + b * radial + 2.0 * p2 * ab + p1 * (t2 + 2.0 * b2) + sy1 * t2)
        # rad_tan_thin_prism
        radial_coefficients, (p0, p1), (s0, s1, s2, s3) = full[:6], full[6:8], full[8:]
        t2 = a * a + b * b
        th_radial = np.ones_like(t2)
        power = np.ones_like(t2)
        for coefficient in radial_coefficients:
            power = power * t2
            th_radial = th_radial + coefficient * power
        x, y = th_radial * a, th_radial * b
        x2, y2, xy = x * x, y * y, x * y
        r2 = x2 + y2
        r4 = r2 * r2
        return (x + 2.0 * p1 * xy + p0 * (r2 + 2.0 * x2) + s0 * r2 + s1 * r4,
                y + 2.0 * p0 * xy + p1 * (r2 + 2.0 * y2) + s2 * r2 + s3 * r4)

    def project(self, intrinsics: np.ndarray, points_camera: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points (..., 3) to pixels (..., 2) and a validity mask."""
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        points_camera = np.asarray(points_camera, dtype=np.float64)
        a, b, valid = self._family_from_camera(points_camera)
        xd, yd = self._distort(self.full_coefficients(intrinsics), a, b)
        pixels = np.stack([intrinsics[0] * xd + intrinsics[2], intrinsics[1] * yd + intrinsics[3]], axis=-1)
        valid = valid & np.all(np.isfinite(pixels), axis=-1)
        return pixels, valid

    def project_point(self, intrinsics: np.ndarray, pose: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Project one world point; unprojectable points map to (1e6, 1e6)."""
        rotation, _ = cv2.Rodrigues(np.asarray(pose[:3], dtype=np.float64))
        pixel, valid = self.project(intrinsics, rotation @ point + pose[3:])
        if not bool(valid):
            return np.array([1e6, 1e6], dtype=np.float64)
        return pixel

    def project_world(self, intrinsics: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                      points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return self.project(intrinsics, points @ rotation.T + np.asarray(tvec, dtype=np.float64).reshape(3))

    def is_projectable(self, points_camera: np.ndarray) -> np.ndarray:
        return self._family_from_camera(np.asarray(points_camera, dtype=np.float64))[2]

    # ------------------------------------------------------------------
    # Inverse model
    # ------------------------------------------------------------------
    def _undistort_family(self, intrinsics: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        """Invert the distortion; returns undistorted family coordinates (N, 2), NaN if not converged."""
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        full = self.full_coefficients(intrinsics)
        target_a = (pixels[:, 0] - intrinsics[2]) / intrinsics[0]
        target_b = (pixels[:, 1] - intrinsics[3]) / intrinsics[1]
        if not np.any(full):
            return np.column_stack([target_a, target_b])
        a, b = target_a.copy(), target_b.copy()
        # Vectorised Newton iteration with a forward-difference Jacobian, as
        # in COLMAP's IterativeUndistortion.
        for _ in range(100):
            fa, fb = self._distort(full, a, b)
            ea, eb = fa - target_a, fb - target_b
            step_a = 1e-9 * np.maximum(1.0, np.abs(a))
            step_b = 1e-9 * np.maximum(1.0, np.abs(b))
            pa, pb = self._distort(full, a + step_a, b)
            qa, qb = self._distort(full, a, b + step_b)
            j00, j10 = (pa - fa) / step_a, (pb - fb) / step_a
            j01, j11 = (qa - fa) / step_b, (qb - fb) / step_b
            det = j00 * j11 - j01 * j10
            with np.errstate(divide="ignore", invalid="ignore"):
                da = (j11 * ea - j01 * eb) / det
                db = (-j10 * ea + j00 * eb) / det
            a, b = a - da, b - db
            if np.all(~np.isfinite(da) | (np.abs(da) + np.abs(db) < 1e-14)):
                break
        fa, fb = self._distort(full, a, b)
        error = np.hypot(fa - target_a, fb - target_b)
        bad = ~np.isfinite(error) | (error > 1e-9 * np.maximum(1.0, np.hypot(target_a, target_b)))
        a[bad] = np.nan
        b[bad] = np.nan
        return np.column_stack([a, b])

    def bearings(self, intrinsics: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        """Unit viewing rays (N, 3) for distorted pixels (NaN where not invertible)."""
        if self.projection == "pinhole":
            normalized = self.normalize(intrinsics, pixels)
            rays = np.column_stack([normalized, np.ones(len(normalized))])
            return rays / np.linalg.norm(rays, axis=1, keepdims=True)
        family = self._undistort_family(intrinsics, pixels)
        theta = np.hypot(family[:, 0], family[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(theta > 1e-12, np.sin(theta) / np.where(theta > 1e-12, theta, 1.0), 1.0)
        rays = np.column_stack([family[:, 0] * scale, family[:, 1] * scale, np.cos(theta)])
        if self.check_cheirality:
            rays[theta >= 0.5 * np.pi] = np.nan
        return rays

    def normalize(self, intrinsics: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        """Undistorted normalized pinhole coordinates x/z, y/z (N, 2).

        Equivalent to ``cv2.undistortPoints(pixels, K, D)``.  Fisheye
        directions at or beyond 90 degrees have no such coordinates (NaN).
        """
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if self.is_opencv_pinhole:
            if len(pixels) == 0:
                return np.zeros((0, 2), dtype=np.float64)
            # cv2.undistortPoints stops after a few fixed-point iterations:
            # ~1e-4 px error for k1 k2 p1 p2 at the corners of a normal lens,
            # but tenths of a pixel once k3 is estimated.  Iterate to
            # convergence instead, and reject points that did not converge.
            normalized = cv2.undistortPointsIter(
                pixels.reshape(-1, 1, 2), _camera_matrix(intrinsics), self.opencv_distortion(intrinsics),
                None, None, _UNDISTORT_CRITERIA,
            ).reshape(-1, 2).astype(np.float64)
            check = cv2.projectPoints(
                np.column_stack([normalized, np.ones(len(normalized))]), np.zeros(3), np.zeros(3),
                _camera_matrix(intrinsics), self.opencv_distortion(intrinsics),
            )[0].reshape(-1, 2)
            normalized[~(np.linalg.norm(check - pixels, axis=1) <= 1e-4)] = np.nan
            return normalized
        family = self._undistort_family(intrinsics, pixels)
        theta = np.hypot(family[:, 0], family[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(theta > 1e-12, np.tan(theta) / np.where(theta > 1e-12, theta, 1.0), 1.0)
        normalized = family * scale[:, None]
        normalized[~(theta < _MAX_PINHOLE_THETA)] = np.nan
        return normalized

    def undistort_pixels(self, intrinsics: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        """Ideal pinhole pixels with the same K (``cv2.undistortPoints(..., P=K)``)."""
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        normalized = self.normalize(intrinsics, pixels)
        return np.column_stack([
            intrinsics[0] * normalized[:, 0] + intrinsics[2],
            intrinsics[1] * normalized[:, 1] + intrinsics[3],
        ])

    def distort_pixels(self, intrinsics: np.ndarray, ideal_pixels: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`undistort_pixels` (NaN where not projectable)."""
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        ideal_pixels = np.asarray(ideal_pixels, dtype=np.float64).reshape(-1, 2)
        rays = np.column_stack([
            (ideal_pixels[:, 0] - intrinsics[2]) / intrinsics[0],
            (ideal_pixels[:, 1] - intrinsics[3]) / intrinsics[1],
            np.ones(len(ideal_pixels)),
        ])
        if self.is_opencv_pinhole:
            if len(rays) == 0:
                return np.zeros((0, 2), dtype=np.float64)
            return cv2.projectPoints(
                rays, np.zeros(3), np.zeros(3), _camera_matrix(intrinsics), self.opencv_distortion(intrinsics),
            )[0].reshape(-1, 2)
        pixels, valid = self.project(intrinsics, rays)
        pixels[~valid] = np.nan
        return pixels

    # ------------------------------------------------------------------
    # OpenCV PnP support
    # ------------------------------------------------------------------
    def pnp_inputs(
        self, intrinsics: np.ndarray, image_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        """Camera matrix, distortion, image points and validity mask for cv2.solvePnP*.

        OpenCV-pinhole models pass the measured pixels and distortion vector
        unchanged.  Other models pass ideal pinhole pixels (same K, no
        distortion); directions without a pinhole image are masked out.
        """
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        matrix = _camera_matrix(intrinsics)
        if self.is_opencv_pinhole:
            return matrix, self.opencv_distortion(intrinsics), image_points, np.ones(len(image_points), bool)
        ideal = self.undistort_pixels(intrinsics, image_points)
        return matrix, None, ideal, np.all(np.isfinite(ideal), axis=1)


def _camera_matrix(intrinsics: np.ndarray) -> np.ndarray:
    return np.array(
        [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


_COLMAP_PARAM_NAMES: Dict[str, Tuple[str, ...]] = {
    "OPENCV": ("k1", "k2", "p1", "p2"),
    "FULL_OPENCV": _SUPERSETS["opencv"],
    "OPENCV_FISHEYE": ("k1", "k2", "k3", "k4"),
    "THIN_PRISM_FISHEYE": _SUPERSETS["thin_prism"],
    "RAD_TAN_THIN_PRISM_FISHEYE": _SUPERSETS["rad_tan_thin_prism"],
}
_COLMAP_FAMILY: Dict[str, Tuple[str, str]] = {
    "OPENCV": ("pinhole", "opencv"),
    "FULL_OPENCV": ("pinhole", "opencv"),
    "OPENCV_FISHEYE": ("fisheye", "thin_prism"),
    "THIN_PRISM_FISHEYE": ("fisheye", "thin_prism"),
    "RAD_TAN_THIN_PRISM_FISHEYE": ("fisheye", "rad_tan_thin_prism"),
}

DEFAULT_CAMERA_MODEL = CameraModel()


def build_camera_model(
    model: str = "pinhole",
    distortion: Sequence[str] | None = None,
    check_cheirality: bool = True,
) -> CameraModel:
    """Build a model from the ``--model`` / ``--dist`` command-line values.

    ``distortion`` is either a single preset name (e.g. ``full_opencv``,
    ``opencv_fisheye``, ``thin_prism_fisheye``, ``none``) or a list of
    coefficient names (e.g. ``k1 k2 k3 p1 p2``; commas are also accepted).
    Fisheye coefficient lists are read in the thin-prism family (the
    OPENCV_FISHEYE extension) when possible, otherwise in the
    RAD_TAN_THIN_PRISM family.
    """
    projection = model.lower().strip()
    if projection not in PROJECTIONS:
        raise ValueError(f"--model must be one of {PROJECTIONS}, got {model!r}")
    tokens = [token.strip().lower() for item in (distortion or []) for token in str(item).split(",")]
    tokens = [token for token in tokens if token]
    if not tokens:
        tokens = [DEFAULT_DISTORTION[projection]]
    presets = _PRESETS[projection]
    if len(tokens) == 1 and tokens[0] in presets:
        family, coefficients = presets[tokens[0]]
        return CameraModel(projection, family, coefficients, check_cheirality)
    if any(token in presets for token in tokens):
        raise ValueError(f"a distortion preset ({', '.join(presets)}) cannot be combined with other names")
    for family in _FAMILIES_BY_PROJECTION[projection]:
        if all(token in _SUPERSETS[family] for token in tokens):
            return CameraModel(projection, family, tuple(tokens), check_cheirality)
    available = sorted({name for family in _FAMILIES_BY_PROJECTION[projection] for name in _SUPERSETS[family]})
    raise ValueError(
        f"invalid --dist {' '.join(tokens)!r} for the {projection} model; use a preset "
        f"({', '.join(presets)}) or coefficient names from one family: "
        + "; ".join(f"{family}: {' '.join(_SUPERSETS[family])}" for family in _FAMILIES_BY_PROJECTION[projection])
        + f" (known names: {', '.join(available)})"
    )


def add_camera_model_arguments(parser) -> None:
    """Register ``--model``, ``--dist`` and ``--no-cheirality-check``."""
    parser.add_argument(
        "--model", choices=PROJECTIONS, default="pinhole",
        help="Camera projection model (default: pinhole).",
    )
    parser.add_argument(
        "--dist", nargs="+", default=None, metavar="NAME",
        help="Distortion coefficients to estimate, as a preset or a list of names. "
             "pinhole: default 'opencv' = k1 k2 p1 p2 (COLMAP OPENCV); any subset of "
             "k1 k2 p1 p2 k3 k4 k5 k6 (e.g. 'k1 k2 k3 p1 p2'; exported as FULL_OPENCV), "
             "'full_opencv' or 'none'. fisheye: default 'opencv_fisheye' = k1 k2 k3 k4 "
             "(COLMAP OPENCV_FISHEYE); any subset of k1 k2 k3 k4 p1 p2 sx1 sy1 "
             "(THIN_PRISM_FISHEYE), 'thin_prism_fisheye', 'rad_tan_thin_prism_fisheye' "
             "(or a subset of k0..k5 p0 p1 s0..s3) or 'none'.",
    )
    parser.add_argument(
        "--no-cheirality-check", action="store_true",
        help="Fisheye only: accept directions behind the camera plane (field of view above "
             "180 degrees). By default, as in COLMAP, points with z <= 0 are not projectable.",
    )


def camera_model_from_args(args) -> CameraModel:
    return build_camera_model(args.model, args.dist, check_cheirality=not args.no_cheirality_check)
