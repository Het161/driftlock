#!/usr/bin/env python3
"""Shared helpers for the Drift-Sense navigation-error-recovery pipeline.

Deliberately small (see PROJECT_SPEC.md §2): this module holds only the
utilities that BOTH sides of the problem need -- the dataset generator and the
localizer -- so the two cannot silently drift apart on image conventions,
seeding, or the affine geometry contract.

Conventions enforced here
-------------------------
* Images are single-channel grayscale ``float32`` in ``[0, 1]`` in memory and
  8-bit PNG on disk.
* Pixel coordinates follow OpenCV's convention: integer coordinate ``i`` is the
  *centre* of pixel ``i``, which spans ``[i - 0.5, i + 0.5)``.
* Every random draw comes from an explicitly seeded ``numpy.random.Generator``.
  Nothing in this project touches the global numpy random state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Upper clip bound of the sensor-noise model (PROJECT_SPEC.md §3.4).  Noise is
#: allowed to overshoot the true signal range slightly, mirroring a real
#: detector, so the float -> uint8 mapping *rescales* by this ceiling instead of
#: saturating at 1.0 -- otherwise the deliberate head-room would be thrown away.
PIXEL_CEILING: float = 1.05

#: Number of independent RNG streams derived per pair.  The order is fixed and
#: each concern owns its own stream, so a CLI flag that disables one feature
#: cannot perturb any other: ``--noise-level`` leaves geometry untouched,
#: ``--pure-lattice`` leaves the lattice and geometry untouched, and
#: ``--no-defects`` leaves the superstructure untouched.  That is what makes
#: the Phase-4 robustness plots and the SPEC_AMENDMENT_v1.1 §D ablation gate
#: controlled comparisons rather than independent redraws.
N_RNG_STREAMS: int = 6

STREAM_NAMES: tuple[str, ...] = (
    "structure",       # base lattice: F, line jitter, global gain/bias
    "geometry",        # stage error (theta, scale), drift vector, search PSF
    "superstructure",  # sense-amp / wordline-driver stripes  (v1.1 §A.1-2)
    "defects",         # contamination particles              (v1.1 §A.3)
    "ref_noise",       # reference capture only
    "search_noise",    # search capture only
)


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Deterministic seeding
# --------------------------------------------------------------------------- #


def derive_seeds(master_seed: int, pair_index: int, n_streams: int = N_RNG_STREAMS) -> tuple[int, ...]:
    """Derive ``n_streams`` independent 32-bit seeds for one pair.

    Uses ``numpy.random.SeedSequence`` so the streams are provably independent
    (not merely ``seed + k``), while staying plain ints so they can be recorded
    verbatim in ``ground_truth.json`` and replayed later.

    Args:
        master_seed: The dataset-wide ``--seed`` value.
        pair_index: 1-based index of the pair.
        n_streams: How many independent streams to derive.

    Returns:
        A tuple of ``n_streams`` non-negative ints.
    """
    if n_streams < 1:
        raise ValueError(f"n_streams must be >= 1, got {n_streams}")
    seq = np.random.SeedSequence([int(master_seed), int(pair_index)])
    return tuple(int(v) for v in seq.generate_state(n_streams, dtype=np.uint32))


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh PCG64 generator for ``seed``."""
    return np.random.default_rng(int(seed))


# --------------------------------------------------------------------------- #
# Image I/O
# --------------------------------------------------------------------------- #


def save_gray_png(path: str | Path, image: np.ndarray, ceiling: float = PIXEL_CEILING) -> None:
    """Write a float image in ``[0, ceiling]`` as an 8-bit grayscale PNG.

    The mapping is ``uint8 = round(clip(image, 0, ceiling) / ceiling * 255)``.

    Raises:
        ValueError: If ``image`` is not 2-D or ``ceiling`` is not positive.
        IOError: If OpenCV fails to write the file.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {image.shape}")
    if not ceiling > 0:
        raise ValueError(f"ceiling must be positive, got {ceiling}")
    path = Path(path)
    ensure_dir(path.parent)
    scaled = np.clip(image.astype(np.float32), 0.0, ceiling) * (255.0 / ceiling)
    if not cv2.imwrite(str(path), np.rint(scaled).astype(np.uint8)):
        raise IOError(f"cv2.imwrite failed to write {path}")


def load_gray_float(path: str | Path) -> np.ndarray:
    """Load an 8-bit grayscale PNG as ``float32`` in ``[0, 1]``.

    Note that this is the inverse of :func:`save_gray_png` only up to the global
    ``ceiling`` factor.  That is harmless: every downstream step (normalized
    cross-correlation, z-scoring) is invariant to a global affine rescale of
    intensity.

    Raises:
        FileNotFoundError: If the file is missing or not decodable as an image.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img.astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Intensity transforms
# --------------------------------------------------------------------------- #


def normalize_unit(image: np.ndarray) -> np.ndarray:
    """Min-max rescale to ``[0, 1]``; returns zeros for a constant image."""
    img = image.astype(np.float32, copy=False)
    lo = float(img.min())
    hi = float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def zscore(image: np.ndarray) -> np.ndarray:
    """Rescale to zero mean / unit standard deviation (float32)."""
    img = image.astype(np.float32, copy=False)
    std = float(img.std())
    if std < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - float(img.mean())) / std).astype(np.float32)


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """Return the 3x3 Sobel gradient magnitude of ``image`` as float32."""
    img = image.astype(np.float32, copy=False)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic Gaussian blur; ``sigma <= 0`` returns a copy."""
    img = image.astype(np.float32, copy=False)
    if sigma <= 0:
        return img.copy()
    # ksize=(0, 0) lets OpenCV derive the kernel size from sigma.
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def rotation_scale_matrix(center: tuple[float, float], theta_deg: float, scale: float) -> np.ndarray:
    """Build the 2x3 affine ``A``: rotate by ``theta_deg`` and scale about ``center``.

    This is exactly ``cv2.getRotationMatrix2D``, wrapped so that the generator
    and the localizer are guaranteed to use the same sign convention.  The
    matrix is the *forward* source -> destination map, which is also how
    ``cv2.warpAffine`` interprets it (it inverts internally).
    """
    return cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                   float(theta_deg), float(scale))


def apply_affine(matrix: np.ndarray, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to an ``(N, 2)`` array of ``(x, y)`` points."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.shape[1] != 2:
        raise ValueError(f"expected points of shape (N, 2), got {pts.shape}")
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (2, 3):
        raise ValueError(f"expected a 2x3 affine matrix, got {mat.shape}")
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return homogeneous @ mat.T
