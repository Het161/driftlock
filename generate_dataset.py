#!/usr/bin/env python3
"""Drift-Sense Phase 1 -- synthetic wafer image-pair generator.

Implements PROJECT_SPEC.md §3 as amended by SPEC_AMENDMENT_v1.1 (§A replaces
§3.1.2, §B replaces §3.1.4).  Produces physically-motivated reference/search
image pairs for the Applied Materials PS2 navigation-error-recovery task:

    reference : 1000x1000 crop of the die at 100x-equivalent resolution
    search    : 1000x1000 view of the same die at 10x, i.e. the reference
                pattern appears ~10x smaller (~100x100 px, ~1% of the frame),
                displaced by the tool's stage rotation/magnification error and
                corrupted by an *independent* physical capture.

Pipeline per pair
-----------------
    world (10000x10000 clean truth)
      = DRAM/FinFET lattice
      + sense-amplifier stripes (horizontal)      -- v1.1 §A.1
      + wordline-driver stripes (vertical)        -- v1.1 §A.2
      + contamination particles                   -- v1.1 §A.3
      |
      |-- crop 1000x1000 at (cx, cy) ---------------------> reference capture
      `-- warpAffine(A) then resize x0.1 (INTER_AREA) ----> search capture

    each capture: SEM edge-brightening -> optics PSF blur -> sensor noise
                  (+ scan-line correlated noise, search only)

Why the superstructure exists (v1.1 §A): a globally uniform lattice is
translation-invariant, so the true site carries no more correlation energy than
any other lattice cell and localization is ill-posed.  Real DRAM fields are
organised into subarray mats separated by sense-amp and decoder regions, and a
1000x1000 field at 10x spans many mats -- so block boundaries are visible
structure that makes the problem well-posed while cells stay locally periodic.
``--pure-lattice`` reproduces the degenerate uniform world on demand; it is the
control for the ablation gate and the honest-failure exhibit for the results
slide.  Note that LINE_INTENSITY_JITTER stays at the spec value of 0.05 -- the
amendment explicitly forbids papering over the modelling gap by raising it.

Where the ground truth comes from (v1.1 §B): the true position is sampled
*first* in search coordinates as a drift offset from the frame centre (the tool
navigates to the site and lands a short drift away, then captures), and the
world crop centre is back-solved through the inverse affine.  This is why the
official "return the match closest to the frame centre" rule is a physical
prior rather than an arbitrary tie-break.

Every physics-motivated constant below carries a ``# CITE: [Sn]`` tag pointing
at the corresponding entry in CITATIONS.md (see PROJECT_SPEC.md §7).

Example
-------
    python generate_dataset.py --style dram --num-pairs 30 --output-dir data \\
        --seed 42 --noise-level medium --preview
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from common import (
    N_RNG_STREAMS,
    PIXEL_CEILING,
    STREAM_NAMES,
    apply_affine,
    derive_seeds,
    ensure_dir,
    gaussian_blur,
    make_rng,
    normalize_unit,
    rotation_scale_matrix,
    save_gray_png,
    sobel_magnitude,
)

# --------------------------------------------------------------------------- #
# Tunable constants -- named and documented here, never inlined below
# (PROJECT_SPEC.md §9: "keep constants named and documented at the top").
# --------------------------------------------------------------------------- #

GENERATOR_VERSION = "phase1.1"

# --- geometry (PROJECT_SPEC.md §3.1) ---
WORLD_SIZE = 10_000          # clean "physical truth" canvas, 100x-equivalent px
CAPTURE_SIZE = 1_000         # both reference and search are 1000x1000
DOWNSAMPLE = 10              # search magnification ratio: 100x -> 10x
CROP_MARGIN = 1_500          # min distance of the crop centre from world edges
GT_FRAME_MARGIN = 80         # ground truth must land >= this far inside search
MAX_PLACEMENT_RESAMPLES = 64  # give up rather than loop forever (v1.1 §B.3)

THETA_RANGE_DEG = (-2.0, 2.0)   # stage rotation error
SCALE_RANGE = (0.97, 1.03)      # magnification jitter, ~+/-3%

# --- drift-prior placement (SPEC_AMENDMENT_v1.1 §B) ---
# CITE: [S11] Stage navigation lands a short drift from the intended site, so
# the target sits near the frame centre -- this is the physical basis for the
# official "closest to the centre" tie-break rule.
# Defaults are provisional: the organizers' own generator parameters are unknown
# until the official starter code drops (4 Aug webinar).  Re-align then (§B.5).
DRIFT_SIGMA = 120.0          # search-image px, sigma of |Normal| drift magnitude
DRIFT_CAP = 350.0            # search-image px, hard cap on drift magnitude

# --- DRAM structure (PROJECT_SPEC.md §3.1.2 lattice, retained verbatim) ---
# CITE: [S5] DRAM 6F^2 open-bitline cell -- word-line pitch 3F, bit-line pitch
# 2F, line width 1F, contact/via at every WL x BL intersection.
F_RANGE = (40.0, 80.0)       # base feature size F, in world px
BACKGROUND = 0.15
WL_PITCH_F, WL_WIDTH_F, WL_INTENSITY = 3.0, 1.0, 0.75   # CITE: [S5] word-lines
BL_PITCH_F, BL_WIDTH_F, BL_INTENSITY = 2.0, 1.0, 0.85   # CITE: [S5] bit-lines
VIA_RADIUS_F, VIA_INTENSITY = 0.4, 0.95                 # CITE: [S5] contacts
# Per-line process variation.  DO NOT RAISE THIS to fix localizability:
# SPEC_AMENDMENT_v1.1 measured that >=0.15 is unphysical (no working process has
# 15-30% line-to-line brightness variation) and would substitute a fake
# signature for the real modelling fix, which is the superstructure below.
LINE_INTENSITY_JITTER = 0.05
GLOBAL_BRIGHTNESS_JITTER = 0.03
GLOBAL_CONTRAST_JITTER = 0.05

# --- DRAM superstructure (SPEC_AMENDMENT_v1.1 §A.1-2) ---
# CITE: [S9] DRAM arrays are organised as subarray mats separated by
# sense-amplifier stripes and wordline-driver/decoder regions; those block
# boundaries are visible structure at field scale.
SA_PITCH_WL_RANGE = (8, 16)          # sense-amp stripe every N word-line pitches (inclusive)
SA_WIDTH_F_RANGE = (2.0, 4.0)        # stripe width, in units of F
SA_INTENSITY, SA_INTENSITY_JITTER = 0.45, 0.03
SA_SUBLINE_COUNT_RANGE = (1, 2)      # faint internal texture so stripes read as
SA_SUBLINE_WIDTH_F = 0.5             #   circuitry rather than empty voids
SA_SUBLINE_BOOST = 0.10

DR_PITCH_BL_RANGE = (10, 20)         # driver stripe every N bit-line pitches (inclusive)
DR_WIDTH_F_RANGE = (3.0, 5.0)        # stripe width, in units of F
DR_INTENSITY, DR_INTENSITY_JITTER = 0.35, 0.03

# --- contamination particles (SPEC_AMENDMENT_v1.1 §A.3) ---
# CITE: [S10] Contamination particles are a standard artifact in SEM-based
# wafer inspection.
DEFECT_RATE = 2.0                    # Poisson mean count per 10000^2 world
DEFECT_SIGMA_RANGE = (15.0, 40.0)    # world px
DEFECT_AMPLITUDE_RANGE = (0.10, 0.20)  # signed: brighter or darker than local
DEFECT_MIN_CROSSING_DIST = 300.0     # keep particles incidental, not landmarks
DEFECT_MAX_PLACE_TRIES = 64
DEFECT_STAMP_SIGMAS = 4.0            # render radius, in units of sigma

# --- FinFET structure (PROJECT_SPEC.md §3.1.3, secondary / EXPERIMENTAL) ---
FIN_PITCH_F, FIN_WIDTH_F, FIN_INTENSITY = 1.5, 0.5, 0.80
GATE_WIDTH_F, GATE_INTENSITY, GATE_CROSS_INTENSITY = 2.0, 0.90, 0.98
# The spec says "1-2 horizontal gate bars"; that is read as 1-2 bars *per field
# of view* (a 1000 px capture), so the world-level gate pitch is drawn to put
# 1-2 bars in any 1000 px window.  A literal 1-2 bars across the whole 10000 px
# world would leave most crops with a purely periodic fin grating.
FINFET_GATE_PITCH_RANGE = (600.0, 1000.0)

# --- SEM edge-brightening (PROJECT_SPEC.md §3.2) ---
# CITE: [S4] SEM edge effect -- secondary-electron yield rises at feature edges,
# producing bright rims; exploited industrially in CD-SEM metrology.
EDGE_GAIN_REF = 0.25
EDGE_GAIN_SEARCH = 0.20

# --- optics PSF (PROJECT_SPEC.md §3.3) ---
BLUR_SIGMA_REF = 0.8
BLUR_SIGMA_SEARCH_RANGE = (1.2, 2.0)   # search is always blurrier (lower mag)

# --- sensor noise presets: (N_e_ref, N_e_search, b_ref, b_search) §3.4 ---
# CITE: [S1] SEM noise is dominated by signal-dependent Poisson shot noise.
# CITE: [S2] Mixed Poisson-Gaussian model, Var = a*x + b^2.
NOISE_PRESETS: dict[str, dict[str, float]] = {
    "low":    {"N_e_ref": 400.0, "N_e_search": 150.0, "b_ref": 0.010, "b_search": 0.020},
    "medium": {"N_e_ref": 250.0, "N_e_search":  80.0, "b_ref": 0.015, "b_search": 0.030},
    "high":   {"N_e_ref": 150.0, "N_e_search":  40.0, "b_ref": 0.020, "b_search": 0.050},
}

# --- scan-line correlated noise, search image only (PROJECT_SPEC.md §3.4) ---
# CITE: [S3] Row/line-scan correlated noise is present in SEM acquisitions.
SCANLINE_SIGMA = 3.0       # 1-D Gaussian smoothing of the per-row offsets
SCANLINE_AMPLITUDE = 0.015  # scale applied *after* smoothing, per spec

STYLES = ("dram", "finfet")
NOISE_LEVELS = tuple(NOISE_PRESETS)


# --------------------------------------------------------------------------- #
# Structure rendering -- base lattice
# --------------------------------------------------------------------------- #


def _line_profile(
    length: int,
    pitch: float,
    width: float,
    phase: float,
    base_intensity: float,
    jitter: float,
    background: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one axis of a periodic line grating as 1-D profiles.

    Lines are anti-aliased by exact fractional-pixel coverage: pixel ``i``
    spans ``[i, i + 1)`` and its value is the background alpha-blended with the
    line intensity by the overlap length.  This matters because ``F`` (and
    therefore the pitch) is a float -- hard integer rounding would quantise the
    lattice pitch and introduce a spurious, non-physical beat pattern.

    Args:
        length: Profile length in pixels.
        pitch: Line-to-line spacing in pixels.
        width: Line width in pixels.
        phase: Offset of the first line's leading edge.
        base_intensity: Nominal line intensity in [0, 1].
        jitter: Half-width of the uniform per-line intensity jitter.
        background: Intensity between lines.
        rng: Structure RNG stream.

    Returns:
        ``(value, alpha, centers)`` where ``value`` and ``alpha`` have shape
        ``(length,)`` and ``centers`` holds the line centre coordinates.
    """
    if pitch <= 0 or width <= 0:
        raise ValueError(f"pitch and width must be positive, got {pitch}, {width}")

    start0 = phase - np.ceil((phase + width) / pitch) * pitch  # start left of 0
    n_lines = int(np.ceil((length - start0) / pitch)) + 1
    starts = start0 + np.arange(n_lines, dtype=np.float64) * pitch
    ends = starts + width
    intensities = base_intensity + rng.uniform(-jitter, jitter, n_lines)  # process variation

    px0 = np.arange(length, dtype=np.float64)
    overlap = np.clip(
        np.minimum(ends[:, None], px0[None, :] + 1.0) - np.maximum(starts[:, None], px0[None, :]),
        0.0,
        1.0,
    )
    alpha = overlap.max(axis=0).astype(np.float32)
    values = background + (intensities[:, None] - background) * overlap
    value = values.max(axis=0).astype(np.float32)

    centers = starts + width / 2.0
    centers = centers[(centers >= -width) & (centers <= length + width)]
    return value, alpha, centers


def _stamp_max(canvas: np.ndarray, stamp: np.ndarray, cy: int, cx: int) -> None:
    """In-place ``canvas = max(canvas, stamp)`` centred at ``(cy, cx)``, clipped."""
    half = stamp.shape[0] // 2
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    sy0, sx0 = max(0, -y0), max(0, -x0)
    y0c, x0c = max(0, y0), max(0, x0)
    y1c, x1c = min(canvas.shape[0], y1), min(canvas.shape[1], x1)
    if y1c <= y0c or x1c <= x0c:
        return
    sub = stamp[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    np.maximum(canvas[y0c:y1c, x0c:x1c], sub, out=canvas[y0c:y1c, x0c:x1c])


def _via_stamp(radius: float, background: float, intensity: float) -> np.ndarray:
    """Anti-aliased filled disc of ``radius`` px, blended over ``background``."""
    r = int(np.ceil(radius)) + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    dist = np.hypot(xx, yy)
    alpha = np.clip(radius + 0.5 - dist, 0.0, 1.0)  # 1 px soft edge
    return (background + (intensity - background) * alpha).astype(np.float32)


def _apply_global_jitter(world: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Apply small per-pair brightness/contrast jitter in place (§3.1.2).

    Models detector gain/offset drift between acquisitions of different dies.
    Contrast pivots about mid-grey.
    """
    gain = 1.0 + float(rng.uniform(-GLOBAL_CONTRAST_JITTER, GLOBAL_CONTRAST_JITTER))
    bias = float(rng.uniform(-GLOBAL_BRIGHTNESS_JITTER, GLOBAL_BRIGHTNESS_JITTER))
    world -= 0.5
    world *= gain
    world += 0.5 + bias
    np.clip(world, 0.0, 1.0, out=world)
    return gain, bias


# --------------------------------------------------------------------------- #
# Structure rendering -- superstructure (SPEC_AMENDMENT_v1.1 §A)
# --------------------------------------------------------------------------- #


def _stripe_system(
    length: int,
    pitch: float,
    width: float,
    phase: float,
    intensity: float,
    intensity_jitter: float,
    rng: np.random.Generator,
    subline_count_range: tuple[int, int] | None = None,
    subline_width: float = 0.0,
    subline_boost: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a periodic stripe system as ``(alpha, value, centers)`` profiles.

    Unlike the lattice, stripes *replace* what is underneath rather than
    max-compositing with it -- a sense-amp region is different circuitry, not a
    brighter version of the array -- so the caller alpha-blends ``value`` over
    the world using ``alpha``.  Edges are anti-aliased by fractional coverage
    for the same reason as :func:`_line_profile`.

    Args:
        length: Profile length in pixels.
        pitch: Stripe-to-stripe spacing in pixels.
        width: Stripe width in pixels.
        phase: Offset of the first stripe's leading edge.
        intensity: Nominal flat stripe intensity.
        intensity_jitter: Half-width of the uniform per-stripe intensity jitter.
        rng: Superstructure RNG stream.
        subline_count_range: Inclusive ``(lo, hi)`` count of internal sub-lines,
            or ``None`` for a featureless stripe.
        subline_width: Sub-line width in pixels.
        subline_boost: Intensity added along each sub-line.

    Returns:
        ``(alpha, value, centers)``, each 1-D; ``centers`` holds stripe centres.
    """
    if pitch <= 0 or width <= 0:
        raise ValueError(f"pitch and width must be positive, got {pitch}, {width}")

    alpha = np.zeros(length, dtype=np.float32)
    value = np.zeros(length, dtype=np.float32)
    centers: list[float] = []

    k_lo = int(np.floor(-phase / pitch)) - 1
    k_hi = int(np.ceil((length - phase) / pitch)) + 1
    for k in range(k_lo, k_hi + 1):
        start = phase + k * pitch
        end = start + width
        if end <= 0.0 or start >= length:
            continue
        i0 = max(0, int(np.floor(start)))
        i1 = min(length, int(np.ceil(end)) + 1)
        idx = np.arange(i0, i1, dtype=np.float64)
        cov = np.clip(np.minimum(end, idx + 1.0) - np.maximum(start, idx), 0.0, 1.0)

        vals = np.full(idx.size, intensity + float(rng.uniform(-intensity_jitter, intensity_jitter)))
        if subline_count_range is not None and subline_width > 0.0:
            n_sub = int(rng.integers(subline_count_range[0], subline_count_range[1] + 1))
            for j in range(n_sub):
                centre = start + width * (j + 1) / (n_sub + 1)
                s0, s1 = centre - subline_width / 2.0, centre + subline_width / 2.0
                sub_cov = np.clip(np.minimum(s1, idx + 1.0) - np.maximum(s0, idx), 0.0, 1.0)
                vals += subline_boost * sub_cov

        sl = slice(i0, i1)
        np.maximum(alpha[sl], cov.astype(np.float32), out=alpha[sl])
        np.copyto(value[sl], vals.astype(np.float32), where=cov > 0.0)
        centers.append(start + width / 2.0)

    return alpha, value, np.asarray(centers, dtype=np.float64)


def _blend_rows(world: np.ndarray, alpha: np.ndarray, value: np.ndarray) -> None:
    """In-place ``world = world*(1-a) + v*a`` for a row-indexed stripe profile."""
    world *= (1.0 - alpha)[:, None]
    world += (value * alpha)[:, None]


def _blend_cols(world: np.ndarray, alpha: np.ndarray, value: np.ndarray) -> None:
    """In-place ``world = world*(1-a) + v*a`` for a column-indexed stripe profile."""
    world *= (1.0 - alpha)[None, :]
    world += (value * alpha)[None, :]


def apply_superstructure(
    world: np.ndarray, f: float, rng: np.random.Generator
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Overlay sense-amp and wordline-driver stripes in place (v1.1 §A.1-2).

    CITE: [S9] Subarray mats are separated by sense-amplifier stripes
    (horizontal) and wordline-driver / decoder regions (vertical); at a 10x
    field these block boundaries are the structure that makes localization
    well-posed while the cells themselves stay locally periodic.

    Order matters and follows the amendment: horizontal sense-amp stripes are
    laid down first, then vertical driver stripes, which therefore win at
    crossings -- matching a floorplan where the driver column runs through.

    Returns:
        ``(params, sa_centers, dr_centers)``; the centres feed the defect
        keep-out test in :func:`add_defects`.
    """
    sa_pitch_units = int(rng.integers(SA_PITCH_WL_RANGE[0], SA_PITCH_WL_RANGE[1] + 1))
    sa_pitch = sa_pitch_units * WL_PITCH_F * f
    sa_width = float(rng.uniform(*SA_WIDTH_F_RANGE)) * f
    sa_alpha, sa_value, sa_centers = _stripe_system(
        world.shape[0], sa_pitch, sa_width, float(rng.uniform(0.0, sa_pitch)),
        SA_INTENSITY, SA_INTENSITY_JITTER, rng,
        subline_count_range=SA_SUBLINE_COUNT_RANGE,
        subline_width=SA_SUBLINE_WIDTH_F * f,
        subline_boost=SA_SUBLINE_BOOST,
    )
    _blend_rows(world, sa_alpha, sa_value)

    dr_pitch_units = int(rng.integers(DR_PITCH_BL_RANGE[0], DR_PITCH_BL_RANGE[1] + 1))
    dr_pitch = dr_pitch_units * BL_PITCH_F * f
    dr_width = float(rng.uniform(*DR_WIDTH_F_RANGE)) * f
    dr_alpha, dr_value, dr_centers = _stripe_system(
        world.shape[1], dr_pitch, dr_width, float(rng.uniform(0.0, dr_pitch)),
        DR_INTENSITY, DR_INTENSITY_JITTER, rng,
    )
    _blend_cols(world, dr_alpha, dr_value)

    np.clip(world, 0.0, 1.0, out=world)
    params = {
        "sa_pitch_px": round(sa_pitch, 2),
        "sa_width_px": round(sa_width, 2),
        "dr_pitch_px": round(dr_pitch, 2),
        "dr_width_px": round(dr_width, 2),
        "sa_stripes_in_ref": int(np.ceil(CAPTURE_SIZE / sa_pitch)),
        "dr_stripes_in_ref": int(np.ceil(CAPTURE_SIZE / dr_pitch)),
    }
    return params, sa_centers, dr_centers


def add_defects(
    world: np.ndarray,
    rng: np.random.Generator,
    sa_centers: np.ndarray,
    dr_centers: np.ndarray,
) -> list[dict[str, float]]:
    """Sprinkle contamination particles in place (v1.1 §A.3).

    CITE: [S10] Contamination particles are a standard artifact in SEM-based
    wafer inspection.

    The count is Poisson with mean :data:`DEFECT_RATE` over the whole world, so
    most reference crops contain none -- particles are incidental realism, not
    landmarks on demand.  Placement is rejected within
    :data:`DEFECT_MIN_CROSSING_DIST` of a stripe crossing for the same reason.

    Returns:
        One record per placed particle: ``x``, ``y``, ``sigma``, ``amplitude``.
    """
    placed: list[dict[str, float]] = []
    if sa_centers.size == 0 or dr_centers.size == 0:
        return placed

    for _ in range(int(rng.poisson(DEFECT_RATE))):
        for _try in range(DEFECT_MAX_PLACE_TRIES):
            x = float(rng.uniform(0.0, world.shape[1]))
            y = float(rng.uniform(0.0, world.shape[0]))
            dy = float(np.min(np.abs(sa_centers - y)))
            dx = float(np.min(np.abs(dr_centers - x)))
            if np.hypot(dx, dy) >= DEFECT_MIN_CROSSING_DIST:
                break
        else:
            continue  # crowded floorplan: skip this particle rather than force it

        sigma = float(rng.uniform(*DEFECT_SIGMA_RANGE))
        amp = float(rng.uniform(*DEFECT_AMPLITUDE_RANGE)) * (1.0 if rng.random() < 0.5 else -1.0)

        r = int(np.ceil(DEFECT_STAMP_SIGMAS * sigma))
        y0, y1 = max(0, int(y) - r), min(world.shape[0], int(y) + r + 1)
        x0, x1 = max(0, int(x) - r), min(world.shape[1], int(x) + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy = np.arange(y0, y1, dtype=np.float32)[:, None] - y
        xx = np.arange(x0, x1, dtype=np.float32)[None, :] - x
        world[y0:y1, x0:x1] += amp * np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        placed.append({"x": round(x, 1), "y": round(y, 1),
                       "sigma": round(sigma, 2), "amplitude": round(amp, 4)})

    if placed:
        np.clip(world, 0.0, 1.0, out=world)
    return placed


# --------------------------------------------------------------------------- #
# World assembly
# --------------------------------------------------------------------------- #


def render_dram_world(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Render the clean 10000x10000 DRAM lattice (PROJECT_SPEC.md §3.1.2).

    CITE: [S5] 6F^2 open-bitline geometry -- orthogonal WL/BL grid with pitch
    ratios 3F / 2F, line width 1F, and a contact at every intersection.
    Superstructure is applied separately by :func:`apply_superstructure`.
    """
    f = float(rng.uniform(*F_RANGE))

    wl_value, _, wl_centers = _line_profile(
        WORLD_SIZE, WL_PITCH_F * f, WL_WIDTH_F * f, float(rng.uniform(0, WL_PITCH_F * f)),
        WL_INTENSITY, LINE_INTENSITY_JITTER, BACKGROUND, rng,
    )
    bl_value, _, bl_centers = _line_profile(
        WORLD_SIZE, BL_PITCH_F * f, BL_WIDTH_F * f, float(rng.uniform(0, BL_PITCH_F * f)),
        BL_INTENSITY, LINE_INTENSITY_JITTER, BACKGROUND, rng,
    )

    # Brightest-feature-wins compositing: the WL/BL crossing takes the brighter
    # of the two, and vias then sit on top.  One 400 MB float32 allocation.
    world = np.maximum(wl_value[:, None], bl_value[None, :])

    stamp = _via_stamp(VIA_RADIUS_F * f, BACKGROUND, VIA_INTENSITY)
    rows = np.rint(wl_centers).astype(int)
    cols = np.rint(bl_centers).astype(int)
    for ry in rows:
        for cx in cols:
            _stamp_max(world, stamp, int(ry), int(cx))

    gain, bias = _apply_global_jitter(world, rng)
    params: dict[str, Any] = {
        "F": round(f, 4),
        "wl_pitch_px": round(WL_PITCH_F * f, 4),
        "bl_pitch_px": round(BL_PITCH_F * f, 4),
        "n_vias": int(rows.size * cols.size),
        "global_gain": round(gain, 4),
        "global_bias": round(bias, 4),
    }
    return world, params


def render_finfet_world(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Render the clean FinFET lattice (PROJECT_SPEC.md §3.1.3).

    EXPERIMENTAL / secondary style: dense vertical fins crossed by sparse
    horizontal gate bars, with brighter fin-gate crossings.  The same two
    stripe systems are overlaid afterwards (v1.1 §A, FinFET note).
    """
    f = float(rng.uniform(*F_RANGE))
    gate_pitch = float(rng.uniform(*FINFET_GATE_PITCH_RANGE))

    fin_value, fin_alpha, _ = _line_profile(
        WORLD_SIZE, FIN_PITCH_F * f, FIN_WIDTH_F * f, float(rng.uniform(0, FIN_PITCH_F * f)),
        FIN_INTENSITY, LINE_INTENSITY_JITTER, BACKGROUND, rng,
    )
    gate_value, gate_alpha, gate_centers = _line_profile(
        WORLD_SIZE, gate_pitch, GATE_WIDTH_F * f, float(rng.uniform(0, gate_pitch)),
        GATE_INTENSITY, LINE_INTENSITY_JITTER, BACKGROUND, rng,
    )

    world = np.maximum(gate_value[:, None], fin_value[None, :])
    # Fin-gate crossings emit more strongly than either feature alone.
    boost = np.multiply(gate_alpha[:, None], fin_alpha[None, :])
    boost *= (GATE_CROSS_INTENSITY - GATE_INTENSITY)
    world += boost
    del boost
    np.clip(world, 0.0, 1.0, out=world)

    gain, bias = _apply_global_jitter(world, rng)
    params = {
        "F": round(f, 4),
        "fin_pitch_px": round(FIN_PITCH_F * f, 4),
        "gate_pitch_px": round(gate_pitch, 4),
        "n_gate_bars": int(gate_centers.size),
        "global_gain": round(gain, 4),
        "global_bias": round(bias, 4),
    }
    return world, params


def build_world(
    style: str,
    rng_structure: np.random.Generator,
    rng_super: np.random.Generator,
    rng_defects: np.random.Generator,
    pure_lattice: bool = False,
    defects: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the full clean world: lattice, then superstructure, then particles.

    ``pure_lattice=True`` skips both superstructure and particles, reproducing
    the degenerate v1.0 world on demand (v1.1 §A.5).  Because superstructure and
    defects own separate RNG streams, toggling either flag leaves the lattice
    and the stage geometry bit-for-bit identical -- which is what makes the
    §D ablation a controlled comparison.
    """
    if style == "dram":
        world, params = render_dram_world(rng_structure)
    elif style == "finfet":
        world, params = render_finfet_world(rng_structure)
    else:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")

    params["pure_lattice"] = bool(pure_lattice)
    params["defects_enabled"] = bool(defects and not pure_lattice)
    if pure_lattice:
        params["n_defects"] = 0
        params["_defects"] = []
        return world, params

    super_params, sa_centers, dr_centers = apply_superstructure(world, params["F"], rng_super)
    params.update(super_params)

    placed = add_defects(world, rng_defects, sa_centers, dr_centers) if defects else []
    params["n_defects"] = len(placed)
    params["_defects"] = placed
    return world, params


# --------------------------------------------------------------------------- #
# Capture model: emission -> optics -> detector
# --------------------------------------------------------------------------- #


def apply_edge_brightening(clean: np.ndarray, gain: float) -> np.ndarray:
    """Add SEM edge rims to a clean structure image (PROJECT_SPEC.md §3.2).

    CITE: [S4] Secondary-electron yield rises sharply at feature edges and
    protrusions, so real SEM images show bright rims around every edge.  Applied
    to the CLEAN structure, before blur and noise, and independently for each
    capture (the two captures have different resolutions and different gains).
    """
    edges = normalize_unit(sobel_magnitude(clean))
    return np.clip(clean + gain * edges, 0.0, 1.0).astype(np.float32)


def apply_sensor_noise(
    clean: np.ndarray,
    n_e: float,
    read_noise: float,
    rng: np.random.Generator,
    scanline: bool = False,
) -> np.ndarray:
    """Mixed Poisson-Gaussian sensor noise (PROJECT_SPEC.md §3.4).

    CITE: [S1] SEM noise is dominated by Poisson shot noise, which is
    signal-dependent -- not pure additive Gaussian.
    CITE: [S2] Mixed Poisson-Gaussian model: ``y = Poisson(x*N_e)/N_e + N(0, b^2)``,
    giving ``Var = x/N_e + b^2``.
    CITE: [S3] Row-correlated (line-scan) noise, added for the search capture.

    ``rng`` MUST be the stream belonging to this capture: the reference and the
    search image are separate physical acquisitions, so no noise sample may ever
    be shared between them (PROJECT_SPEC.md §3.4, hard requirement).

    The clip ceiling is 1.05 rather than 1.0 so that noise is allowed to
    overshoot the true signal range, as a real detector does; the overshoot is
    preserved by rescaling at 8-bit save time.  Clipping happens once, at the
    end, so the additive terms are not silently biased by an intermediate clamp.
    """
    if n_e <= 0:
        raise ValueError(f"N_e must be positive, got {n_e}")
    counts = rng.poisson(np.clip(clean, 0.0, None) * n_e).astype(np.float32) / n_e  # shot noise
    noisy = counts + rng.normal(0.0, read_noise, clean.shape).astype(np.float32)    # readout noise

    if scanline:
        # 1-D white noise of one value per row, smoothed then scaled, added to
        # every pixel of that row.  Smoothing by sigma=3 reduces the std to
        # ~0.31, so the realised per-row offset std is ~0.0047 (~1.2 DN of 256).
        rows = gaussian_filter1d(
            rng.normal(0.0, 1.0, clean.shape[0]), SCANLINE_SIGMA, mode="nearest"
        ) * SCANLINE_AMPLITUDE
        noisy = noisy + rows[:, None].astype(np.float32)

    return np.clip(noisy, 0.0, PIXEL_CEILING).astype(np.float32)


def sem_capture(
    clean: np.ndarray,
    *,
    edge_gain: float,
    blur_sigma: float,
    n_e: float,
    read_noise: float,
    rng: np.random.Generator,
    scanline: bool = False,
) -> np.ndarray:
    """Full capture chain for one image: emission -> optics PSF -> detector.

    The order matters physically and is fixed by the spec: edge-brightening is
    an emission effect at the sample (§3.2, before noise), Gaussian blur is the
    optical PSF (§3.3, before noise), and Poisson-Gaussian noise is introduced
    by the detector last (§3.4).
    """
    img = apply_edge_brightening(clean, edge_gain)
    img = gaussian_blur(img, blur_sigma)
    return apply_sensor_noise(img, n_e, read_noise, rng, scanline=scanline)


# --------------------------------------------------------------------------- #
# Pair generation
# --------------------------------------------------------------------------- #


def _ground_truth(matrix: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Ground-truth search coordinates: ``p_search = A(cx, cy) / 10`` (§3.1.6)."""
    warped = apply_affine(matrix, [(cx, cy)])[0]
    return float(warped[0] / DOWNSAMPLE), float(warped[1] / DOWNSAMPLE)


def _exact_center_gt(matrix: np.ndarray, x0: int, y0: int) -> tuple[float, float]:
    """Ground truth under strict OpenCV pixel-centre conventions.

    The spec's formula ``A(cx, cy) / 10`` treats the crop centre as the integer
    ``(cx, cy)`` and the decimation as a plain division.  Under OpenCV's
    convention the crop's true centre is ``(x0 + 499.5, y0 + 499.5)`` and
    ``cv2.resize`` maps source ``u`` to ``(u + 0.5)/10 - 0.5``.  The difference
    is a constant ~0.5 px offset.  ``true_x``/``true_y`` stay the spec formula
    (that is the scored definition); this variant is recorded alongside it as
    ``true_x_exact``/``true_y_exact`` so any residual sub-pixel bias observed in
    Phase 3/4 can be attributed rather than tuned away.
    """
    c = (x0 + (CAPTURE_SIZE - 1) / 2.0, y0 + (CAPTURE_SIZE - 1) / 2.0)
    warped = apply_affine(matrix, [c])[0]
    return float((warped[0] + 0.5) / DOWNSAMPLE - 0.5), float((warped[1] + 0.5) / DOWNSAMPLE - 0.5)


def _place_by_drift(
    matrix: np.ndarray,
    rng: np.random.Generator,
    drift_sigma: float,
    drift_cap: float,
) -> tuple[int, int, float, float, int]:
    """Sample the true site by drift prior, then back-solve the crop centre.

    SPEC_AMENDMENT_v1.1 §B.  The tool navigates to the intended site and lands a
    short drift away before capturing, so the true position is sampled *first*
    in search coordinates -- ``g = (500, 500) + d`` with direction uniform and
    magnitude ``|Normal(0, drift_sigma)|`` capped at ``drift_cap`` -- and the
    world crop centre follows as ``A^-1(10 * g)``.

    The crop centre is rounded to an integer pixel (crops must be pixel-aligned)
    and the recorded ground truth is then *recomputed* forward through ``A``, so
    it stays exact by construction rather than inheriting the rounding.  The
    realised drift therefore differs from the sampled one by < 0.05 px.

    Returns:
        ``(cx, cy, true_x, true_y, attempts)``.

    Raises:
        RuntimeError: If no drift sample yields a legal crop within
            :data:`MAX_PLACEMENT_RESAMPLES` attempts.
    """
    inverse = cv2.invertAffineTransform(matrix)
    lo_c, hi_c = CROP_MARGIN, WORLD_SIZE - CROP_MARGIN
    lo_g, hi_g = GT_FRAME_MARGIN, CAPTURE_SIZE - GT_FRAME_MARGIN

    for attempt in range(1, MAX_PLACEMENT_RESAMPLES + 1):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        magnitude = min(abs(float(rng.normal(0.0, drift_sigma))), drift_cap)
        gx = CAPTURE_SIZE / 2.0 + magnitude * np.cos(angle)
        gy = CAPTURE_SIZE / 2.0 + magnitude * np.sin(angle)

        wx, wy = apply_affine(inverse, [(gx * DOWNSAMPLE, gy * DOWNSAMPLE)])[0]
        cx, cy = int(round(float(wx))), int(round(float(wy)))
        true_x, true_y = _ground_truth(matrix, cx, cy)

        if (lo_c <= cx <= hi_c and lo_c <= cy <= hi_c
                and lo_g <= true_x <= hi_g and lo_g <= true_y <= hi_g):
            return cx, cy, true_x, true_y, attempt

    raise RuntimeError(
        f"no drift sample produced a legal crop after {MAX_PLACEMENT_RESAMPLES} attempts "
        f"(drift_sigma={drift_sigma}, drift_cap={drift_cap}); reduce --drift-cap or "
        f"--drift-sigma so the back-solved crop stays {CROP_MARGIN} px inside the world"
    )


def generate_pair(
    index: int,
    style: str,
    noise_level: str,
    master_seed: int,
    pure_lattice: bool = False,
    defects: bool = True,
    drift_sigma: float = DRIFT_SIGMA,
    drift_cap: float = DRIFT_CAP,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one reference/search pair plus its ground-truth record.

    Args:
        index: 1-based pair index (drives the derived seeds).
        style: ``"dram"`` or ``"finfet"``.
        noise_level: Key into :data:`NOISE_PRESETS`.
        master_seed: Dataset-wide seed.
        pure_lattice: Reproduce the degenerate v1.0 world (v1.1 §A.5).
        defects: Sprinkle contamination particles (v1.1 §A.3).
        drift_sigma: Sigma of the drift magnitude, in search px.
        drift_cap: Hard cap on the drift magnitude, in search px.

    Returns:
        ``(reference, search, record)``; both images are float32 in
        ``[0, PIXEL_CEILING]`` with shape ``(1000, 1000)``.
    """
    preset = NOISE_PRESETS[noise_level]
    seeds = derive_seeds(master_seed, index, N_RNG_STREAMS)
    rng_structure, rng_geometry, rng_super, rng_defects, rng_ref, rng_search = (
        make_rng(s) for s in seeds
    )

    # --- stage / magnification error (§3.1.5) -- one draw per pair ---
    theta_deg = float(rng_geometry.uniform(*THETA_RANGE_DEG))
    scale = float(rng_geometry.uniform(*SCALE_RANGE))
    blur_sigma_search = float(rng_geometry.uniform(*BLUR_SIGMA_SEARCH_RANGE))
    matrix = rotation_scale_matrix((WORLD_SIZE / 2.0, WORLD_SIZE / 2.0), theta_deg, scale)

    # --- drift-prior placement, back-solved through A^-1 (v1.1 §B) ---
    cx, cy, true_x, true_y, attempts = _place_by_drift(
        matrix, rng_geometry, drift_sigma, drift_cap
    )

    world, structure_params = build_world(
        style, rng_structure, rng_super, rng_defects,
        pure_lattice=pure_lattice, defects=defects,
    )

    x0, y0 = cx - CAPTURE_SIZE // 2, cy - CAPTURE_SIZE // 2
    ref_clean = world[y0:y0 + CAPTURE_SIZE, x0:x0 + CAPTURE_SIZE].copy()

    # --- search capture: warp the whole world, then decimate x10 (§3.1.5) ---
    # BORDER_REPLICATE, not the default zero fill: rotating/shrinking the world
    # exposes up to ~35 px of frame edge, and a hard black wedge there would be
    # an obviously synthetic cue that a localizer could exploit.
    warped = cv2.warpAffine(
        world, matrix, (WORLD_SIZE, WORLD_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    del world
    search_clean = cv2.resize(
        warped, (CAPTURE_SIZE, CAPTURE_SIZE), interpolation=cv2.INTER_AREA
    )
    del warped

    # --- two independent physical acquisitions (§3.2 - §3.4) ---
    reference = sem_capture(
        ref_clean, edge_gain=EDGE_GAIN_REF, blur_sigma=BLUR_SIGMA_REF,
        n_e=preset["N_e_ref"], read_noise=preset["b_ref"], rng=rng_ref, scanline=False,
    )
    search = sem_capture(
        search_clean, edge_gain=EDGE_GAIN_SEARCH, blur_sigma=blur_sigma_search,
        n_e=preset["N_e_search"], read_noise=preset["b_search"], rng=rng_search, scanline=True,
    )

    placed = structure_params.pop("_defects", [])
    n_in_ref = sum(1 for d in placed
                   if x0 <= d["x"] < x0 + CAPTURE_SIZE and y0 <= d["y"] < y0 + CAPTURE_SIZE)

    exact_x, exact_y = _exact_center_gt(matrix, x0, y0)
    record: dict[str, Any] = {
        "id": f"pair_{index:04d}",
        "index": index,
        "true_x": round(true_x, 4),
        "true_y": round(true_y, 4),
        "true_x_exact": round(exact_x, 4),
        "true_y_exact": round(exact_y, 4),
        "style": style,
        "noise_level": noise_level,
        "theta_deg": round(theta_deg, 5),
        "scale": round(scale, 6),
        "drift_px": round(float(np.hypot(true_x - CAPTURE_SIZE / 2.0,
                                         true_y - CAPTURE_SIZE / 2.0)), 3),
        "drift_sigma": drift_sigma,
        "drift_cap": drift_cap,
        "crop_center_x": cx,
        "crop_center_y": cy,
        "placement_attempts": attempts,
        "blur_sigma_ref": BLUR_SIGMA_REF,
        "blur_sigma_search": round(blur_sigma_search, 4),
        "edge_gain_ref": EDGE_GAIN_REF,
        "edge_gain_search": EDGE_GAIN_SEARCH,
        "N_e_ref": preset["N_e_ref"],
        "N_e_search": preset["N_e_search"],
        "b_ref": preset["b_ref"],
        "b_search": preset["b_search"],
        "expected_template_px": round(CAPTURE_SIZE * scale / DOWNSAMPLE, 2),
        "n_defects_in_ref": n_in_ref,
        "seeds": dict(zip(STREAM_NAMES, seeds)),
        "ref_file": f"pairs/pair_{index:04d}_ref.png",
        "search_file": f"pairs/pair_{index:04d}_search.png",
        **structure_params,
    }
    return reference, search, record


# --------------------------------------------------------------------------- #
# Preview rendering
# --------------------------------------------------------------------------- #


def save_preview(
    path: Path,
    reference: np.ndarray,
    search: np.ndarray,
    record: dict[str, Any],
    zoom_half: int = 70,
) -> None:
    """Save an annotated 3-panel composite for visual QA (PROJECT_SPEC.md §3.5).

    Panels: full reference | search with the GT crosshair, the expected
    ~100x100 footprint and the drift prior (frame centre, drift vector, drift
    cap circle) | a zoom around the GT with the x10-decimated reference inset,
    so the eye can confirm the crosshair really sits on the same structure as
    the template.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display, no GUI backend probing
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    tx, ty = record["true_x"], record["true_y"]
    tmpl = record["expected_template_px"]
    half = CAPTURE_SIZE / 2.0
    ref_small = cv2.resize(reference, (CAPTURE_SIZE // DOWNSAMPLE,) * 2, interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    world_kind = "PURE LATTICE (degenerate control)" if record["pure_lattice"] else "lattice + superstructure"
    fig.suptitle(
        f"{record['id']}  |  {record['style']} / {world_kind}  noise={record['noise_level']}  "
        f"F={record['F']:.1f}px  theta={record['theta_deg']:+.2f}deg  scale={record['scale']:.4f}  "
        f"drift={record['drift_px']:.0f}px  GT=({tx:.2f}, {ty:.2f})",
        fontsize=11,
    )

    axes[0].imshow(reference, cmap="gray", vmin=0, vmax=PIXEL_CEILING)
    axes[0].set_title(f"Reference @100x  1000x1000\nsigma={record['blur_sigma_ref']}  "
                      f"N_e={record['N_e_ref']:.0f}", fontsize=9)

    ax = axes[1]
    ax.imshow(search, cmap="gray", vmin=0, vmax=PIXEL_CEILING)
    ax.set_title(f"Search @10x  1000x1000  (pattern ~{tmpl:.0f}x{tmpl:.0f}px = "
                 f"{100 * tmpl ** 2 / CAPTURE_SIZE ** 2:.1f}% of area)\n"
                 f"sigma={record['blur_sigma_search']:.2f}  N_e={record['N_e_search']:.0f}  "
                 f"+ scan-line noise", fontsize=9)
    # drift prior: frame centre, drift vector, and the cap the drift is drawn under
    ax.add_patch(Circle((half, half), record["drift_cap"], fill=False,
                        ec="#ffaa00", lw=0.8, ls=":"))
    ax.plot([half, tx], [half, ty], color="#ffaa00", lw=0.9, ls="--")
    ax.plot([half], [half], marker="+", color="#ffaa00", ms=9, mew=1.4)
    _crosshair(ax, tx, ty, arm=34, gap=9, color="#00ff66")
    ax.add_patch(Rectangle((tx - tmpl / 2, ty - tmpl / 2), tmpl, tmpl,
                           fill=False, ec="#00ff66", lw=1.0, ls="--"))

    ax = axes[2]
    x0, y0 = int(round(tx)) - zoom_half, int(round(ty)) - zoom_half
    x0 = int(np.clip(x0, 0, CAPTURE_SIZE - 2 * zoom_half))
    y0 = int(np.clip(y0, 0, CAPTURE_SIZE - 2 * zoom_half))
    ax.imshow(search[y0:y0 + 2 * zoom_half, x0:x0 + 2 * zoom_half], cmap="gray",
              vmin=0, vmax=PIXEL_CEILING, extent=(x0, x0 + 2 * zoom_half, y0 + 2 * zoom_half, y0))
    ax.set_title("Search zoom at GT (green)  +  inset: reference decimated x10", fontsize=9)
    _crosshair(ax, tx, ty, arm=18, gap=5, color="#00ff66")

    inset = ax.inset_axes((0.62, 0.62, 0.36, 0.36))
    inset.imshow(ref_small, cmap="gray", vmin=0, vmax=PIXEL_CEILING)
    inset.set_xticks([]); inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("#33aaff")
        spine.set_linewidth(1.4)

    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ensure_dir(path.parent)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _crosshair(ax: Any, x: float, y: float, arm: float, gap: float, color: str) -> None:
    """Draw a gapped crosshair so the structure underneath stays visible."""
    ax.plot([x - arm, x - gap], [y, y], color=color, lw=1.3)
    ax.plot([x + gap, x + arm], [y, y], color=color, lw=1.3)
    ax.plot([x, x], [y - arm, y - gap], color=color, lw=1.3)
    ax.plot([x, x], [y + gap, y + arm], color=color, lw=1.3)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {parsed}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the dataset generator."""
    parser = argparse.ArgumentParser(
        prog="generate_dataset.py",
        description="Generate synthetic SEM wafer reference/search image pairs "
                    "for the Drift-Sense navigation-error-recovery task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python generate_dataset.py --style dram --num-pairs 30 "
               "--output-dir data --seed 42 --noise-level medium --preview",
    )
    parser.add_argument("--style", choices=STYLES, default="dram",
                        help="wafer structure family ('finfet' is experimental)")
    parser.add_argument("--num-pairs", type=_positive_int, default=30,
                        help="number of reference/search pairs to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="destination directory (created if missing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="master seed; per-pair stream seeds are derived from it")
    parser.add_argument("--noise-level", choices=NOISE_LEVELS, default="medium",
                        help="sensor-noise preset (N_e and readout sigma)")
    parser.add_argument("--preview", action="store_true",
                        help="also write annotated composite previews per pair")
    parser.add_argument("--pure-lattice", action="store_true",
                        help="disable superstructure and particles, reproducing the "
                             "degenerate uniform world (the hard/ambiguous control case)")
    parser.add_argument("--defects", action=argparse.BooleanOptionalAction, default=True,
                        help="sprinkle contamination particles (Poisson, ~2 per world)")
    parser.add_argument("--drift-sigma", type=_positive_float, default=DRIFT_SIGMA,
                        help="sigma of the navigation drift magnitude, in search px")
    parser.add_argument("--drift-cap", type=_positive_float, default=DRIFT_CAP,
                        help="hard cap on the navigation drift magnitude, in search px")
    return parser


def _iter_pairs(args: argparse.Namespace) -> Iterator[tuple[int, np.ndarray, np.ndarray, dict[str, Any], float]]:
    for index in range(1, args.num_pairs + 1):
        started = time.perf_counter()
        reference, search, record = generate_pair(
            index, args.style, args.noise_level, args.seed,
            pure_lattice=args.pure_lattice, defects=args.defects,
            drift_sigma=args.drift_sigma, drift_cap=args.drift_cap,
        )
        yield index, reference, search, record, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.drift_cap > CAPTURE_SIZE / 2.0 - GT_FRAME_MARGIN:
        raise SystemExit(
            f"--drift-cap {args.drift_cap} exceeds the usable frame radius "
            f"{CAPTURE_SIZE / 2.0 - GT_FRAME_MARGIN:.0f} px; the ground truth could not "
            f"stay {GT_FRAME_MARGIN} px inside the search image"
        )

    out_dir = ensure_dir(args.output_dir)
    pairs_dir = ensure_dir(out_dir / "pairs")
    preview_dir = ensure_dir(out_dir / "previews") if args.preview else None

    world_kind = "pure lattice (degenerate control)" if args.pure_lattice else \
                 f"lattice + superstructure, defects={'on' if args.defects else 'off'}"
    print(f"Drift-Sense dataset generator ({GENERATOR_VERSION})")
    print(f"  style={args.style}  pairs={args.num_pairs}  noise={args.noise_level}  "
          f"seed={args.seed}  out={out_dir}")
    print(f"  world={WORLD_SIZE}x{WORLD_SIZE}  capture={CAPTURE_SIZE}x{CAPTURE_SIZE}  "
          f"downsample=x{DOWNSAMPLE}")
    print(f"  model={world_kind}")
    print(f"  drift prior: sigma={args.drift_sigma:.0f}px  cap={args.drift_cap:.0f}px")
    print()
    header = (f"{'pair':<11}{'F':>7}{'theta':>9}{'scale':>9}{'GT x':>10}{'GT y':>10}"
              f"{'drift':>8}{'SA px':>9}{'DR px':>9}{'def':>5}{'sec':>7}")
    print(header)
    print("-" * len(header))

    records: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for index, reference, search, record, elapsed in _iter_pairs(args):
        save_gray_png(pairs_dir / f"pair_{index:04d}_ref.png", reference)
        save_gray_png(pairs_dir / f"pair_{index:04d}_search.png", search)
        if preview_dir is not None:
            save_preview(preview_dir / f"pair_{index:04d}_preview.png", reference, search, record)
        records.append(record)
        print(f"{record['id']:<11}{record['F']:>7.1f}{record['theta_deg']:>+9.3f}"
              f"{record['scale']:>9.4f}{record['true_x']:>10.2f}{record['true_y']:>10.2f}"
              f"{record['drift_px']:>8.1f}{record.get('sa_pitch_px', float('nan')):>9.1f}"
              f"{record.get('dr_pitch_px', float('nan')):>9.1f}"
              f"{record.get('n_defects', 0):>5d}{elapsed:>7.2f}")

    total = time.perf_counter() - t_start
    gt_path = out_dir / "ground_truth.json"
    payload = {
        "meta": {
            "generator_version": GENERATOR_VERSION,
            "spec": "PROJECT_SPEC.md §3 + SPEC_AMENDMENT_v1.1 §A/§B",
            "style": args.style,
            "noise_level": args.noise_level,
            "seed": args.seed,
            "num_pairs": args.num_pairs,
            "pure_lattice": args.pure_lattice,
            "defects": args.defects,
            "drift_sigma": args.drift_sigma,
            "drift_cap": args.drift_cap,
            "world_size": WORLD_SIZE,
            "capture_size": CAPTURE_SIZE,
            "downsample": DOWNSAMPLE,
            "pixel_ceiling": PIXEL_CEILING,
            "theta_range_deg": list(THETA_RANGE_DEG),
            "scale_range": list(SCALE_RANGE),
            "noise_preset": NOISE_PRESETS[args.noise_level],
            "rng_streams": list(STREAM_NAMES),
            "gt_convention": "true_x/true_y = A(cx, cy) / 10  (PROJECT_SPEC.md 3.1.6)",
        },
        "pairs": records,
    }
    gt_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("-" * len(header))
    print(f"{args.num_pairs} pairs in {total:.1f}s ({total / args.num_pairs:.2f}s/pair)")
    print(f"  images   -> {pairs_dir}")
    if preview_dir is not None:
        print(f"  previews -> {preview_dir}")
    print(f"  truth    -> {gt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
