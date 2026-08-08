#!/usr/bin/env python3
"""Drift-Sense Phase 1 -- synthetic wafer image-pair generator.

Implements PROJECT_SPEC.md §3 as amended by SPEC_AMENDMENT_v1.1 (§A replaces
§3.1.2, §B replaces §3.1.4) and SPEC_AMENDMENT_v1.2 (§A replaces v1.1 §A.1-2:
aperiodic mat spacing with absolute, coverage-guaranteed pitch bases).  Produces
physically-motivated reference/search image pairs for the Applied Materials PS2
navigation-error-recovery task:

    reference : 1000x1000 crop of the die at 100x-equivalent resolution
    search    : 1000x1000 view of the same die at 10x, i.e. the reference
                pattern appears ~10x smaller (~100x100 px, ~1% of the frame),
                displaced by the tool's stage rotation/magnification error and
                corrupted by an *independent* physical capture.

Pipeline per pair
-----------------
    world (10000x10000 clean truth)
      = DRAM/FinFET lattice
      + sense-amplifier stripes (horizontal, aperiodic)  -- v1.2 §A
      + wordline-driver stripes (vertical, aperiodic)    -- v1.2 §A
      + one bank-boundary stripe per axis (p=0.7)        -- v1.2 §A.4
      + contamination particles                          -- v1.1 §A.3
      |
      |-- crop 1000x1000 at (cx, cy) ---------------------> reference capture
      `-- warpAffine(A) then resize x0.1 (INTER_AREA) ----> search capture

    each capture: SEM edge-brightening -> optics PSF blur -> sensor noise
                  (+ scan-line correlated noise, search only)

Why the superstructure exists, and why it is aperiodic: a globally uniform
lattice is translation-invariant, so the true site carries no more correlation
energy than any other lattice cell and localization is ill-posed (v1.0).  Adding
*regularly* pitched mat stripes only trades lattice ambiguity for mat ambiguity
-- measured under v1.1, false peaks landed on exact integer multiples of the
stripe pitch, ~20 interchangeable mat cells per frame.  v1.2 therefore makes the
mat spacing irregular (``--mat-jitter``) with absolute pitch bases sized so every
reference crop straddles at least one stripe of each family.  The stripe
*spacing sequence* is then a unique code, so a template containing a stripe can
only align one way.

An isolation ablation later showed the load-bearing change was not the spacing
jitter but the move to pitch bases *incommensurate* with the lattice: under v1.1
the stripe pitch was an integer multiple of the lattice pitch, so a one-stripe
shift was also a whole number of lattice periods and the combined pattern was
genuinely shift-invariant.  With incommensurate pitches the joint period becomes
the LCM of the two, which exceeds the frame.  That is the recorded causal story
(CITE: [S12]).

Two controls are kept permanently: ``--pure-lattice`` (no superstructure at all,
the degenerate world and our honest-failure exhibit) and ``--commensurate-mats``
(superstructure quantized back onto lattice multiples, reproducing v1.1's actual
defect).  LINE_INTENSITY_JITTER stays at the spec value of 0.05 throughout -- the
amendments explicitly forbid papering over a modelling gap by inflating a fake
signature.

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

GENERATOR_VERSION = "phase1.5"

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

# --- DRAM superstructure (SPEC_AMENDMENT_v1.2 §A, supersedes v1.1 §A.1-2) ---
# CITE: [S9] DRAM arrays are organised as subarray mats separated by
# sense-amplifier stripes and wordline-driver/decoder regions; those block
# boundaries are visible structure at field scale.
# CITE: [S12] Real floorplans are NOT exactly periodic at field scale --
# redundancy/spare rows and columns, edge mats and bank boundaries break mat
# periodicity.  That aperiodicity is what makes an individual mat cell
# identifiable: under a strictly regular grid the frame holds ~20 interchangeable
# mat cells and the true site is merely one of them (measured under v1.1: false
# peaks landed on exact integer multiples of the stripe pitch, rank median 30).
MAT_JITTER = 0.25            # spacing irregularity; 0.0 == strictly regular mats
MAT_JITTER_UP_FACTOR = 1.2   # asymmetric: step ~ base * U(1 - j, 1 + 1.2*j)

# The two stripe families are ISOTROPIC as of v1.4: identical contrast, width and
# pitch.  Up to v1.3 they differed (SA 0.45 / U(50,110) / U(550,720) against
# DR 0.35 / U(60,140) / U(480,680)), which handed the vertical family roughly 5x
# the discriminating variance of the horizontal one -- contrast against the ~0.85
# array enters squared, so 0.20^2 vs 0.10^2, times width/pitch.  The measured
# consequence was a purely anisotropic failure: x pinned to sub-pixel while y lost
# to whole-word-line-pitch shifts (Phase 2 baseline, 7/10 at 10 px; every miss had
# |dx| < 9 px and |dy| of 243-287 px, with dy/WL landing on exact integers).
# Nothing physical ever justified the asymmetry -- it was an arbitrary choice --
# so v1.4 removes it.  Keep these equal unless a physical argument says otherwise.
#
# Pitch bases are absolute world px (v1.2 §A.2), sized so the largest realised
# *clear* gap stays below the 1000 px reference crop, which guarantees by
# construction that every reference field straddles at least one stripe of each
# family:  700 * (1 + 1.2*0.25) = 910 px step, minus the 55 px minimum width
# -> <= 855 px < 1000 px.  Asserted per world in apply_superstructure().
# Field-scale stripe density is stylized so a reference field spans mat
# boundaries; re-tune these once the official Applied Materials starter code is
# released (4 Aug) -- a one-line change (v1.2 §A.6).
STRIPE_BASE_RANGE = (500.0, 700.0)     # world px between stripes, both families
STRIPE_WIDTH_RANGE_PX = (55.0, 125.0)  # per-stripe width, both families

# --- sparse-landmark tier (SPEC_AMENDMENT_v1.6 §D) --------------------------
# Our default stripe pitch (500-700 world px) is SMALLER than the 1000 px
# reference crop, so every reference is guaranteed to contain at least one
# stripe of each family -- the guarantee asserted below.  That guarantee makes
# the localization problem strictly easier than the official generator's, where
# the array blocks (2600 nm) are 2.6x the reference field (1000 nm) and a crop
# can land entirely inside uniform periodic array with no landmark at all.
#
# MEASURED on the official generator: crops containing no non-array material
# scored 18.8% within 5 px against 89.6% for crops with >20% coverage, and they
# are 16% of its pairs.  That single split accounts for essentially all of the
# gap between our 94% on our own data and 70% on theirs -- so until this tier
# exists we cannot measure progress on the case that dominates the error.
#
# The range brackets the official period (2600 nm mat + 320 nm strip = 2920 nm).
#
# MEASURED, 40 dram/medium pairs at seed 20260808:
#   landmark-free crops      38%  (official generator: 16%)
#   acc@5 on those           13.3%  (official generator: 18.8%)  <- the mode reproduces
#   acc@5 on landmark crops  32.0%  (official generator: 89.6%)  <- ours is harsher
# The failure mode itself transfers closely; the landmark-bearing half does not,
# because our landmarks are 1-D stripes that pin one axis, where the official
# strips are 2-D regions with orthogonal routing lines that pin both. This tier
# is therefore a deliberate stress case, not a calibrated replica -- it is sized
# for statistical power on the failing half, which is the half we need to fix.
SPARSE_STRIPE_BASE_RANGE = (2400.0, 3400.0)
STRIPE_INTENSITY, STRIPE_INTENSITY_JITTER = 0.35, 0.03

SA_BASE_RANGE = DR_BASE_RANGE = STRIPE_BASE_RANGE
SA_WIDTH_RANGE_PX = DR_WIDTH_RANGE_PX = STRIPE_WIDTH_RANGE_PX
SA_INTENSITY, SA_INTENSITY_JITTER = STRIPE_INTENSITY, STRIPE_INTENSITY_JITTER
DR_INTENSITY, DR_INTENSITY_JITTER = STRIPE_INTENSITY, STRIPE_INTENSITY_JITTER

# Internal sub-line texture, so a stripe reads as circuitry rather than a void.
# Width scales with the (now much thinner) stripe; a narrow band only has room
# for a single sub-line.
SA_SUBLINE_COUNT_RANGE = (1, 2)
SA_SUBLINE_WIDTH_FRAC = 0.18         # of the stripe width
SA_SUBLINE_SINGLE_BELOW_PX = 80.0    # below this width, exactly one sub-line
SA_SUBLINE_BOOST = 0.10

# Bank boundary (v1.2 §A.4): one extra-wide, extra-dark stripe per axis.  A bank
# edge is a far larger break than a mat edge, so it is a strong landmark.
BANK_PROB = 0.7
BANK_WIDTH_RANGE_PX = (200.0, 350.0)
BANK_INTENSITY, BANK_INTENSITY_JITTER = 0.30, 0.03

# --- contamination particles (SPEC_AMENDMENT_v1.1 §A.3) ---
# CITE: [S10] Contamination particles are a standard artifact in SEM-based
# wafer inspection.
DEFECT_RATE = 2.0                    # Poisson mean count per 10000^2 world
DEFECT_SIGMA_RANGE = (15.0, 40.0)    # world px
DEFECT_AMPLITUDE_RANGE = (0.10, 0.20)  # signed: brighter or darker than local
DEFECT_MIN_CROSSING_DIST = 300.0     # keep particles incidental, not landmarks
DEFECT_MAX_PLACE_TRIES = 64
DEFECT_STAMP_SIGMAS = 4.0            # render radius, in units of sigma

# --- FinFET structure (PROJECT_SPEC.md §3.1.3 + SPEC_AMENDMENT_v1.5) --------
# CITE: [S13] Standard-cell logic is organised into rows of fixed height,
# separated by row-boundary bands (power rails / n-well edges), with diffusion
# breaks between cells and dummy gates at cell edges.  The contacted poly pitch
# (CPP) is regular by construction; the rows, breaks and dummies are what vary.
FIN_PITCH_F, FIN_WIDTH_F, FIN_INTENSITY = 1.5, 0.5, 0.80
GATE_WIDTH_F, GATE_INTENSITY, GATE_CROSS_INTENSITY = 2.0, 0.90, 0.98

# Why v1.5 exists: a fin field is a 1-D grating.  Fins repeat in x and are
# perfectly uniform in y, so between gate bars there is nothing at all to fix y.
# Phase 4 measured the consequence -- FinFET trailed DRAM by ~10 points at every
# noise level, every failure was pure-y, and the correlation surface degenerated
# into horizontal ridges spanning the full frame (dy landing on integer multiples
# of the gate pitch).  That is the pure-lattice degeneracy marginalised onto one
# axis.  The fix is the y-structure real logic actually has: cell rows.
#
# Gate pitch stays REGULAR -- CPP is regular in real logic, and pretending
# otherwise would be inventing physics to make the maths easier.  Tightened from
# v1.4's U(600,1000) so a 1000 px reference crop is guaranteed to contain a gate
# bar with margin rather than by a hair (max clear gap 750 - 2F = 590..670 px).
FINFET_GATE_PITCH_RANGE = (450.0, 750.0)

#: Standard-cell row boundaries: horizontal bands at semi-regular pitch.  These
#: carry the aperiodic y-signature the fin field cannot.
ROW_BASE_RANGE = (400.0, 620.0)      # world px between cell-row boundaries
ROW_JITTER = 0.18                    # semi-regular: rows vary, gates do not
ROW_WIDTH_RANGE_PX = (30.0, 70.0)
ROW_INTENSITY, ROW_INTENSITY_JITTER = 0.55, 0.04

#: Diffusion breaks: irregular horizontal cuts where the active region ends and
#: the fins are interrupted.
DIFF_BREAK_RATE = 26.0               # Poisson mean count across the world
DIFF_BREAK_WIDTH_RANGE_PX = (20.0, 55.0)
DIFF_BREAK_INTENSITY = 0.22

#: Dummy-gate doublets: a second bar placed one half-pitch off an existing gate.
#: Occasional, so the gate sequence carries a code without disturbing the CPP.
DUMMY_GATE_PROB = 0.30
DUMMY_GATE_OFFSET_FRAC = 0.34        # of the gate pitch

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
    # v1.6 §E: the organiser's starter generator uses the SAME Poisson-Gaussian
    # form as ours, just in 0-255 units -- its `dose` is exactly our N_e and its
    # `detector_noise_sigma/255` is exactly our b. Transcribing its defaults
    # (dose_reference 2000, dose_search 200, sigma 2 and 5 of 255) shows our own
    # presets run 5-13x noisier on the REFERENCE than the official nominal, and
    # that our "low" search tier is already noisier than its default. Included so
    # that claim is reproducible and so we can train against the official
    # operating point rather than only against harsher ones.
    "official": {"N_e_ref": 2000.0, "N_e_search": 200.0,
                 "b_ref": 2.0 / 255.0, "b_search": 5.0 / 255.0},
}

# --- optional noise kinds the official generator enables in its own evaluation ---
# CITE: [S2] Multiplicative (signal-proportional) gain variation and impulse
# (dead/hot pixel) noise are standard detector artifacts distinct from the
# Poisson-Gaussian pair above.
# The official baseline_solution/evaluate.py turns speckle on at its "high" tier
# (0.15) and speckle + salt-pepper at "severe" (0.3 / 0.01), so these are part of
# the distribution it actually scores against -- not merely demo knobs. Default
# 0.0 keeps every existing seed bit-identical.
SPECKLE_SIGMA = 0.0        # multiplicative: out = x * (1 + N(0, sigma))
SALT_PEPPER_PROB = 0.0     # fraction of pixels forced to 0 or the ceiling

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


def _paint_stripe(
    alpha: np.ndarray,
    value: np.ndarray,
    start: float,
    width: float,
    intensity: float,
    subline: tuple[int, float, float] | None = None,
) -> bool:
    """Paint one anti-aliased stripe into ``(alpha, value)``; True if visible.

    ``subline`` is ``(count, width_px, boost)``.  Sub-lines are spaced evenly
    inside the stripe so it reads as circuitry rather than an empty void.
    """
    length = alpha.size
    end = start + width
    if end <= 0.0 or start >= length:
        return False
    i0 = max(0, int(np.floor(start)))
    i1 = min(length, int(np.ceil(end)) + 1)
    idx = np.arange(i0, i1, dtype=np.float64)
    cov = np.clip(np.minimum(end, idx + 1.0) - np.maximum(start, idx), 0.0, 1.0)

    vals = np.full(idx.size, intensity, dtype=np.float64)
    if subline is not None:
        count, sub_width, boost = subline
        for j in range(count):
            centre = start + width * (j + 1) / (count + 1)
            s0, s1 = centre - sub_width / 2.0, centre + sub_width / 2.0
            vals += boost * np.clip(np.minimum(s1, idx + 1.0) - np.maximum(s0, idx), 0.0, 1.0)

    sl = slice(i0, i1)
    np.maximum(alpha[sl], cov.astype(np.float32), out=alpha[sl])
    np.copyto(value[sl], vals.astype(np.float32), where=cov > 0.0)
    return True


def _stripe_starts(
    length: int, base_pitch: float, mat_jitter: float, rng: np.random.Generator
) -> list[float]:
    """Irregular stripe positions, generated sequentially (v1.2 §A.1).

    ``pos[k+1] = pos[k] + base_pitch * U(1 - j, 1 + 1.2*j)``.  The asymmetric
    upper factor keeps the mean spacing slightly above ``base_pitch`` while the
    *maximum* step stays bounded, which is what the coverage guarantee rests on.
    ``mat_jitter == 0`` degenerates to exactly regular spacing (the v1.1 model),
    retained as the middle tier of the three-way ablation.

    The sequence starts two maximal pitches left of the origin with a random
    phase, so the left edge is covered by the same irregular process as the
    interior rather than by a special case.
    """
    if base_pitch <= 0:
        raise ValueError(f"base_pitch must be positive, got {base_pitch}")
    if mat_jitter < 0:
        raise ValueError(f"mat_jitter must be >= 0, got {mat_jitter}")
    lo = 1.0 - mat_jitter
    hi = 1.0 + MAT_JITTER_UP_FACTOR * mat_jitter

    pos = -2.0 * base_pitch * hi + float(rng.uniform(0.0, base_pitch))
    starts: list[float] = []
    while pos < length:
        starts.append(pos)
        pos += base_pitch * float(rng.uniform(lo, hi))
    return starts


def _max_clear_gap(alpha: np.ndarray) -> int:
    """Longest run of uncovered pixels in a stripe alpha profile."""
    covered = np.flatnonzero(alpha > 0.0)
    if covered.size == 0:
        return int(alpha.size)
    interior = int(np.max(np.diff(covered) - 1)) if covered.size > 1 else 0
    return int(max(interior, covered[0], alpha.size - 1 - covered[-1]))


def _stripe_system(
    length: int,
    base_pitch: float,
    mat_jitter: float,
    width_range: tuple[float, float],
    intensity: float,
    intensity_jitter: float,
    rng: np.random.Generator,
    *,
    per_stripe_width: bool = True,
    subline_count_range: tuple[int, int] | None = None,
    subline_width_frac: float = 0.0,
    subline_single_below: float = 0.0,
    subline_boost: float = 0.0,
    bank_prob: float = 0.0,
    bank_width_range: tuple[float, float] | None = None,
    bank_intensity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Render one aperiodic stripe family as ``(alpha, value, centers, has_bank)``.

    Unlike the lattice, stripes *replace* what is underneath rather than
    max-compositing with it -- a sense-amp region is different circuitry, not a
    brighter version of the array -- so the caller alpha-blends ``value`` over
    the world using ``alpha``.  Edges are anti-aliased by fractional coverage
    for the same reason as :func:`_line_profile`.

    Per v1.2 §A.3 the width and intensity are drawn *per stripe*, so the stripe
    sequence carries a unique code rather than being a repeating unit.  Setting
    ``per_stripe_width=False`` draws one width for the whole family instead,
    which is part of what ``--commensurate-mats`` needs to reconstruct the v1.1
    defect faithfully (v1.3 §A.2).

    Args:
        length: Profile length in pixels.
        base_pitch: Nominal stripe-to-stripe spacing in pixels.
        mat_jitter: Spacing irregularity (0 = strictly regular).
        width_range: ``(lo, hi)`` stripe width in pixels.
        intensity: Nominal flat stripe intensity.
        intensity_jitter: Half-width of the uniform per-stripe intensity jitter.
        rng: Superstructure RNG stream.
        per_stripe_width: Draw the width once per stripe (True) or once per
            world (False).
        subline_count_range: Inclusive ``(lo, hi)`` internal sub-line count, or
            ``None`` for featureless stripes.
        subline_width_frac: Sub-line width as a fraction of the stripe width.
        subline_single_below: Stripes narrower than this get exactly one
            sub-line, whatever ``subline_count_range`` says.
        subline_boost: Intensity added along each sub-line.
        bank_prob: Probability of adding one extra-wide bank-boundary stripe.
        bank_width_range: ``(lo, hi)`` bank stripe width in pixels.
        bank_intensity: Nominal bank stripe intensity.

    Returns:
        ``(alpha, value, centers, has_bank)``.
    """
    alpha = np.zeros(length, dtype=np.float32)
    value = np.zeros(length, dtype=np.float32)
    centers: list[float] = []

    world_width = None if per_stripe_width else float(rng.uniform(*width_range))
    for start in _stripe_starts(length, base_pitch, mat_jitter, rng):
        width = float(rng.uniform(*width_range)) if world_width is None else world_width
        inten = intensity + float(rng.uniform(-intensity_jitter, intensity_jitter))
        subline = None
        if subline_count_range is not None and subline_width_frac > 0.0:
            count = (1 if width < subline_single_below
                     else int(rng.integers(subline_count_range[0], subline_count_range[1] + 1)))
            subline = (count, width * subline_width_frac, subline_boost)
        if _paint_stripe(alpha, value, start, width, inten, subline):
            centers.append(start + width / 2.0)

    has_bank = False
    if bank_width_range is not None and float(rng.random()) < bank_prob:
        bank_width = float(rng.uniform(*bank_width_range))
        bank_start = float(rng.uniform(0.0, max(1.0, length - bank_width)))
        bank_value = bank_intensity + float(
            rng.uniform(-BANK_INTENSITY_JITTER, BANK_INTENSITY_JITTER)
        )
        if _paint_stripe(alpha, value, bank_start, bank_width, bank_value, None):
            centers.append(bank_start + bank_width / 2.0)
            has_bank = True

    return alpha, value, np.asarray(sorted(centers), dtype=np.float64), has_bank


def _blend_rows(world: np.ndarray, alpha: np.ndarray, value: np.ndarray) -> None:
    """In-place ``world = world*(1-a) + v*a`` for a row-indexed stripe profile."""
    world *= (1.0 - alpha)[:, None]
    world += (value * alpha)[:, None]


def _blend_cols(world: np.ndarray, alpha: np.ndarray, value: np.ndarray) -> None:
    """In-place ``world = world*(1-a) + v*a`` for a column-indexed stripe profile."""
    world *= (1.0 - alpha)[None, :]
    world += (value * alpha)[None, :]


def apply_superstructure(
    world: np.ndarray,
    f: float,
    rng: np.random.Generator,
    mat_jitter: float = MAT_JITTER,
    commensurate: bool = False,
    sparse_landmarks: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Overlay sense-amp and wordline-driver stripes in place (v1.2 §A).

    CITE: [S9] Subarray mats are separated by sense-amplifier stripes
    (horizontal) and wordline-driver / decoder regions (vertical); at a 10x
    field these block boundaries are the structure that makes localization
    well-posed while the cells themselves stay locally periodic.
    CITE: [S12] The spacing is irregular because real floorplans break exact mat
    periodicity (redundancy/spare rows, edge mats, bank boundaries).  A strictly
    regular grid would merely trade lattice ambiguity for mat ambiguity.

    Order matters and follows the amendment: horizontal sense-amp stripes are
    laid down first, then vertical driver stripes, which therefore win at
    crossings -- matching a floorplan where the driver column runs through.

    ``sparse_landmarks=True`` (v1.6 §D) widens the stripe pitch past the capture
    size so a reference crop can contain no superstructure at all, mirroring the
    official generator's block geometry. The §A.2 guarantee is deliberately not
    asserted in that mode -- violating it is the entire point of the tier.

    Raises:
        RuntimeError: If the realised clear gap of either family exceeds the
            capture size, which would break the v1.2 §A.2 guarantee that every
            reference crop straddles at least one stripe of each family.
            Not raised when ``sparse_landmarks`` is set.

    Returns:
        ``(params, sa_centers, dr_centers, sa_alpha, dr_alpha)``.  The centres
        feed the defect keep-out test; the alpha profiles let the caller measure
        per-crop stripe coverage.
    """
    # v1.3 §A: the commensurate ablation mode quantizes each base onto an exact
    # multiple of the corresponding lattice pitch, forces regular spacing, drops
    # the bank landmarks and fixes the width per world -- reconstructing v1.1's
    # real defect, where a one-stripe shift was also a whole number of lattice
    # periods and the combined pattern was therefore genuinely shift-invariant.
    if commensurate:
        mat_jitter = 0.0
        bank_prob = 0.0
        per_stripe_width = False
    else:
        bank_prob = BANK_PROB
        per_stripe_width = True

    base_range = SPARSE_STRIPE_BASE_RANGE if sparse_landmarks else SA_BASE_RANGE
    sa_base = float(rng.uniform(*base_range))
    dr_base = float(rng.uniform(*(SPARSE_STRIPE_BASE_RANGE if sparse_landmarks
                                  else DR_BASE_RANGE)))
    if commensurate:
        sa_base = max(1.0, round(sa_base / (WL_PITCH_F * f))) * (WL_PITCH_F * f)
        dr_base = max(1.0, round(dr_base / (BL_PITCH_F * f))) * (BL_PITCH_F * f)

    sa_alpha, sa_value, sa_centers, sa_bank = _stripe_system(
        world.shape[0], sa_base, mat_jitter, SA_WIDTH_RANGE_PX,
        SA_INTENSITY, SA_INTENSITY_JITTER, rng,
        per_stripe_width=per_stripe_width,
        subline_count_range=SA_SUBLINE_COUNT_RANGE,
        subline_width_frac=SA_SUBLINE_WIDTH_FRAC,
        subline_single_below=SA_SUBLINE_SINGLE_BELOW_PX,
        subline_boost=SA_SUBLINE_BOOST,
        bank_prob=bank_prob,
        bank_width_range=BANK_WIDTH_RANGE_PX,
        bank_intensity=BANK_INTENSITY,
    )
    _blend_rows(world, sa_alpha, sa_value)

    dr_alpha, dr_value, dr_centers, dr_bank = _stripe_system(
        world.shape[1], dr_base, mat_jitter, DR_WIDTH_RANGE_PX,
        DR_INTENSITY, DR_INTENSITY_JITTER, rng,
        per_stripe_width=per_stripe_width,
        bank_prob=bank_prob,
        bank_width_range=BANK_WIDTH_RANGE_PX,
        bank_intensity=BANK_INTENSITY,
    )
    _blend_cols(world, dr_alpha, dr_value)

    np.clip(world, 0.0, 1.0, out=world)

    sa_gap, dr_gap = _max_clear_gap(sa_alpha), _max_clear_gap(dr_alpha)
    if not sparse_landmarks and max(sa_gap, dr_gap) >= CAPTURE_SIZE:
        raise RuntimeError(
            f"stripe coverage guarantee violated (v1.2 §A.2): largest clear gap "
            f"SA={sa_gap}px DR={dr_gap}px >= capture size {CAPTURE_SIZE}px; a reference "
            f"crop could contain no superstructure. Lower SA_BASE_RANGE/DR_BASE_RANGE "
            f"or MAT_JITTER."
        )

    params = {
        "mat_jitter": mat_jitter,
        "commensurate_mats": bool(commensurate),
        "sparse_landmarks": bool(sparse_landmarks),
        "sa_base_px": round(sa_base, 2),
        "dr_base_px": round(dr_base, 2),
        "sa_base_over_wl_pitch": round(sa_base / (WL_PITCH_F * f), 4),
        "dr_base_over_bl_pitch": round(dr_base / (BL_PITCH_F * f), 4),
        "sa_stripes": int(sa_centers.size),
        "dr_stripes": int(dr_centers.size),
        "sa_bank": bool(sa_bank),
        "dr_bank": bool(dr_bank),
        "sa_max_clear_gap_px": sa_gap,
        "dr_max_clear_gap_px": dr_gap,
    }
    return params, sa_centers, dr_centers, sa_alpha, dr_alpha


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

    # Dummy-gate doublets: an occasional second bar offset from a real gate.
    # The CPP stays regular; what varies is which gates carry a dummy, and that
    # sequence is a per-world code (v1.5, CITE [S13]).
    n_dummy = 0
    for centre in gate_centers:
        if float(rng.random()) >= DUMMY_GATE_PROB:
            continue
        start = centre + DUMMY_GATE_OFFSET_FRAC * gate_pitch - GATE_WIDTH_F * f / 2.0
        if _paint_stripe(gate_alpha, gate_value, start, GATE_WIDTH_F * f,
                         GATE_INTENSITY + float(rng.uniform(-LINE_INTENSITY_JITTER,
                                                            LINE_INTENSITY_JITTER)), None):
            n_dummy += 1

    world = np.maximum(gate_value[:, None], fin_value[None, :])
    # Fin-gate crossings emit more strongly than either feature alone.
    boost = np.multiply(gate_alpha[:, None], fin_alpha[None, :])
    boost *= (GATE_CROSS_INTENSITY - GATE_INTENSITY)
    world += boost
    del boost
    np.clip(world, 0.0, 1.0, out=world)

    # Standard-cell row boundaries (v1.5): horizontal bands at semi-regular
    # pitch.  This is the y-structure the fin grating cannot provide, and it is
    # what real logic actually looks like at field scale.
    row_alpha, row_value, row_centers, _ = _stripe_system(
        WORLD_SIZE, float(rng.uniform(*ROW_BASE_RANGE)), ROW_JITTER,
        ROW_WIDTH_RANGE_PX, ROW_INTENSITY, ROW_INTENSITY_JITTER, rng,
    )
    _blend_rows(world, row_alpha, row_value)

    # Diffusion breaks: irregular horizontal cuts through the active region.
    n_breaks = 0
    for _ in range(int(rng.poisson(DIFF_BREAK_RATE))):
        width = float(rng.uniform(*DIFF_BREAK_WIDTH_RANGE_PX))
        start = float(rng.uniform(-width, WORLD_SIZE))
        brk_alpha = np.zeros(WORLD_SIZE, np.float32)
        brk_value = np.zeros(WORLD_SIZE, np.float32)
        if _paint_stripe(brk_alpha, brk_value, start, width, DIFF_BREAK_INTENSITY, None):
            _blend_rows(world, brk_alpha, brk_value)
            n_breaks += 1
    np.clip(world, 0.0, 1.0, out=world)

    row_gap = _max_clear_gap(row_alpha)
    gate_gap = _max_clear_gap(gate_alpha)
    if max(row_gap, gate_gap) >= CAPTURE_SIZE:
        raise RuntimeError(
            f"FinFET coverage guarantee violated (v1.5): largest clear gap "
            f"rows={row_gap}px gates={gate_gap}px >= capture size {CAPTURE_SIZE}px; a "
            f"reference crop could contain no row boundary or no gate bar. Lower "
            f"ROW_BASE_RANGE or FINFET_GATE_PITCH_RANGE."
        )

    gain, bias = _apply_global_jitter(world, rng)
    params = {
        "F": round(f, 4),
        "fin_pitch_px": round(FIN_PITCH_F * f, 4),
        "gate_pitch_px": round(gate_pitch, 4),
        "n_gate_bars": int(gate_centers.size),
        "n_dummy_gates": n_dummy,
        "n_row_boundaries": int(row_centers.size),
        "n_diffusion_breaks": n_breaks,
        "row_max_clear_gap_px": row_gap,
        "gate_max_clear_gap_px": gate_gap,
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
    mat_jitter: float = MAT_JITTER,
    commensurate: bool = False,
    sparse_landmarks: bool = False,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Build the full clean world: lattice, then superstructure, then particles.

    Four-tier ablation (v1.3 §A, extended by v1.6 §D):
      ``pure_lattice=True``      -- no superstructure at all, the degenerate v1.0 world
      ``commensurate=True``      -- superstructure pitched on exact lattice multiples,
                                    reproducing v1.1's real defect
      ``sparse_landmarks=True``  -- superstructure pitched WIDER than the capture,
                                    so some crops contain no landmark (official-like)
      default                    -- aperiodic, incommensurate superstructure (v1.2/v1.3)

    ``mat_jitter`` remains available as an independent knob, but it is no longer
    the ablation's middle tier: the isolation ablation showed spacing jitter was
    not the load-bearing change -- commensurability was.

    Because superstructure and defects own separate RNG streams, toggling either
    flag leaves the lattice and the stage geometry bit-for-bit identical --
    which is what makes the gate a controlled comparison rather than a redraw.

    Returns:
        ``(world, params, profiles)``; ``profiles`` carries the stripe alpha
        profiles (empty for ``pure_lattice``) so the caller can measure per-crop
        coverage.
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
        return world, params, {}

    super_params, sa_centers, dr_centers, sa_alpha, dr_alpha = apply_superstructure(
        world, params["F"], rng_super, mat_jitter, commensurate,
        sparse_landmarks=sparse_landmarks,
    )
    params.update(super_params)

    placed = add_defects(world, rng_defects, sa_centers, dr_centers) if defects else []
    params["n_defects"] = len(placed)
    params["_defects"] = placed
    return world, params, {"sa_alpha": sa_alpha, "dr_alpha": dr_alpha}


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
    speckle_sigma: float = SPECKLE_SIGMA,
    salt_pepper_prob: float = SALT_PEPPER_PROB,
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

    # v1.6 §E, both no-ops at their 0.0 defaults so existing seeds are unchanged.
    if speckle_sigma > 0:
        noisy = noisy * (1.0 + rng.normal(0.0, speckle_sigma, clean.shape).astype(np.float32))
    if salt_pepper_prob > 0:
        hit = rng.random(clean.shape) < salt_pepper_prob
        noisy = np.where(hit & (rng.random(clean.shape) < 0.5), PIXEL_CEILING,
                         np.where(hit, 0.0, noisy)).astype(np.float32)

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
    speckle_sigma: float = SPECKLE_SIGMA,
    salt_pepper_prob: float = SALT_PEPPER_PROB,
) -> np.ndarray:
    """Full capture chain for one image: emission -> optics PSF -> detector.

    The order matters physically and is fixed by the spec: edge-brightening is
    an emission effect at the sample (§3.2, before noise), Gaussian blur is the
    optical PSF (§3.3, before noise), and Poisson-Gaussian noise is introduced
    by the detector last (§3.4).
    """
    img = apply_edge_brightening(clean, edge_gain)
    img = gaussian_blur(img, blur_sigma)
    return apply_sensor_noise(img, n_e, read_noise, rng, scanline=scanline,
                              speckle_sigma=speckle_sigma,
                              salt_pepper_prob=salt_pepper_prob)


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
    uniform: bool = False,
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
        if uniform:
            # v1.6 §F control: no drift prior at all, the target lands anywhere
            # in the legal frame -- which is what the official generator does.
            gx = float(rng.uniform(lo_g, hi_g))
            gy = float(rng.uniform(lo_g, hi_g))
        else:
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
    mat_jitter: float = MAT_JITTER,
    commensurate: bool = False,
    sparse_landmarks: bool = False,
    uniform_placement: bool = False,
    speckle_sigma: float = SPECKLE_SIGMA,
    salt_pepper_prob: float = SALT_PEPPER_PROB,
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
        mat_jitter: Mat-spacing irregularity; 0 = strictly regular (v1.2 §A.1).
        commensurate: Quantize stripe pitches onto lattice multiples, the v1.1
            defect reproduced for ablation (v1.3 §A). Not a realistic mode.
        sparse_landmarks: Widen the stripe pitch past the capture size so some
            reference crops contain no superstructure, matching the official
            generator's block geometry (v1.6 §D).
        uniform_placement: Place the target uniformly across the search frame
            instead of near the centre via the drift prior (v1.6 §F). The
            official generator samples crop origins uniformly, so this is the
            control that shows how much our results lean on the drift prior.
            Note this changes only where the TARGET IS PLACED; the official
            closest-to-centre TIE-BREAK is a problem-statement requirement and
            stays in localize.py either way.
        speckle_sigma: Multiplicative noise sigma on the search capture (v1.6 §E).
        salt_pepper_prob: Impulse-noise pixel fraction on the search capture.

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
        matrix, rng_geometry, drift_sigma, drift_cap, uniform=uniform_placement
    )

    world, structure_params, profiles = build_world(
        style, rng_structure, rng_super, rng_defects,
        pure_lattice=pure_lattice, defects=defects, mat_jitter=mat_jitter,
        commensurate=commensurate, sparse_landmarks=sparse_landmarks,
    )

    x0, y0 = cx - CAPTURE_SIZE // 2, cy - CAPTURE_SIZE // 2
    ref_clean = world[y0:y0 + CAPTURE_SIZE, x0:x0 + CAPTURE_SIZE].copy()

    # --- v1.2 §A.2 per-pair coverage assertion + logging ---
    coverage: dict[str, Any] = {}
    if profiles:
        sa_a = profiles["sa_alpha"][y0:y0 + CAPTURE_SIZE]
        dr_a = profiles["dr_alpha"][x0:x0 + CAPTURE_SIZE]
        sa_cov, dr_cov = float(sa_a.mean()), float(dr_a.mean())
        # v1.6 §D: the sparse tier exists precisely to produce landmark-free
        # crops, so the §A.2 guarantee is recorded there rather than enforced.
        if not sparse_landmarks and (sa_cov <= 0.0 or dr_cov <= 0.0):
            raise RuntimeError(
                f"pair {index}: reference crop at ({cx}, {cy}) contains no "
                f"{'sense-amp' if sa_cov <= 0 else 'driver'} stripe (SA {sa_cov:.4f}, "
                f"DR {dr_cov:.4f}); v1.2 §A.2 guarantees >=1 stripe of each family"
            )
        # Union coverage is separable: 1 - mean(1-sa_row) * mean(1-dr_col).
        coverage = {
            "sa_coverage_ref_pct": round(100.0 * sa_cov, 2),
            "dr_coverage_ref_pct": round(100.0 * dr_cov, 2),
            "superstructure_coverage_ref_pct": round(
                100.0 * (1.0 - float((1.0 - sa_a).mean()) * float((1.0 - dr_a).mean())), 2
            ),
        }

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
        speckle_sigma=speckle_sigma, salt_pepper_prob=salt_pepper_prob,
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
        "uniform_placement": bool(uniform_placement),
        "speckle_sigma": speckle_sigma,
        "salt_pepper_prob": salt_pepper_prob,
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
        **coverage,
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
    if record["pure_lattice"]:
        world_kind = "PURE LATTICE (degenerate control)"
    elif record.get("commensurate_mats"):
        world_kind = "COMMENSURATE mats (v1.1 defect control)"
    else:
        world_kind = "lattice + aperiodic superstructure"
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


def _nonneg_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
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
    ablation = parser.add_mutually_exclusive_group()
    ablation.add_argument("--pure-lattice", action="store_true",
                          help="disable superstructure and particles, reproducing the "
                               "degenerate uniform world (the hard/ambiguous control case)")
    ablation.add_argument("--sparse-landmarks", action="store_true",
                          help="widen stripe pitch past the capture size so some reference "
                               "crops contain no landmark, as in the official generator "
                               "(v1.6 §D); relaxes the v1.2 §A.2 coverage guarantee")
    ablation.add_argument("--commensurate-mats", action="store_true",
                          help="reproduces the v1.1 commensurate-superstructure defect for "
                               "ablation; not a realistic mode")
    parser.add_argument("--defects", action=argparse.BooleanOptionalAction, default=True,
                        help="sprinkle contamination particles (Poisson, ~2 per world)")
    parser.add_argument("--drift-sigma", type=_positive_float, default=DRIFT_SIGMA,
                        help="sigma of the navigation drift magnitude, in search px")
    parser.add_argument("--drift-cap", type=_positive_float, default=DRIFT_CAP,
                        help="hard cap on the navigation drift magnitude, in search px")
    parser.add_argument("--uniform-placement", action="store_true",
                        help="place the target uniformly in the frame instead of via the "
                             "drift prior (v1.6 §F); the official generator does this")
    parser.add_argument("--speckle-sigma", type=_nonneg_float, default=SPECKLE_SIGMA,
                        help="multiplicative noise sigma on the search capture (v1.6 §E)")
    parser.add_argument("--salt-pepper-prob", type=_nonneg_float, default=SALT_PEPPER_PROB,
                        help="impulse-noise pixel fraction on the search capture (v1.6 §E)")
    parser.add_argument("--mat-jitter", type=_nonneg_float, default=MAT_JITTER,
                        help="mat-spacing irregularity; 0 = strictly regular mats "
                             "(the v1.1 model, kept as the middle ablation tier)")
    return parser


def _iter_pairs(args: argparse.Namespace) -> Iterator[tuple[int, np.ndarray, np.ndarray, dict[str, Any], float]]:
    for index in range(1, args.num_pairs + 1):
        started = time.perf_counter()
        reference, search, record = generate_pair(
            index, args.style, args.noise_level, args.seed,
            pure_lattice=args.pure_lattice, defects=args.defects,
            drift_sigma=args.drift_sigma, drift_cap=args.drift_cap,
            mat_jitter=args.mat_jitter, commensurate=args.commensurate_mats,
            sparse_landmarks=args.sparse_landmarks,
            uniform_placement=args.uniform_placement,
            speckle_sigma=args.speckle_sigma, salt_pepper_prob=args.salt_pepper_prob,
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

    if args.pure_lattice:
        world_kind = "pure lattice (degenerate control, NOT realistic)"
    elif args.commensurate_mats:
        world_kind = "lattice + COMMENSURATE superstructure (v1.1 defect control, NOT realistic)"
    else:
        mats = "aperiodic mats" if args.mat_jitter > 0 else "regular mats"
        world_kind = (f"lattice + superstructure, {mats} (jitter={args.mat_jitter:g}), "
                      f"defects={'on' if args.defects else 'off'}")
    print(f"Drift-Sense dataset generator ({GENERATOR_VERSION})")
    print(f"  style={args.style}  pairs={args.num_pairs}  noise={args.noise_level}  "
          f"seed={args.seed}  out={out_dir}")
    print(f"  world={WORLD_SIZE}x{WORLD_SIZE}  capture={CAPTURE_SIZE}x{CAPTURE_SIZE}  "
          f"downsample=x{DOWNSAMPLE}")
    print(f"  model={world_kind}")
    print(f"  drift prior: sigma={args.drift_sigma:.0f}px  cap={args.drift_cap:.0f}px")
    print()
    header = (f"{'pair':<11}{'F':>7}{'theta':>9}{'scale':>9}{'GT x':>9}{'GT y':>9}"
              f"{'drift':>7}{'SA base':>9}{'DR base':>9}{'cov%':>7}{'bank':>6}{'def':>5}{'sec':>7}")
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
        bank = "".join(c for c, k in (("S", "sa_bank"), ("D", "dr_bank")) if record.get(k)) or "-"
        print(f"{record['id']:<11}{record['F']:>7.1f}{record['theta_deg']:>+9.3f}"
              f"{record['scale']:>9.4f}{record['true_x']:>9.2f}{record['true_y']:>9.2f}"
              f"{record['drift_px']:>7.1f}{record.get('sa_base_px', float('nan')):>9.1f}"
              f"{record.get('dr_base_px', float('nan')):>9.1f}"
              f"{record.get('superstructure_coverage_ref_pct', float('nan')):>7.1f}"
              f"{bank:>6}{record.get('n_defects', 0):>5d}{elapsed:>7.2f}")

    total = time.perf_counter() - t_start
    gt_path = out_dir / "ground_truth.json"
    payload = {
        "meta": {
            "generator_version": GENERATOR_VERSION,
            "spec": ("docs/PROJECT_SPEC.md §3 + SPEC_AMENDMENT v1.1 §B + v1.2 §A "
                     "+ v1.3 §A/§B + v1.4 + v1.5 §A"),
            "style": args.style,
            "noise_level": args.noise_level,
            "seed": args.seed,
            "num_pairs": args.num_pairs,
            "pure_lattice": args.pure_lattice,
            "commensurate_mats": args.commensurate_mats,
            "defects": args.defects,
            "mat_jitter": args.mat_jitter,
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
