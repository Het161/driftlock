#!/usr/bin/env python3
"""Drift-Sense / DriftLock -- localize a reference pattern inside a search image.

THIS IS THE SCORED SCRIPT (PROJECT_SPEC.md §4).  Applied Materials run it as-is on
their own hidden test pairs, so the contract is deliberately rigid:

    stdout is exactly one line: "x y"   (rounded integers, space separated)

Nothing else ever reaches stdout.  Every diagnostic, warning and error goes to
stderr, so a caller can parse stdout unconditionally.

Usage
-----
    python localize.py --reference ref.png --search search.png
    python localize.py ref.png search.png              # positional, same thing
    python localize.py -r ref.png -s search.png --json out.json --debug dbg.png

Pipeline (PROJECT_SPEC.md §4 baseline, §5 robustness upgrades)
--------------------------------------------------------------
    1. load both as grayscale float32, normalize to zero mean / unit std
    2. light Gaussian denoise on the SEARCH image only (sigma 1.0)
    3. sweep 7 scales (9.6-10.4) x 5 rotations (+/-2 deg), building a template
       from the reference for each and keeping the best correlation surface
    4. collect every local maximum within 0.02 of the winner
    5. ambiguity gate: PSR and rival count decide whether the field is
       degenerate; if so, apply the official closest-to-centre tie-break
    6. parabolic sub-pixel refinement on the chosen peak -> print

``--fast`` drops the rotation sweep (7 matches instead of 35) for benchmarking.
Passing ``sweep=False`` to :func:`localize` reproduces the Phase 2 single-scale
baseline, which is what ``ablation_gate.py`` measures the data against.

Measured on 30 unseen medium-noise pairs: 90% within 5 px at 657 ms/pair, against
83% at 23 ms for the single-scale baseline.  Every threshold in this file was
calibrated on seed 20260810 and evaluated on seeds it had never seen; the two
constants that carry a MEASURED note were set against holdout evidence that
contradicted the calibration set, and both records are kept deliberately.

Coordinate convention
---------------------
Matches the generator's ground truth exactly: ``p_search = A(cx, cy) / 10``
(PROJECT_SPEC.md §3.1.6).  A correlation-map peak at index ``(j, i)`` puts the
template centre at ``(j + w/2, i + h/2)`` in that convention -- verified against
the generator at a median error of 0.42 px on clean pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from common import ensure_dir, load_gray_float, zscore

# --------------------------------------------------------------------------- #
# Tunable constants (PROJECT_SPEC.md §9: named and documented, never inlined)
# --------------------------------------------------------------------------- #

#: Magnification ratio between the reference (100x) and the search image (10x).
#: The reference is decimated by exactly this factor to build the template.
DOWNSAMPLE = 10

#: Light denoise applied to the SEARCH image only -- it is the noisier capture
#: (lower magnification, fewer electrons per pixel).  Blurring the reference too
#: would throw away the template detail the match depends on.
SEARCH_DENOISE_SIGMA = 1.0

#: Correlation metric.  TM_CCOEFF_NORMED is invariant to affine intensity
#: changes, which matters because the two captures have independent gain,
#: brightness and noise (PROJECT_SPEC.md §3.4).
MATCH_METHOD = cv2.TM_CCOEFF_NORMED

#: Minimum usable template edge, in pixels.  Below this the correlation surface
#: is too small to be meaningful and the caller almost certainly passed the
#: wrong image.
MIN_TEMPLATE_PX = 8

# --- Phase 3: multi-scale x multi-rotation sweep (PROJECT_SPEC.md §5.1) ------
# CITE: [S6] NCC degrades under small rotation/scale change, so a single-scale
# match is fragile against the tool's magnification jitter and stage rotation.
# The ranges mirror the physical error the generator injects: scale 0.97-1.03
# about a nominal 10x, rotation +/-2 degrees.
SCALE_SWEEP = (9.6, 10.4, 7)          # (lo, hi, steps) -- reference is resized by 1/s
ROTATION_SWEEP_DEG = (-2.0, 2.0, 5)   # (lo, hi, steps) -- the template is rotated

#: Fraction trimmed from each template edge after rotation, to keep the
#: undefined corners of a rotated square out of the correlation.
#:
#: MEASURED TO ZERO.  Trimming is a net loss: on 40 calibration pairs (seed
#: 20260810, medium noise) accuracy within 5 px ran 37 / 35 / 32 / 32 at trims of
#: 0.00 / 0.02 / 0.05 / 0.08.  At 2 degrees the corner triangles are only ~1.7%
#: of the linear extent and BORDER_REFLECT_101 fills them with plausible
#: texture, whereas a 5% trim discards 19% of the template area -- and that area
#: carries the aperiodic superstructure the match depends on.  Kept as a named
#: knob so the finding is reproducible, not deleted.
ROTATION_CROP_FRAC = 0.0

# --- Phase 3: periodic-ambiguity resolution (PROJECT_SPEC.md §5.2) -----------
#: A candidate peak must score within this of the best to be considered a rival.
CANDIDATE_SCORE_TOL = 0.02
#: Minimum separation between candidate peaks, as a fraction of template size.
CANDIDATE_MIN_SEP_FRAC = 0.5

# --- Phase 3: confidence (PROJECT_SPEC.md §5.4) ------------------------------
#: Half-width of the window excluded from the sidelobe statistics.
PSR_WINDOW = 11

# --- Phase 3: ambiguity gate ------------------------------------------------
# Calibrated on seed 20260810 only -- never on an acceptance seed.
#
# Three signals were measured; two are used.
#   peak NCC      REJECTED in Phase 2: hit scores spanned 0.705-0.913 and miss
#                 scores 0.800-0.879 over 40 pairs -- fully overlapping.
#   score gap     REJECTED here: standard fields p10 = 0.001 against pure-lattice
#                 max = 0.002. It does not separate. Still computed and reported
#                 in --json as a diagnostic.
#   PSR           SEPARATES: standard min 2.655 vs pure-lattice max 2.151.
#   n_candidates  SEPARATES hardest: standard median 3 (max 10) vs pure-lattice
#                 median 243. This is the count-based generalisation of the gap
#                 -- how many rivals sit within CANDIDATE_SCORE_TOL, rather than
#                 how close the single best one is.
# Thresholds sit in the empty band between the two populations, giving a 0%
# trigger rate on standard fields and 100% on --pure-lattice.
#
# HONEST LIMIT: these separate *populations*, not individual failures. Within
# standard fields, missed pairs had a median PSR of 3.73 against 3.71 for hits
# -- no separation. The gate can say "this whole field is degenerate"; it cannot
# say "this particular answer is wrong".
AMBIGUITY_PSR_MIN = 2.5
AMBIGUITY_CANDIDATES_MAX = 20

#: Axis-resolved ambiguity (Phase 4.5).  Degeneracy is often confined to ONE
#: axis: a FinFET fin field is a 1-D grating, so rival peaks stack up in a
#: vertical line -- same x, scattered y -- and the correlation surface shows
#: horizontal ridges.  Collapsing that to a single boolean throws away the fact
#: that x was determined perfectly.
#:
#: An axis counts as degenerate when the rival candidates span more than this
#: fraction of the SEARCH FRAME along it -- "the rivals are spread across the
#: whole image", not merely "there is more than one".  The frame, not the
#: template, is the right yardstick: two candidates a template-width apart is a
#: normal near-tie, whereas candidates strewn across the frame means the image
#: genuinely does not determine that coordinate.
#:
#: Calibrated on seed 20260910 to separate populations, exactly as the Phase 3
#: gate was.  Accuracy is FLAT across every value tried (0.25 to 8.0 template
#: widths moved dram 19-21/24 and finfet 19-21/24 -- pure noise), so this
#: threshold is chosen to make the *flag* informative, not to buy accuracy.
COLLINEAR_TOL_FRAME_FRAC = 0.60

#: Whether the official closest-to-centre tie-break fires only when the gate
#: above trips, or on every near-tie as PROJECT_SPEC.md §5.2 states literally.
#:
#: MEASURED True, but only after the calibration set was contradicted by holdout.
#: On 40 calibration pairs (seed 20260810) the ungated rule looked clearly better
#: -- 37/40 against 32/40 -- which would have made this False.  On 90 fresh pairs
#: across three unseen seeds the ordering reversed and held in every one:
#:      seed 20260820   gated 27/30   ungated 24/30
#:      seed 20260822   gated 29/30   ungated 26/30
#:      seed 20260823   gated 27/30   ungated 27/30
#:      pooled          gated 92.2%   ungated 85.6%   (-6.7% +/- 9.1%, not significant)
#: The difference is inside the noise band, so the honest reading is that the
#: policy barely matters; gated is chosen because it never lost on holdout and
#: because it is what the drift-prior physics argues for.  The single-seed
#: calibration result was overfitting -- recorded here so it is not rediscovered.
#:
#: Note the gate's trigger rate on standard fields is 0%, so in practice this
#: setting makes the centre rule a no-op there; its real work is on degenerate
#: fields, and the ambiguity flag's real value is as the §5.4 confidence report.
CENTRE_RULE_ONLY_WHEN_AMBIGUOUS = True

# --- Phase 3: sub-pixel refinement (PROJECT_SPEC.md §5.3) -------------------
# CITE: [S7] Parabolic interpolation of the correlation peak recovers the
# fractional offset.  A quadratic fit is only valid inside one sample of the
# peak, so the correction is clamped.
SUBPIXEL_MAX_OFFSET = 1.0


@dataclass
class LocalizationResult:
    """Outcome of one localization, in search-image pixel coordinates."""

    x: float
    y: float
    confidence: float          # peak TM_CCOEFF_NORMED score, in [-1, 1]
    psr: float | None          # peak-to-sidelobe ratio (§5.4)
    ambiguous: bool | None     # ambiguity gate tripped on either axis (§5.2)
    template_size: int
    search_size: tuple[int, int]
    runtime_ms: float
    method: str = "baseline-ncc-single-scale"
    ambig_x: bool = False               # rival peaks disagree about x
    ambig_y: bool = False               # rival peaks disagree about y
    scale: float | None = None          # best scale from the sweep
    theta_deg: float | None = None      # best rotation from the sweep
    peak_gap: float | None = None       # winner minus best separated rival
    n_candidates: int = 1               # rivals within CANDIDATE_SCORE_TOL
    centre_rule_applied: bool = False   # official tie-break actually fired
    subpixel_dx: float = 0.0
    subpixel_dy: float = 0.0

    @property
    def x_int(self) -> int:
        return int(round(self.x))

    @property
    def y_int(self) -> int:
        return int(round(self.y))


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


def _template_bank(
    ref_n: np.ndarray, scales: np.ndarray, rotations: np.ndarray, crop_frac: float
) -> list[tuple[float, float, np.ndarray]]:
    """Build every (scale, rotation, template) combination for the sweep.

    The reference is resized once per scale and then rotated cheaply at template
    resolution, rather than rotating the full-resolution reference each time.
    Every template is trimmed symmetrically by ``crop_frac`` so that rotation
    corners never enter the correlation and all scores remain comparable.
    """
    bank: list[tuple[float, float, np.ndarray]] = []
    for scale in scales:
        th = max(1, int(round(ref_n.shape[0] / scale)))
        tw = max(1, int(round(ref_n.shape[1] / scale)))
        base = cv2.resize(ref_n, (tw, th), interpolation=cv2.INTER_AREA)
        margin = int(round(min(th, tw) * crop_frac))
        for theta in rotations:
            if abs(theta) < 1e-9:
                rotated = base
            else:
                matrix = cv2.getRotationMatrix2D(((tw - 1) / 2.0, (th - 1) / 2.0),
                                                 float(theta), 1.0)
                rotated = cv2.warpAffine(base, matrix, (tw, th), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT_101)
            cropped = rotated[margin:th - margin, margin:tw - margin] if margin else rotated
            bank.append((float(scale), float(theta), np.ascontiguousarray(cropped)))
    return bank


def _candidate_peaks(corr: np.ndarray, tol: float, min_sep: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local maxima scoring within ``tol`` of the best, sorted best-first (§5.2).

    Local maxima are found by dilate-and-compare, which is the cheap way to ask
    "is this pixel the largest in its neighbourhood".
    """
    k = max(3, int(min_sep) | 1)
    dilated = cv2.dilate(corr, np.ones((k, k), np.uint8))
    best = float(corr.max())
    mask = (corr >= dilated - 1e-9) & (corr >= best - tol)
    ys, xs = np.nonzero(mask)
    scores = corr[ys, xs]
    order = np.argsort(-scores)
    return xs[order], ys[order], scores[order]


def _peak_gap(corr: np.ndarray, px: int, py: int, min_sep: int) -> float:
    """Score gap between the winner and the best peak ``min_sep`` away from it.

    A small gap means a rival explains the image nearly as well -- the signature
    of periodic ambiguity, and unlike the absolute score it is comparable across
    images of different contrast.
    """
    suppressed = corr.copy()
    y0, y1 = max(0, py - min_sep), min(corr.shape[0], py + min_sep + 1)
    x0, x1 = max(0, px - min_sep), min(corr.shape[1], px + min_sep + 1)
    suppressed[y0:y1, x0:x1] = -np.inf
    rival = float(suppressed.max())
    if not np.isfinite(rival):
        return float("inf")     # nothing else in the frame: unambiguous
    return float(corr[py, px]) - rival


def _psr(corr: np.ndarray, px: int, py: int, window: int = PSR_WINDOW) -> float:
    """Peak-to-sidelobe ratio: ``(peak - mean(rest)) / std(rest)`` (§5.4)."""
    half = window // 2
    mask = np.ones(corr.shape, dtype=bool)
    mask[max(0, py - half):py + half + 1, max(0, px - half):px + half + 1] = False
    sidelobe = corr[mask]
    if sidelobe.size == 0:
        return float("inf")
    std = float(sidelobe.std())
    return float((corr[py, px] - float(sidelobe.mean())) / (std + 1e-12))


def _subpixel_offset(corr: np.ndarray, px: int, py: int) -> tuple[float, float]:
    """Parabolic fit to the 3x3 correlation neighbourhood (§5.3).

    CITE: [S7] Standard sub-pixel peak interpolation: fit a quadratic through
    the three samples along each axis and take its vertex.
    """
    if not (0 < px < corr.shape[1] - 1 and 0 < py < corr.shape[0] - 1):
        return 0.0, 0.0
    c = corr[py - 1:py + 2, px - 1:px + 2].astype(np.float64)

    def vertex(lo: float, mid: float, hi: float) -> float:
        denom = lo - 2.0 * mid + hi
        if abs(denom) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (lo - hi) / denom, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET))

    return vertex(c[1, 0], c[1, 1], c[1, 2]), vertex(c[0, 1], c[1, 1], c[2, 1])


def localize(
    reference: np.ndarray,
    search: np.ndarray,
    *,
    downsample: int = DOWNSAMPLE,
    denoise_sigma: float = SEARCH_DENOISE_SIGMA,
    fast: bool = False,
    sweep: bool = True,
    return_corr: bool = False,
) -> LocalizationResult | tuple[LocalizationResult, np.ndarray]:
    """Locate ``reference`` inside ``search`` and return its centre.

    Public API: ``evaluate.py`` imports this directly rather than shelling out,
    so it must not print, exit, or touch the filesystem.

    Args:
        reference: High-magnification reference, float32 in [0, 1].
        search: Low-magnification search image, float32 in [0, 1].
        downsample: Magnification ratio between the two captures.
        denoise_sigma: Gaussian sigma applied to the search image only.
        fast: Skip the rotation sweep (scales only) -- the ``--fast`` benchmark
            path from §5.5.
        sweep: Run the Phase 3 sweep at all. ``False`` reproduces the Phase 2
            single-scale baseline, which is what the ablation gate measures.
        return_corr: Also return the correlation map (for ``--debug``).

    Returns:
        A :class:`LocalizationResult`, or ``(result, corr_map)`` when
        ``return_corr`` is set.

    Raises:
        ValueError: If either image is not 2-D, or the derived template is
            unusably small or larger than the search image.
    """
    started = time.perf_counter()

    if reference.ndim != 2 or search.ndim != 2:
        raise ValueError(
            f"expected 2-D grayscale images, got reference {reference.shape} "
            f"and search {search.shape}"
        )
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {downsample}")

    # 1. normalize both to zero mean / unit std (§4.1)
    ref_n = zscore(reference)
    search_n = zscore(search)

    # 2. light denoise, search image only (§4.2)
    if denoise_sigma > 0:
        search_n = cv2.GaussianBlur(search_n, (0, 0), sigmaX=float(denoise_sigma),
                                    sigmaY=float(denoise_sigma),
                                    borderType=cv2.BORDER_REPLICATE)

    # 3. decimate the reference by exactly `downsample` (§4.3)
    th = max(1, int(round(ref_n.shape[0] / downsample)))
    tw = max(1, int(round(ref_n.shape[1] / downsample)))
    if min(th, tw) < MIN_TEMPLATE_PX:
        raise ValueError(
            f"reference {reference.shape} decimated by {downsample} gives a {th}x{tw} "
            f"template, below the {MIN_TEMPLATE_PX}px minimum -- is the reference image "
            f"correct, or does this pair use a different magnification ratio?"
        )
    if th > search_n.shape[0] or tw > search_n.shape[1]:
        raise ValueError(
            f"template {th}x{tw} is larger than the search image "
            f"{search_n.shape[0]}x{search_n.shape[1]}; the two image arguments may be "
            f"swapped, or the magnification ratio is not {downsample}x"
        )
    # 4. sweep scale x rotation, keeping the best correlation surface (§5.1)
    if sweep:
        scales = np.linspace(*SCALE_SWEEP)
        rotations = (np.array([0.0]) if fast else np.linspace(*ROTATION_SWEEP_DEG))
        method = "sweep-scale-only" if fast else "sweep-scale-rotation"
    else:
        scales = np.array([float(downsample)])
        rotations = np.array([0.0])
        method = "baseline-ncc-single-scale"

    best_score, best = -np.inf, None
    for scale, theta, template in _template_bank(ref_n, scales, rotations, ROTATION_CROP_FRAC):
        if template.shape[0] > search_n.shape[0] or template.shape[1] > search_n.shape[1]:
            continue
        corr = cv2.matchTemplate(search_n, template, MATCH_METHOD)
        score = float(corr.max())
        if score > best_score:
            best_score, best = score, (scale, theta, template.shape, corr)
    if best is None:
        raise ValueError("no template in the sweep fitted inside the search image")
    scale, theta, (tpl_h, tpl_w), corr = best

    # 5. candidate peaks, then the ambiguity gate (§5.2)
    min_sep = max(1, int(round(min(tpl_h, tpl_w) * CANDIDATE_MIN_SEP_FRAC)))
    xs, ys, scores = _candidate_peaks(corr, CANDIDATE_SCORE_TOL, min_sep)
    px, py = int(xs[0]), int(ys[0])

    psr = _psr(corr, px, py)
    gap = _peak_gap(corr, px, py, min_sep)   # diagnostic only; does not separate

    # Axis-resolved: how far do the rival candidates actually disagree, per axis?
    cx_c = xs + tpl_w / 2.0
    cy_c = ys + tpl_h / 2.0
    tol_x = COLLINEAR_TOL_FRAME_FRAC * search_n.shape[1]
    tol_y = COLLINEAR_TOL_FRAME_FRAC * search_n.shape[0]
    spread_x = float(cx_c.max() - cx_c.min()) if xs.size > 1 else 0.0
    spread_y = float(cy_c.max() - cy_c.min()) if xs.size > 1 else 0.0
    # A globally degenerate field (very low PSR, or a swarm of rivals) is
    # ambiguous on both axes regardless of how the candidates happen to line up.
    degenerate = bool(psr < AMBIGUITY_PSR_MIN or xs.size > AMBIGUITY_CANDIDATES_MAX)
    ambig_x = bool(degenerate or spread_x > tol_x)
    ambig_y = bool(degenerate or spread_y > tol_y)
    ambiguous = bool(ambig_x or ambig_y)

    # 6. the official tie-break, applied PER DEGENERATE AXIS only.  Scoring
    #    candidates by distance to the frame centre along the ambiguous axes
    #    alone preserves whichever axis the image already determined -- the
    #    FinFET ridge case knows x to a fraction of a pixel and only needs help
    #    with y.  Firing it on every loose tie measured net-negative in Phase 2.
    centre_applied = False
    if (ambiguous or not CENTRE_RULE_ONLY_WHEN_AMBIGUOUS) and xs.size > 1:
        centre = (search_n.shape[1] / 2.0, search_n.shape[0] / 2.0)
        cost = np.zeros(xs.size, dtype=np.float64)
        if ambig_x:
            cost += (cx_c - centre[0]) ** 2
        if ambig_y:
            cost += (cy_c - centre[1]) ** 2
        pick = int(np.argmin(cost))
        if pick != 0:
            px, py = int(xs[pick]), int(ys[pick])
            centre_applied = True

    # 7. parabolic sub-pixel refinement on the chosen peak (§5.3)
    dx, dy = _subpixel_offset(corr, px, py)

    # 8. template top-left -> centre, in the generator's GT convention
    #    p_search = A(cx, cy) / 10 (§3.1.6).  The rotation trim is symmetric, so
    #    the cropped template's centre is still the reference's centre.
    result = LocalizationResult(
        x=float(px + dx + tpl_w / 2.0),
        y=float(py + dy + tpl_h / 2.0),
        confidence=float(corr[py, px]),
        psr=float(psr),
        ambiguous=ambiguous,
        ambig_x=ambig_x,
        ambig_y=ambig_y,
        template_size=int(tpl_w),
        search_size=(int(search.shape[1]), int(search.shape[0])),
        runtime_ms=(time.perf_counter() - started) * 1000.0,
        method=method,
        scale=float(scale),
        theta_deg=float(theta),
        peak_gap=float(gap) if np.isfinite(gap) else None,
        n_candidates=int(xs.size),
        centre_rule_applied=centre_applied,
        subpixel_dx=float(dx),
        subpixel_dy=float(dy),
    )
    return (result, corr) if return_corr else result


# --------------------------------------------------------------------------- #
# Optional outputs
# --------------------------------------------------------------------------- #


def _crosshair(canvas: np.ndarray, x: float, y: float, color: tuple[int, int, int],
               arm: int = 26, gap: int = 7) -> None:
    """Draw a gapped crosshair so the structure underneath stays visible."""
    xi, yi = int(round(x)), int(round(y))
    for dx0, dy0, dx1, dy1 in ((-arm, 0, -gap, 0), (gap, 0, arm, 0),
                               (0, -arm, 0, -gap), (0, gap, 0, arm)):
        cv2.line(canvas, (xi + dx0, yi + dy0), (xi + dx1, yi + dy1), color, 1, cv2.LINE_AA)


def save_debug_image(path: Path, search: np.ndarray, corr: np.ndarray,
                     result: LocalizationResult) -> None:
    """Write a side-by-side debug view: search + crosshair | correlation heatmap.

    Uses OpenCV only -- this is the scored script's hot path and pulling in a
    plotting stack here would cost more startup time than the match itself.
    """
    h, w = search.shape
    half = result.template_size // 2

    left = cv2.cvtColor((np.clip(search, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(left,
                  (int(result.x) - half, int(result.y) - half),
                  (int(result.x) + half, int(result.y) + half), (0, 255, 100), 1)
    _crosshair(left, result.x, result.y, (0, 255, 100))

    # Embed the correlation map at its true offset so it stays spatially aligned
    # with the search image rather than being stretched to fit.
    field = np.zeros((h, w), np.float32)
    ch, cw = corr.shape
    y0, x0 = half, half
    field[y0:y0 + ch, x0:x0 + cw] = corr
    norm = cv2.normalize(field, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    right = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    _crosshair(right, result.x, result.y, (255, 255, 255))

    for img, text in ((left, f"predicted ({result.x_int}, {result.y_int})"),
                      (right, f"NCC peak {result.confidence:.3f}")):
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(path.parent)
    if not cv2.imwrite(str(path), cv2.hconcat([left, right])):
        raise IOError(f"failed to write debug image {path}")


def result_to_json(result: LocalizationResult, reference: Path, search: Path) -> dict:
    """Build the ``--json`` payload (PROJECT_SPEC.md §4 CLI contract)."""
    payload = asdict(result)
    payload.update({
        "x_int": result.x_int,
        "y_int": result.y_int,
        "reference": str(reference),
        "search": str(search),
        "phase": 3,
    })
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Both named and positional forms are accepted because the exact invocation
    used by the graders is unknown (PROJECT_SPEC.md §4).
    """
    parser = argparse.ArgumentParser(
        prog="localize.py",
        description="Locate a high-magnification reference pattern inside a "
                    "low-magnification search image. Prints 'x y' to stdout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="stdout is exactly one line: 'x y'. All diagnostics go to stderr.",
    )
    parser.add_argument("paths", nargs="*", metavar="REFERENCE SEARCH",
                        help="reference and search image paths, in that order "
                             "(alternative to --reference/--search)")
    parser.add_argument("-r", "--reference", type=Path, help="reference image (100x)")
    parser.add_argument("-s", "--search", type=Path, help="search image (10x)")
    parser.add_argument("--json", type=Path, metavar="OUT.json",
                        help="also write coordinates, confidence and timing as JSON")
    parser.add_argument("--debug", type=Path, metavar="OUT.png",
                        help="also write a heatmap + predicted crosshair overlay")
    parser.add_argument("--downsample", type=int, default=DOWNSAMPLE,
                        help="magnification ratio between reference and search")
    parser.add_argument("--fast", action="store_true",
                        help="skip the rotation sweep (scales only), for speed benchmarking")
    return parser


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Reconcile the named and positional forms into one (reference, search) pair."""
    positional = list(args.paths)
    reference, search = args.reference, args.search

    if reference is None and positional:
        reference = Path(positional.pop(0))
    if search is None and positional:
        search = Path(positional.pop(0))
    if positional:
        raise SystemExit(f"error: unexpected extra argument(s): {' '.join(positional)}")
    if reference is None or search is None:
        raise SystemExit(
            "error: need both a reference and a search image.\n"
            "  usage: localize.py --reference REF.png --search SEARCH.png\n"
            "     or: localize.py REF.png SEARCH.png"
        )
    for label, path in (("reference", reference), ("search", search)):
        if not path.exists():
            raise SystemExit(f"error: {label} image not found: {path}")
    return reference, search


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Prints exactly one line to stdout; returns an exit code."""
    args = build_parser().parse_args(argv)
    reference_path, search_path = _resolve_paths(args)

    try:
        reference = load_gray_float(reference_path)
        search = load_gray_float(search_path)
    except FileNotFoundError as exc:
        raise SystemExit(f"error: {exc}")

    if reference.size > search.size:
        print(f"warning: reference {reference.shape} is larger than search {search.shape}; "
              f"the arguments may be swapped", file=sys.stderr)

    try:
        result, corr = localize(reference, search, downsample=args.downsample,
                                fast=args.fast, return_corr=True)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    # THE CONTRACT: exactly one line, nothing else, ever.
    print(f"{result.x_int} {result.y_int}")

    if args.json:
        ensure_dir(args.json.parent)
        args.json.write_text(
            json.dumps(result_to_json(result, reference_path, search_path), indent=2) + "\n",
            encoding="utf-8",
        )
    if args.debug:
        save_debug_image(args.debug, search, corr, result)

    print(f"localized in {result.runtime_ms:.0f} ms  peak NCC {result.confidence:.4f}  "
          f"psr {result.psr:.1f}  scale {result.scale:.2f}  theta {result.theta_deg:+.1f}deg  "
          f"{'AMBIGUOUS[' + ('x' if result.ambig_x else '') + ('y' if result.ambig_y else '') + ']'
             if result.ambiguous else 'unambiguous'}"
          f"{' (centre rule fired)' if result.centre_rule_applied else ''}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
