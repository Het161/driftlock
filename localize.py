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

Baseline pipeline (PROJECT_SPEC.md §4, Phase 2)
-----------------------------------------------
    1. load both as grayscale float32, normalize to zero mean / unit std
    2. light Gaussian denoise on the SEARCH image only (sigma 1.0)
    3. downscale the reference by exactly 10 (INTER_AREA) -> ~100x100 template
    4. cv2.matchTemplate(search, template, TM_CCOEFF_NORMED)
    5. global peak -> template top-left converted to centre coords -> print

This is the honest baseline: a single scale, no rotation search, no sub-pixel
refinement, and no periodic-ambiguity handling.  Those are Phase 3 upgrades and
land behind this same CLI without breaking it.  The ``--json`` schema already
carries the fields Phase 3 fills (``psr``, ``ambiguous``); the baseline reports
them as null rather than guessing a value it cannot compute.

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


@dataclass
class LocalizationResult:
    """Outcome of one localization, in search-image pixel coordinates."""

    x: float
    y: float
    confidence: float          # peak TM_CCOEFF_NORMED score, in [-1, 1]
    psr: float | None          # Phase 3: peak-to-sidelobe ratio
    ambiguous: bool | None     # Phase 3: multiple near-equal peaks found
    template_size: int
    search_size: tuple[int, int]
    runtime_ms: float
    method: str = "baseline-ncc-single-scale"

    @property
    def x_int(self) -> int:
        return int(round(self.x))

    @property
    def y_int(self) -> int:
        return int(round(self.y))


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


def localize(
    reference: np.ndarray,
    search: np.ndarray,
    *,
    downsample: int = DOWNSAMPLE,
    denoise_sigma: float = SEARCH_DENOISE_SIGMA,
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
    template = cv2.resize(ref_n, (tw, th), interpolation=cv2.INTER_AREA)

    # 4. normalized cross-correlation over the whole frame (§4.4)
    corr = cv2.matchTemplate(search_n, template, MATCH_METHOD)

    # 5. global peak -> template top-left -> centre, in the generator's GT
    #    convention p_search = A(cx, cy) / 10 (§3.1.6)
    _, peak, _, loc = cv2.minMaxLoc(corr)
    result = LocalizationResult(
        x=float(loc[0] + tw / 2.0),
        y=float(loc[1] + th / 2.0),
        confidence=float(peak),
        psr=None,               # Phase 3
        ambiguous=None,         # Phase 3
        template_size=int(tw),
        search_size=(int(search.shape[1]), int(search.shape[0])),
        runtime_ms=(time.perf_counter() - started) * 1000.0,
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
        "phase": 2,
        "notes": "psr and ambiguous are Phase 3 fields; the baseline cannot compute them",
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
                                return_corr=True)
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

    print(f"localized in {result.runtime_ms:.1f} ms  "
          f"peak NCC {result.confidence:.4f}  template {result.template_size}px",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
