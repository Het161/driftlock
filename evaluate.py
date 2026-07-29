#!/usr/bin/env python3
"""Drift-Sense / DriftLock -- Phase 4 evaluation harness (PROJECT_SPEC.md §6).

Scores :mod:`localize` against generated ground truth and writes the numbers and
figures the submission needs.

    python evaluate.py --data-dir data --tolerance 5 --report-dir report

By default the localization function is imported and called directly, which is
fast and gives exact per-pair timings.  ``--subprocess`` instead shells out to
``localize.py`` through its real CLI, exactly the way the graders will invoke
it, and parses only stdout.  Running both and comparing is how we prove the
scored script actually works end to end rather than merely importing cleanly.

``--data-dir`` accepts several directories, which pools them into one
evaluation -- that is how the mixed DRAM + FinFET row is produced, since the
official test set contains both families.

Outputs written to ``--report-dir``
-----------------------------------
    results.csv             per pair: predictions, truth, error, runtime, psr, ambiguity
    success_case.png        best pair: reference | search + predicted/GT | correlation
    failure_case.png        worst pair, same layout, annotated with its PSR
    robustness_noise.png    accuracy against noise preset
    robustness_rotation.png error against |theta|
    error_hist.png          error distribution
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from common import ensure_dir, load_gray_float
from localize import localize

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Secondary tolerance always reported alongside ``--tolerance``.
SECONDARY_TOLERANCE_PX = 10.0

#: Errors below this are plotted at this value so a log-scale histogram can show
#: them; sub-0.01 px errors are indistinguishable from exact for our purposes.
ERROR_FLOOR_PX = 0.01

CSV_COLUMNS = [
    "id", "style", "noise_level", "pure_lattice",
    "pred_x", "pred_y", "true_x", "true_y", "error_px",
    "runtime_ms", "confidence", "psr", "ambiguous", "ambig_x", "ambig_y", "n_candidates",
    "centre_rule_applied", "theta_deg", "scale", "drift_px", "source",
]


@dataclass
class PairResult:
    """One evaluated pair."""

    id: str
    style: str
    noise_level: str
    pure_lattice: bool
    pred_x: float
    pred_y: float
    true_x: float
    true_y: float
    error_px: float
    runtime_ms: float
    confidence: float
    psr: float
    ambiguous: bool
    ambig_x: bool
    ambig_y: bool
    n_candidates: int
    centre_rule_applied: bool
    theta_deg: float
    scale: float
    drift_px: float
    source: str
    data_dir: Path = field(default=Path("."), repr=False)
    ref_file: str = field(default="", repr=False)
    search_file: str = field(default="", repr=False)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _record_common(record: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """Pull the descriptive fields a row needs, tolerating older generators."""
    return {
        "id": record["id"],
        "style": record.get("style", meta.get("style", "?")),
        "noise_level": record.get("noise_level", meta.get("noise_level", "?")),
        "pure_lattice": bool(record.get("pure_lattice", meta.get("pure_lattice", False))),
        "theta_deg": float(record.get("theta_deg", float("nan"))),
        "scale": float(record.get("scale", float("nan"))),
        "drift_px": float(record.get("drift_px", float("nan"))),
    }


def evaluate_dir(
    data_dir: Path,
    use_subprocess: bool = False,
    limit: int | None = None,
    interpreter: str | None = None,
) -> list[PairResult]:
    """Evaluate every pair in one generated dataset directory.

    Raises:
        FileNotFoundError: If ``ground_truth.json`` is missing.
    """
    truth_path = data_dir / "ground_truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(
            f"no ground_truth.json in {data_dir} -- run generate_dataset.py "
            f"--output-dir {data_dir} first"
        )
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    meta, records = payload["meta"], payload["pairs"]
    if limit is not None:
        records = records[:limit]

    rows: list[PairResult] = []
    for record in records:
        ref_path = data_dir / record["ref_file"]
        search_path = data_dir / record["search_file"]
        common = _record_common(record, meta)

        if use_subprocess:
            pred_x, pred_y, extra, elapsed = _localize_via_cli(
                ref_path, search_path, interpreter or sys.executable)
        else:
            reference = load_gray_float(ref_path)
            search = load_gray_float(search_path)
            started = time.perf_counter()
            result = localize(reference, search)
            elapsed = (time.perf_counter() - started) * 1000.0
            pred_x, pred_y = result.x, result.y
            extra = {
                "confidence": result.confidence, "psr": result.psr,
                "ambiguous": result.ambiguous, "n_candidates": result.n_candidates,
                "ambig_x": result.ambig_x, "ambig_y": result.ambig_y,
                "centre_rule_applied": result.centre_rule_applied,
            }

        rows.append(PairResult(
            **common,
            pred_x=float(pred_x), pred_y=float(pred_y),
            true_x=float(record["true_x"]), true_y=float(record["true_y"]),
            error_px=float(np.hypot(pred_x - record["true_x"], pred_y - record["true_y"])),
            runtime_ms=float(elapsed),
            confidence=float(extra["confidence"]),
            psr=float(extra["psr"]),
            ambiguous=bool(extra["ambiguous"]),
            ambig_x=bool(extra.get("ambig_x", False)),
            ambig_y=bool(extra.get("ambig_y", False)),
            n_candidates=int(extra["n_candidates"]),
            centre_rule_applied=bool(extra["centre_rule_applied"]),
            source="subprocess" if use_subprocess else "import",
            data_dir=data_dir,
            ref_file=record["ref_file"], search_file=record["search_file"],
        ))
    return rows


def _localize_via_cli(
    ref_path: Path, search_path: Path, interpreter: str
) -> tuple[float, float, dict[str, Any], float]:
    """Invoke localize.py through its real CLI, parsing stdout as a judge would.

    stdout is the authoritative answer; ``--json`` is requested only for the
    diagnostic fields the report tabulates.

    Raises:
        RuntimeError: If the CLI fails or stdout is not exactly one "x y" line.
    """
    script = Path(__file__).resolve().parent / "localize.py"
    json_path = ref_path.parent / f".{ref_path.stem}_eval.json"
    started = time.perf_counter()
    completed = subprocess.run(
        [interpreter, str(script), "--reference", str(ref_path),
         "--search", str(search_path), "--json", str(json_path)],
        capture_output=True, text=True,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        raise RuntimeError(f"localize.py exited {completed.returncode}: {completed.stderr.strip()}")

    lines = [ln for ln in completed.stdout.split("\n") if ln.strip()]
    if len(lines) != 1 or len(lines[0].split()) != 2:
        raise RuntimeError(
            f"localize.py violated its stdout contract; expected one 'x y' line, got "
            f"{completed.stdout!r}"
        )
    x_str, y_str = lines[0].split()
    extra = json.loads(json_path.read_text(encoding="utf-8"))
    json_path.unlink(missing_ok=True)
    return float(x_str), float(y_str), extra, elapsed


# --------------------------------------------------------------------------- #
# Summarising
# --------------------------------------------------------------------------- #


def summarize(rows: Sequence[PairResult], tolerance: float) -> dict[str, Any]:
    """Aggregate one group of results into the reported statistics."""
    if not rows:
        return {"n": 0}
    err = np.array([r.error_px for r in rows])
    rt = np.array([r.runtime_ms for r in rows])
    amb = np.array([r.ambiguous for r in rows])
    amb_x = np.array([r.ambig_x for r in rows]); amb_y = np.array([r.ambig_y for r in rows])
    n = len(rows)
    hit = int((err <= tolerance).sum())
    p = hit / n
    return {
        "n": n,
        "hit": hit,
        "rate": p,
        # Wilson-free normal-approximation half-width, for boundary reading only.
        "ci95": 1.96 * float(np.sqrt(max(p * (1 - p), 1e-12) / n)),
        f"hit@{int(SECONDARY_TOLERANCE_PX)}": int((err <= SECONDARY_TOLERANCE_PX).sum()),
        "mean_err": float(err.mean()),
        "median_err": float(np.median(err)),
        "p95_err": float(np.percentile(err, 95)),
        "max_err": float(err.max()),
        "mean_ms": float(rt.mean()),
        "ambiguity_rate": float(amb.mean()),
        "ambig_x_rate": float(amb_x.mean()), "ambig_y_rate": float(amb_y.mean()),
    }


def summary_line(label: str, stats: dict[str, Any], tolerance: float) -> str:
    """One fixed-width table row."""
    if not stats.get("n"):
        return f"{label:<30}  (no pairs)"
    return (f"{label:<30}{stats['hit']:>4}/{stats['n']:<4}{100 * stats['rate']:>6.0f}%"
            f"{stats[f'hit@{int(SECONDARY_TOLERANCE_PX)}']:>6}/{stats['n']:<4}"
            f"{stats['mean_err']:>9.2f}{stats['median_err']:>9.2f}{stats['p95_err']:>9.2f}"
            f"{100 * stats['ambig_x_rate']:>7.0f}%{100 * stats['ambig_y_rate']:>7.0f}%"
            f"{stats['mean_ms']:>8.0f}")


def summary_header(tolerance: float) -> str:
    return (f"{'group':<30}{f'<={tolerance:g}px':>10}{'rate':>6}"
            f"{f'<={SECONDARY_TOLERANCE_PX:g}px':>11}{'mean':>9}{'median':>9}{'p95':>9}"
            f"{'amb_x':>8}{'amb_y':>7}{'ms':>8}")


def write_csv(path: Path, rows: Iterable[PairResult]) -> None:
    """Write the per-pair results table (PROJECT_SPEC.md §6)."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in asdict(row).items() if k in CSV_COLUMNS})


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def case_figure(path: Path, row: PairResult, title: str, annotate_psr: bool) -> None:
    """Reference | search with predicted + GT crosshairs | correlation heatmap (§6)."""
    plt = _plt()
    reference = load_gray_float(row.data_dir / row.ref_file)
    search = load_gray_float(row.data_dir / row.search_file)
    _, corr = localize(reference, search, return_corr=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6))
    fig.suptitle(
        f"{title} -- {row.id} ({row.style}, {row.noise_level} noise)   "
        f"error {row.error_px:.2f} px   PSR {row.psr:.2f}   "
        f"{'AMBIGUOUS' if row.ambiguous else 'unambiguous'}   "
        f"theta {row.theta_deg:+.2f}deg  scale {row.scale:.4f}",
        fontsize=11)

    axes[0].imshow(reference, cmap="gray")
    axes[0].set_title("Reference @100x", fontsize=9)

    ax = axes[1]
    ax.imshow(search, cmap="gray")
    _cross(ax, row.true_x, row.true_y, "#3399ff", 40, 10)
    _cross(ax, row.pred_x, row.pred_y, "#00ff66", 28, 7)
    ax.set_title(f"Search @10x -- predicted (green) vs ground truth (blue)\n"
                 f"pred ({row.pred_x:.1f}, {row.pred_y:.1f})  "
                 f"true ({row.true_x:.1f}, {row.true_y:.1f})", fontsize=9)

    ax = axes[2]
    im = ax.imshow(corr, cmap="inferno")
    ax.set_title("Correlation surface (best scale/rotation)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    if annotate_psr:
        ax.text(0.02, 0.98,
                f"PSR {row.psr:.2f}\n{row.n_candidates} candidates within 0.02\n"
                f"{'centre rule fired' if row.centre_rule_applied else 'centre rule idle'}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color="white",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    for a in axes[:2]:
        a.set_xticks([]); a.set_yticks([])
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ensure_dir(path.parent)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _cross(ax, x: float, y: float, color: str, arm: float, gap: float) -> None:
    ax.plot([x - arm, x - gap], [y, y], color=color, lw=1.5)
    ax.plot([x + gap, x + arm], [y, y], color=color, lw=1.5)
    ax.plot([x, x], [y - arm, y - gap], color=color, lw=1.5)
    ax.plot([x, x], [y + gap, y + arm], color=color, lw=1.5)


def noise_figure(path: Path, rows: Sequence[PairResult], tolerance: float) -> None:
    """Accuracy against noise preset, split by style (§6)."""
    plt = _plt()
    order = ["low", "medium", "high"]
    styles = sorted({r.style for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    width = 0.8 / max(1, len(styles))
    for i, style in enumerate(styles):
        levels, rates, labels = [], [], []
        for j, level in enumerate(order):
            group = [r for r in rows if r.style == style and r.noise_level == level
                     and not r.pure_lattice]
            if not group:
                continue
            stats = summarize(group, tolerance)
            levels.append(j + i * width - 0.4 + width / 2)
            rates.append(100 * stats["rate"])
            labels.append(f"{stats['hit']}/{stats['n']}")
        bars = ax.bar(levels, rates, width=width, label=style)
        for bar, lab in zip(bars, labels):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, lab,
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylim(0, 108); ax.set_ylabel(f"accuracy within {tolerance:g} px (%)")
    ax.set_xlabel("sensor-noise preset")
    ax.set_title("Robustness against sensor noise")
    ax.axhline(90, ls="--", lw=0.8, color="grey")
    ax.legend(); fig.tight_layout()
    ensure_dir(path.parent); fig.savefig(path, dpi=120); plt.close(fig)


def rotation_figure(path: Path, rows: Sequence[PairResult], tolerance: float) -> None:
    """Error against |theta| (§6)."""
    plt = _plt()
    usable = [r for r in rows if np.isfinite(r.theta_deg) and not r.pure_lattice]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    theta = np.abs([r.theta_deg for r in usable])
    err = np.maximum([r.error_px for r in usable], ERROR_FLOOR_PX)
    hit = np.array([r.error_px <= tolerance for r in usable])
    ax.scatter(theta[hit], err[hit], s=22, c="#2ea44f", label="within tolerance")
    ax.scatter(theta[~hit], err[~hit], s=34, c="#d63b3b", marker="x", label="miss")
    ax.set_yscale("log"); ax.axhline(tolerance, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("|theta| (degrees)"); ax.set_ylabel("error (px, log)")
    ax.set_title("Error against stage rotation")
    ax.legend(); fig.tight_layout()
    ensure_dir(path.parent); fig.savefig(path, dpi=120); plt.close(fig)


def error_hist_figure(path: Path, rows: Sequence[PairResult], tolerance: float) -> None:
    """Error distribution on a log axis -- errors span four decades (§6)."""
    plt = _plt()
    err = np.maximum([r.error_px for r in rows if not r.pure_lattice], ERROR_FLOOR_PX)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.logspace(np.log10(ERROR_FLOOR_PX), np.log10(max(err.max(), 10.0) * 1.2), 40)
    ax.hist(err, bins=bins, color="#1E3A8A", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(tolerance, ls="--", color="#d63b3b",
               label=f"tolerance {tolerance:g} px")
    ax.set_xlabel("error (px, log)"); ax.set_ylabel("pairs")
    ax.set_title(f"Error distribution (n={len(err)})")
    ax.legend(); fig.tight_layout()
    ensure_dir(path.parent); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation harness."""
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Score localize.py against generated ground truth and write "
                    "the report tables and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, nargs="+", default=[Path("data")],
                        help="dataset directory/directories; several are pooled, which "
                             "is how the mixed DRAM+FinFET row is produced")
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="hit tolerance in search-image pixels")
    parser.add_argument("--report-dir", type=Path, default=Path("report"),
                        help="destination for results.csv and figures")
    parser.add_argument("--subprocess", action="store_true",
                        help="invoke localize.py through its real CLI instead of "
                             "importing it, exactly as a grader would")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate at most this many pairs per directory")
    parser.add_argument("--no-figures", action="store_true",
                        help="tables only, skip figure rendering")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = build_parser().parse_args(argv)

    rows: list[PairResult] = []
    for data_dir in args.data_dir:
        rows.extend(evaluate_dir(data_dir, use_subprocess=args.subprocess,
                                 limit=args.limit))
    if not rows:
        raise SystemExit("error: no pairs evaluated")

    report_dir = ensure_dir(args.report_dir)
    write_csv(report_dir / "results.csv", rows)

    tol = args.tolerance
    print(f"Drift-Sense evaluation  ({'subprocess CLI' if args.subprocess else 'imported'})")
    print(f"  data: {', '.join(str(d) for d in args.data_dir)}   tolerance {tol:g} px\n")
    print(summary_header(tol))
    print("-" * len(summary_header(tol)))

    standard = [r for r in rows if not r.pure_lattice]
    degenerate = [r for r in rows if r.pure_lattice]
    for style in sorted({r.style for r in standard}):
        for level in ("low", "medium", "high"):
            group = [r for r in standard if r.style == style and r.noise_level == level]
            if group:
                print(summary_line(f"{style} / {level}", summarize(group, tol), tol))
    if len({r.style for r in standard}) > 1:
        for level in ("low", "medium", "high"):
            group = [r for r in standard if r.noise_level == level]
            if group:
                print(summary_line(f"MIXED / {level}", summarize(group, tol), tol))
    if degenerate:
        print(summary_line("pure-lattice (degenerate)", summarize(degenerate, tol), tol))
    print("-" * len(summary_header(tol)))
    overall = summarize(standard, tol)
    print(summary_line("ALL standard pairs", overall, tol))
    print(f"\nhit-rate @{tol:g}px {100 * overall['rate']:.1f}% +/- {100 * overall['ci95']:.1f}%  |  "
          f"mean {overall['mean_err']:.2f} px  median {overall['median_err']:.2f} px  "
          f"p95 {overall['p95_err']:.2f} px  |  mean runtime {overall['mean_ms']:.0f} ms/pair")

    if not args.no_figures:
        best = min(standard, key=lambda r: r.error_px)
        worst = max(standard, key=lambda r: r.error_px)
        case_figure(report_dir / "success_case.png", best, "SUCCESS", annotate_psr=False)
        case_figure(report_dir / "failure_case.png", worst, "WORST CASE", annotate_psr=True)
        noise_figure(report_dir / "robustness_noise.png", rows, tol)
        rotation_figure(report_dir / "robustness_rotation.png", rows, tol)
        error_hist_figure(report_dir / "error_hist.png", rows, tol)
        print(f"\nfigures + results.csv -> {report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
