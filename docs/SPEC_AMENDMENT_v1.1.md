# SPEC AMENDMENT v1.1 — fixes the degenerate world model
### Supersedes PROJECT_SPEC.md §3.1.2 and §3.1.4. Everything else stands.

## Why this amendment exists (read first)

Phase-1 ablation proved the v1.0 world is unlocalizable: a globally uniform DRAM lattice with
±0.05 per-line jitter gives the true site an NCC advantage of ~0.001 over hundreds of
lattice-aligned false positives — statistically zero information. Independent reproduction on a
4000² world confirmed the mechanism (clean: truth wins by +0.0002 = luck; capture noise: rank 6,
error 90 px) and confirmed the fix below (superstructure: rank 0, error 0.6 px, clean AND noisy).

The root cause is **not** the jitter constant — it is that v1.0 modeled a die as an *infinite
uniform lattice with uniformly placed targets*. Real DRAM fields are neither:

1. **Real DRAM has hierarchical superstructure.** Memory arrays are organized into subarray
   mats separated by sense-amplifier stripes (horizontal) and wordline-driver / decoder regions
   (vertical). A 1000×1000 px field at 10x spans many mats, so block boundaries are visible and
   make localization well-posed — while individual cells stay locally periodic.
2. **Real search captures are drift-centered.** The tool navigates to the target and lands a
   small drift away, THEN captures the search image — so the true site sits near the frame
   center, offset by the drift. This is exactly why the official rule says "return the match
   closest to the center": it is a physical prior, not an arbitrary tie-break. v1.0's uniform
   placement contradicted both the physics and the official rule; v1.1 makes the official rule
   correct on our data.

**Do NOT fix this by raising LINE_INTENSITY_JITTER.** Keep it at 0.05. Values ≥0.15 are
unphysical (no working process has 15–30% line-to-line brightness variation) and would paper
over the modeling gap with a fake signature. Jitter is no longer load-bearing.

---

## A. Replace §3.1.2 — DRAM world = lattice + superstructure

Keep the v1.0 lattice exactly as implemented (WL pitch 3F / BL pitch 2F / width 1F / via dots
r=0.4F / jitter 0.05). Then overlay, in this order:

1. **Sense-amplifier stripes (horizontal):** every `SA_PITCH` = randint(8, 16) word-line
   pitches (i.e. `SA_PITCH * 3F` px), stripe width randuniform(2F, 4F), flat intensity
   0.45 ± 0.03 per stripe, with a faint internal horizontal line texture (1–2 sub-lines at
   intensity +0.10) so stripes read as circuitry, not voids. Random global offset.
2. **Wordline-driver stripes (vertical):** every `DR_PITCH` = randint(10, 20) bit-line pitches
   (`DR_PITCH * 2F` px), width randuniform(3F, 5F), flat intensity 0.35 ± 0.03, random offset.
3. **Contamination particles (optional realism, default ON):** `--defects/--no-defects`.
   Poisson-count ~2 per world (so many pairs have 0–1 in frame): Gaussian blobs, world-coord
   sigma randuniform(15, 40) px, amplitude ±randuniform(0.10, 0.20), never within 300 px of a
   stripe crossing (keep them incidental, not landmarks-on-demand).
4. Stripe geometry randomizes **per pair** (own RNG stream, like everything else).
5. New flag `--pure-lattice`: disables 1–3 entirely, reproducing the v1.0 degenerate world on
   demand. Keep it forever — this deliberately manufactures the "highly periodic region where
   correct localization is genuinely difficult" case that the official test set includes, and
   it is our honest-failure exhibit for the results slide.

FinFET style: apply the same two stripe systems over the fin field (gate bars already give some
vertical structure; stripes still required).

## B. Replace §3.1.4 — drift-prior placement (back-solved)

Sample the TRUE position in the search frame first, then back-solve the world crop:

1. Sample drift vector `d`: direction ~ Uniform(0, 2π), magnitude ~ |Normal(0, DRIFT_SIGMA)|
   capped at DRIFT_CAP. Defaults: `--drift-sigma 120`, `--drift-cap 350` (search-image px).
2. Intended ground truth in search coords: `g = (500, 500) + d`.
3. Back-solve the world crop center so the capture lands there: `(cx, cy) = A⁻¹(10 · g)`,
   where A is the same rotation+scale affine already defined in §3.1.5. Assert the 1000×1000
   crop stays ≥ MARGIN inside the world and `g` stays ≥ 80 px inside the search frame; if not,
   resample d (log the retry count as before).
4. Record in ground_truth.json additionally: `drift_px` (magnitude), `drift_sigma` used.
5. DRIFT_SIGMA/CAP are named top-level constants + CLI flags because the organizers' own
   generator parameters are unknown until the official starter code drops (4 Aug webinar).
   When it does: re-align these defaults to match, one-line change, and note it in README.

The localizer must still search the FULL frame (no center-window shortcut) — the drift prior
belongs in the data model and in the official tie-break rule, never as a search restriction.

## C. CITATIONS.md — add two entries

```
[S9]  DRAM arrays are organized as subarray mats separated by sense-amplifier stripes and
      wordline-driver/decoder regions; block boundaries are visible structure at field scale.
      → used in: superstructure geometry (§A)
      refs: DRAM architecture literature (memory-systems texts; subarray-level DRAM
      organization papers)  (TODO Het: exact titles/DOIs)
[S10] Contamination particles are a standard artifact in SEM-based wafer inspection.
      → used in: --defects blobs (§A.3)   (TODO Het: refs)
```

## D. Phase-1 acceptance — REPLACED with an ablation gate

Rerun generation and reproduce the Phase-1 ablation table on the NEW world. Gate to pass
before Phase 2 (plain single-scale NCC, no Phase-3 upgrades):

| condition                                   | required result            |
|---------------------------------------------|----------------------------|
| v1.1 world, clean, no warp                  | truth rank 0, err < 3 px   |
| v1.1 world, full capture noise (medium)     | truth rank 0, err < 3 px   |
| v1.1 world, capture noise + warp            | err < 10 px                |
| `--pure-lattice`, capture noise             | must still LOSE (rank ≫ 0) |

The last row is not a bug — it is the control proving the hard case still exists on demand.
Also confirm: byte-identical reruns at fixed seed; geometry invariant across noise levels;
previews clearly show mat boundaries crossing the frame.

## E. Git hygiene (do this before any commit)

The working folder currently resolves to a git root at ~/Desktop with ~25k pending changes.
Do NOT make a scoped commit into that repo. Instead, inside `drift-sense/`:

```
git init
printf "data/\n__pycache__/\n.venv/\nreport/\n*.pyc\n" > .gitignore
git add -A && git commit -m "phase-1: dataset generator (dram + superstructure v1.1)"
```

A nested repo takes precedence for everything under it. Keep exactly one sample pair tracked
under `data/sample/` per §8 (adjust .gitignore with `!data/sample/`).

## F. Unchanged

Noise model, edge-brightening, blur, scan-line correlation, capture independence, CLI, seeds,
outputs, Phase 2–5 definitions and acceptance criteria — all unchanged. Phase 2's ≥8/10 @10 px
gate is now reachable and stays as written.
