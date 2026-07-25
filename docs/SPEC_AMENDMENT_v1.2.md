# SPEC AMENDMENT v1.2 — aperiodic mat grid
### Supersedes AMENDMENT v1.1 §A stripe-placement only. Everything else in v1.1 and the base spec stands.

## Why

The v1.1 gate failed for a measured, understood reason: regularly pitched stripes over a
regular lattice produce a structure that is still periodic at the mat pitch (~20 interchangeable
mat cells per frame → observed rank 3–39), and stripe pitches defined in F-multiples let 2/8
reference crops contain zero superstructure at all. The fix is aperiodicity + guaranteed
coverage, not further pitch tuning. (Physical basis: real floorplans break exact mat
periodicity via redundancy/spare rows, edge mats, and bank boundaries.)

Note for the record: the v1.1 §D expectation ("rank 0") was calibrated on a 4000² validation
world that held too few mats per frame to express this ambiguity. The 10000² behaviour reported
by the gate is the correct one. The gate did its job.

## A. Stripe placement — replace v1.1 §A.1–2

1. **Irregular spacing sequence** (both families): stripe positions are generated
   sequentially: `pos[k+1] = pos[k] + base_pitch * U(1 - MAT_JITTER, 1 + 1.2*MAT_JITTER)`,
   with `MAT_JITTER = 0.25` as a named constant and CLI flag `--mat-jitter` (0.0 exactly
   reproduces v1.1 regular mats — keep for the three-tier ablation:
   pure-lattice → regular mats → aperiodic mats).
2. **Absolute pitch bases, coverage-guaranteed:** `SA_BASE ~ U(550, 720)` world px,
   `DR_BASE ~ U(480, 680)` world px (no longer F-multiples). Max realized gap ≈ 720×1.30 ≈
   936 px < 1000 px crop ⇒ every reference crop intersects ≥1 stripe of each family by
   construction. Assert this per pair and log per-family stripe coverage (%) in
   ground_truth.json.
3. **Per-stripe individuality:** width sampled per stripe (SA: U(2F, 4F); DR: U(3F, 5F)),
   intensity per stripe as in v1.1 (±0.03), internal sub-lines as in v1.1.
4. **Bank boundary (extra landmark):** per axis, with probability 0.7, one extra-wide stripe
   (width U(8F, 12F), intensity 0.30 ± 0.03) at a uniform random position.
5. Lattice, jitter 0.05, defects, --pure-lattice, drift prior (§B), all unchanged.
6. Stylization note (README + citations narrative): field-scale stripe density is stylized so
   a reference field spans mat boundaries; re-tune visual density once the official Applied
   Materials starter code is released (4 Aug), one-line change to the two BASE constants.

## B. CITATIONS.md — add

```
[S12] Real DRAM floorplans are not exactly periodic at field scale: redundancy/spare
      rows and columns, edge mats, and bank boundaries break mat periodicity.
      → used in: MAT_JITTER irregular spacing, bank-boundary stripe (§A)
      refs: DRAM redundancy/repair literature; memory floorplanning texts (TODO Het)
```

## C. §D gate v1.2 — same baseline localizer (plain single-scale NCC), n=8 per row

| condition                                  | required result                          |
|--------------------------------------------|------------------------------------------|
| v1.2 world, clean, no warp                 | rank0 ≥ 7/8, err median < 3 px           |
| v1.2 world, capture noise (medium)         | rank0 ≥ 6/8, err median < 5 px           |
| v1.2 world, capture noise + warp           | err median < 10 px                       |
| `--mat-jitter 0` (v1.1 regular), noise     | intermediate: rank median in ~[2, 60]    |
| `--pure-lattice`, capture noise            | must still LOSE (rank median > 300)      |

Also report per-row: err max, rank0 count, mean stripe coverage. Reruns byte-identical at
fixed seed; geometry invariant across noise levels and flags (per-concern RNG streams).

If a v1.2 row misses its bound, STOP and report the same peak-offset diagnostic table as last
time (offsets in stripe-pitch and lattice-pitch units) — do not tune constants to pass.

## D. Repo decisions

- Commit `ablation_gate.py` at the repo top level (promote it out of scratchpad/). It is now
  part of the project's evidence chain and will be reused for Round-2 robustness work. Add a
  one-line mention in README's repo map. §2's "flat layout" now includes it.
- The `.gitignore` deviation (`data/*` + `!data/sample/`) is approved — correct git behaviour,
  keep it.

## E. Explicit non-goals for this amendment

Do not compensate for residual mat-level ambiguity inside the generator beyond §A (no
brighter landmarks-on-demand, no reduced drift). Division of labour is fixed:
- image content resolves location when the field contains aperiodic information (v1.2 makes
  that the common case);
- the official closest-to-centre rule (Phase 3) resolves residual periodic ties — the drift
  prior gives it real discriminating power;
- PSR (Phase 3) flags the cases where we are effectively guessing (--pure-lattice regime).
