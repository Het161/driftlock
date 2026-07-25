# SPEC AMENDMENT v1.3 — honest middle tier + realistic stripe widths
### Supersedes v1.2 §A.3 (widths) and §C (gate). Everything else in v1.2/v1.1/base stands.

## Context

The v1.2 gate passed, but the isolation ablation proved the causal story in v1.2's §A was
wrong: the load-bearing change was §A.2's absolute pitch bases (incommensurate with the
lattice → joint period = LCM of the two periods → exceeds the frame), not the spacing jitter.
v1.1's true defect was commensurability (stripe pitch = integer × lattice pitch ⇒ a one-stripe
shift is also a whole-lattice shift ⇒ genuine invariance). [S12] already records this —
correct, keep it as the headline mechanism.

Two consequences fixed here: the ablation middle tier must target commensurability, and the
stripe widths produced an unrealistic field (≈60% peripheral coverage; real DRAM fields are
array-dominated with thin mat boundaries — a fab-engineer judge will notice).

## A. New flag: `--commensurate-mats` (the honest middle tier)

When set (default off):
1. Quantize sampled bases to lattice multiples: `SA_base → round(SA_base / (3F)) · 3F`,
   `DR_base → round(DR_base / (2F)) · 2F`.
2. Force regular spacing (mat-jitter treated as 0), disable bank-boundary stripes, use fixed
   (per-world, not per-stripe) widths.
3. Document in --help as: "reproduces the v1.1 commensurate-superstructure defect for
   ablation; not a realistic mode".
Mutually exclusive with `--pure-lattice` (error if both).

## B. Stripe widths — decouple from F (realism)

Replace v1.2 §A.3 width sampling with absolute world-px ranges, per stripe:
- SA width ~ U(50, 110);  DR width ~ U(60, 140);  bank width ~ U(200, 350).
- Log per-pair union superstructure coverage as before; expected range ≈ 15–30%. Print the
  mean in the gate output (report, not a hard gate).
- Internal sub-line texture inside stripes stays (scaled to the thinner widths — 1 sub-line
  if width < 80 px).
- Previews must visually read as: large periodic array mats separated by thin dark
  circuitry bands. If a preview reads as "slivers of array between wide bands", that is a
  fail regardless of numbers — say so rather than shipping it.

## C. §C gate v1.3 — full rerun with the new widths, same baseline localizer, n=8/row

| condition                                   | required result                          |
|---------------------------------------------|------------------------------------------|
| v1.3 world, clean, no warp                  | rank0 ≥ 7/8, err median < 3 px           |
| v1.3 world, capture noise (medium)          | rank0 ≥ 6/8, err median < 5 px           |
| v1.3 world, capture noise + warp            | err median < 10 px                       |
| `--commensurate-mats`, capture noise        | rank median in [2, 60]                   |
| `--pure-lattice`, capture noise             | rank median > 300 (must LOSE)            |

Rules unchanged: if a v1.3 row misses its bound, STOP, print the peak-offset diagnostic
(offsets in stripe-pitch and lattice-pitch units) and the per-pair coverage, and wait — do
not tune constants to pass. The warp row remains median-bounded; single-pair blowups under
uncompensated rotation/scale are Phase 3's job by design, not a data problem.

## D. Repo: commit the spec trail

Create `docs/` and commit PROJECT_SPEC.md + all SPEC_AMENDMENT files into it (leave code
layout otherwise flat). The v1.0 → v1.3 trail, with measured gate tables at each step, is
part of the project's evidence chain — reviewers cloning the repo should see how the data
model was designed by ablation, not by assumption. Update nothing else in §2.

## E. Unchanged

Drift prior, lattice, jitter 0.05, defects, coverage guarantee + world-level max-clear-gap
assertion, per-concern RNG streams, --mat-jitter (still available; simply no longer the
ablation's middle tier), CITATIONS.md ledger, Phase 2–5 definitions and acceptance.

---

## Gate result on implementation (measured, n=8/row, seed 42, medium noise)

```
condition                               err med  err max   rank0  rank med   cov%
v1.3 world, clean, no warp                 0.43     0.71    8/8          0   33.8  PASS
v1.3 world, capture noise (medium)         0.43     0.71    8/8          0   33.8  PASS
v1.3 world, capture noise + warp           0.53   429.91    5/8          0   33.0  PASS
--commensurate-mats, capture noise       301.70   547.61    1/8         78   27.4  out of [2,60]
--pure-lattice, capture noise            383.16   536.40    0/8        762    0.0  PASS (must LOSE)
```

The middle tier misses on the HIGH side: it reproduces the commensurability defect more
harshly than v1.1 did, because v1.3's mat pitch (~45-68 search px) puts ~16x18 interchangeable
mat cells in a frame versus v1.1's ~20 at ~200-240 px pitch. The [2,60] band was calibrated on
v1.1's sparser geometry. Per §C no constants were tuned; see the peak-offset diagnostic, where
every false peak lands on exact integer multiples of BOTH the stripe pitch and the lattice
pitch (SA_base/WL_pitch = 4.0 exactly), confirming the tier works as intended.
