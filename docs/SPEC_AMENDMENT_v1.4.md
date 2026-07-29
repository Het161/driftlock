# SPEC AMENDMENT v1.4 — isotropic stripe families
### Supersedes v1.3 §B (widths) and the v1.2/v1.3 pitch and intensity constants. Everything else stands.

## Why

The Phase 2 baseline failed its acceptance gate at 7/10 (gate: ≥ 8/10 within 10 px), and the
failure was **purely anisotropic**: every miss had x correct to sub-pixel and y wrong by
243–287 px.

```
pair          theta    scale   err px      dx      dy   dx/DR   dy/SA
pair_0001     +0.62   0.9754   243.06    -2.6  -243.0   -0.04   -3.62
pair_0003     +1.71   0.9985   287.30    -8.5  -287.2   -0.13   -4.23
pair_0004     +0.09   0.9947   248.66     0.6   248.7    0.01    4.40
```

The peak-offset diagnostic named the competing periodicity exactly. `dx/DR` ≈ 0 and
`dx/BL` ≈ 0 — x is fully pinned by the vertical driver stripes. `dy/SA` is non-integer, but
`dy/WL` lands on **exact integers**: the competitors are whole word-line lattice shifts in y.

```
   score       dx       dy   dx/DR   dy/SA   dx/BL   dy/WL
  0.9083      0.4     -0.4    0.01   -0.01    0.03   -0.02  <- truth
  0.9075      4.4   -255.4    0.09   -3.65    0.29  -11.00
  0.9050      2.4   -139.4    0.05   -1.99    0.16   -6.00
  0.9045     -2.6    162.6   -0.05    2.33   -0.17    7.00
  0.8943     -7.6    441.6   -0.16    6.32   -0.49   19.02
  0.8924     -3.6    231.6   -0.07    3.31   -0.23    9.98
  0.8912     -1.6     69.6   -0.03    1.00   -0.10    3.00
  0.8763      3.4   -209.4    0.07   -2.99    0.22   -9.02
```

Warp was ruled out by isolation before the cause was accepted: with rotation and scale switched
off the baseline scored 21/24, with them on 20/24 — a cost of one pair in twenty-four. The
missed pairs were also *less* warped than the hits (`pair_0004`: θ = +0.09°, scale = 0.9947,
essentially unwarped, still 249 px out).

**Root cause.** The two stripe families were never symmetric, and the asymmetry was arbitrary
rather than physical:

| | v1.3 intensity | contrast vs ~0.85 array | width | base pitch |
|---|---|---|---|---|
| SA (horizontal, pins y) | 0.45 | 0.40 | U(50, 110) | U(550, 720) |
| DR (vertical, pins x) | 0.35 | 0.50 | U(60, 140) | U(480, 680) |

Discriminating variance goes as contrast² × width/pitch, giving the vertical family roughly
5× the horizontal one (0.20² vs 0.10² in the amendment's normalisation). x was pinned; y was
left to lose against a whole-lattice shift. Nothing in DRAM physics motivates a sense-amp
stripe being lower-contrast or narrower than a wordline-driver stripe. Removing the asymmetry
is a **model correction, not tuning-to-pass**.

## A. Isotropic stripe families

Both families now share identical parameters:

- SA stripe intensity: `0.45` → `0.35 ± 0.03` (matching DR)
- Stripe widths, both families: `U(55, 125)` world px
- Base pitch ranges, both families: `U(500, 700)` world px
- Nothing else changes.

Coverage guarantee re-derived: `700 × (1 + 1.2 × 0.25) = 910 px` maximum step, minus the 55 px
minimum width → 855 px maximum clear gap < 1000 px reference crop. Still asserted per world in
`apply_superstructure()`.

The constants are now defined once (`STRIPE_BASE_RANGE`, `STRIPE_WIDTH_RANGE_PX`,
`STRIPE_INTENSITY`) and aliased to the per-family names, so the two cannot silently diverge
again without someone deliberately editing them apart.

## B. Results

**§C gate, bands unchanged, n=24/row, seed 42, medium noise:**

```
condition                               err med  err max   rank0  rank med   cov%
v1.4 world, clean, no warp                 0.42   347.50   23/24         0   30.4  PASS
v1.4 world, capture noise (medium)         0.42   347.50   22/24         0   30.4  PASS
v1.4 world, capture noise + warp           0.39   443.54   23/24         0   29.4  PASS
--commensurate-mats, capture noise       269.42   579.30    2/24        46   28.9  PASS (in [2,300])
--pure-lattice, capture noise            340.23   536.40    0/24       583    0.0  PASS (must LOSE)
```

GATE PASSED. The warp row improved from 18/24 to 23/24 — the anisotropy, not the warp, had
been carrying that row's failures.

**Phase 2 acceptance, 10 fresh low-noise pairs, seed 20260801** (seed 20260729 retired: data
changed after seeing its failures, so it is no longer a clean holdout):

```
within 10 px : 8/10   PASS  (gate: >= 8/10)
error        : mean 52.04 px   median 0.46   max 432.67
runtime      : mean 18.0 ms   max 18.5 ms   PASS  (gate: < 500 ms/pair)
axis check   : median |dx| 0.30 px   median |dy| 0.37 px   (v1.3: 0.4 vs 243+ on misses)
```

The anisotropy is resolved — the two axes now behave identically. Wider characterisation on 40
further fresh pairs (seed 20260802, not the gate run) puts the baseline's true hit rate at
**35/40 = 88% ± 10%**, so 8/10 is a representative result rather than a lucky one.

## C. Input carried to Phase 3

Raw peak NCC does **not** separate hits from misses and must not be used as the ambiguity
signal on its own. Over 40 pairs: hits span NCC 0.705–0.913, misses span 0.800–0.879 — fully
overlapping, with miss-scores exceeding hit-scores in 88 of 175 cross pairings. The Phase 3
ambiguity gate needs peak-to-sidelobe ratio or a second-peak score gap.

Separately measured: applying the official closest-to-centre rule to *every* loose tie is
net-negative (20/24 → 18/24; it rescued 3 pairs and broke 5), because the drift prior puts the
truth a mean 103 px off centre, so the truth is frequently not the centre-most candidate. The
rule must be ambiguity-gated.

## D. Unchanged

Drift prior, lattice and its 0.05 line jitter, defects, sub-line texture inside SA stripes,
`--mat-jitter`, `--pure-lattice`, `--commensurate-mats`, coverage guarantee and world-level
max-clear-gap assertion, per-concern RNG streams, CITATIONS.md ledger, Phase 3–5 definitions
and acceptance.
