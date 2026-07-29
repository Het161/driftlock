# SPEC AMENDMENT v1.5 — standard-cell structure for FinFET + axis-resolved ambiguity
### FinFET generation only; DRAM is bit-identical. Adds Phase 4.5 to the localizer.

## Why

Phase 4 ran the acceptance matrix separately per style for the first time. FinFET trailed DRAM by
about ten points at every noise level:

```
              low   medium   high
dram          97%     93%     83%
finfet        87%     83%     77%
MIXED         92%     88%     80%     <- 88% at medium, below the 90% gate
```

Every FinFET failure was pure-y — x correct to a few pixels, y wrong by hundreds — with `dy`
landing on integer multiples of the gate-bar pitch:

```
pair_0001  err 277.4px  dx -2.9 (-0.29 fin)   dy 277.4 -> /gate_pitch 2.98
pair_0005  err 432.5px  dx -4.8 (-0.63 fin)   dy 432.5 -> /gate_pitch 7.16
```

The correlation surface says it without arithmetic — `docs/results/finfet_ridge_pre_v1.5.png`
shows **horizontal ridges spanning the full frame width**. Correlation is near-constant along x
within each ridge, and the ridges repeat at the gate pitch.

**This is the pure-lattice degeneracy marginalised onto one axis.** A fin field is a 1-D grating:
fins repeat in x and are perfectly uniform in y, so between gate bars there is nothing whatsoever
to fix y. Same mechanism as v1.0's uniform lattice, one dimension lower.

## A. FinFET structure — standard-cell rows (generation)

Added to `render_finfet_world()`. CITE [S13].

1. **Row boundaries** — horizontal bands at semi-regular pitch, `ROW_BASE_RANGE = U(400, 620)`
   world px, `ROW_JITTER = 0.18`, width `U(30, 70)` px, intensity `0.55 ± 0.04`. This is the
   aperiodic y-signature the fin grating cannot carry.
2. **Diffusion breaks** — Poisson-count 26 per world, irregular horizontal cuts through the
   active region, width `U(20, 55)` px at intensity 0.22.
3. **Dummy-gate doublets** — with probability 0.30 a gate carries a second bar offset by 0.34 of
   the pitch. Which gates carry one is a per-world code.
4. **Gate pitch stays REGULAR.** CPP is regular in real logic; inventing jitter there to make the
   maths easier would be fabricating physics. What varies is rows, breaks and dummies.
5. **Coverage guarantee**, as v1.3 did for stripes: gate pitch tightened to `U(450, 750)` so a
   1000 px reference crop contains a gate bar with margin rather than by a hair, and the row
   pitch is sized the same way. Both asserted per world via `_max_clear_gap`, which raises rather
   than silently emitting an unlocalizable field.

**DRAM is bit-identical.** Asserted, not assumed — 6 pairs plus ground truth fingerprinted before
and after the change: `699490cdd17b0c91d54bdf9f22ae9ded` both sides.

## B. Phase 4.5 — axis-resolved ambiguity (localizer)

A single `ambiguous` boolean throws away the most useful thing about the ridge case: x was
determined to a fraction of a pixel. The localizer now reports `ambig_x` and `ambig_y`
separately, and the official closest-to-centre rule scores candidates by distance to the frame
centre **along the degenerate axes only**, leaving a well-determined axis untouched.

An axis counts as degenerate when the rival candidates span more than `COLLINEAR_TOL_FRAME_FRAC`
(0.60) of the search frame along it. The frame, not the template, is the right yardstick: two
candidates a template-width apart is an ordinary near-tie; candidates strewn across the frame
means the image genuinely does not determine that coordinate.

Calibrated on seed 20260910. **Accuracy is flat across every tolerance tried** — 0.25 to 8.0
template widths moved dram between 19 and 21 of 24 and finfet between 19 and 21 of 24, which is
noise. The threshold was therefore chosen to make the *flag* informative, not to buy accuracy.

## C. Results (fresh seeds 20260921-27, fixed before inspection)

```
group                              <=5px  rate     <=10px     mean   median      p95   amb_x  amb_y      ms
dram / low                      29/30      97%    29/30       2.66     0.05     0.17      0%      3%     658
dram / medium                   27/30      90%    27/30      16.63     0.06   153.82     13%      3%     656
dram / high                     26/30      87%    26/30      12.11     0.08    71.28      3%      7%     655
finfet / low                    30/30     100%    30/30       0.12     0.09     0.27     10%      0%     655
finfet / medium                 28/30      93%    28/30      17.72     0.10    43.88     13%      0%     657
finfet / high                   29/30      97%    29/30      10.36     0.11     0.21      3%      0%     655
MIXED / low                     59/60      98%    59/60       1.39     0.07     0.24      5%      2%     657
MIXED / medium                  55/60      92%    55/60      17.17     0.08   152.79     13%      2%     656
MIXED / high                    55/60      92%    55/60      11.24     0.09    69.95      3%      3%     655
pure-lattice (degenerate)        1/30       3%     2/30      80.69    73.31   160.47    100%    100%     655
ALL standard pairs             169/180     94%   169/180      9.93     0.08    69.95      7%      2%     656
```

FinFET now **outperforms** DRAM at every noise level, and `ambig_y` on FinFET fields is **0%**
throughout — the y-degeneracy is gone, measured on the flag it was diagnosed with.

Overall 93.9% ± 3.5%, up from 86.7% ± 5.0% in Phase 4.

**Boundary practice on MIXED/medium** (the target row): 92% on the acceptance seeds, and pooling
two further held-out seeds gives **109/120 = 90.8% ± 5.2%**. That clears the ≥90% target but the
threshold sits inside the confidence interval, so it should be quoted as "about 91%", never as a
comfortable pass.

## D. What did NOT work — the flag still misses individual failures

The directive expected the `PSR 3.67 @ 457 px` case class to flag after this change. **It does
not.** Of 11 standard failures, 4 raise any flag; of the 6 failures above 100 px, 1 does:

```
id          style   noise      err px    psr  amb_x  amb_y  ncand
pair_0019   finfet  medium      448.7   4.70  False  False      6   <- the equivalent case, unflagged
pair_0030   finfet  high        307.6   4.74  False  False      4
pair_0007   dram    medium      189.4   3.82  False  False      2
```

Median PSR is 4.29 on hits and 3.58 on misses — still overlapping, exactly as in Phase 2 and
Phase 3. Axis resolution improved the *granularity* of the report and, with §A, the data; it did
not overturn the standing limitation:

> These signals separate degenerate **fields** from sound ones. They do not identify which
> individual answer is wrong. A low flag rate on standard fields and a 100% rate on
> `--pure-lattice` is the honest claim; "the algorithm knows when it is wrong" is not.

Tightening the tolerance to catch the 448 px case would flag roughly a quarter of correct DRAM
answers as well (frame-fraction 0.4 gave 25% x-flags on standard fields). That trade was not
taken, and the threshold was not retuned against acceptance data.

## E. Open

On 4 Aug, when the official Applied Materials starter code drops: **visually compare its FinFET
against v1.5 and report differences before changing anything.**
