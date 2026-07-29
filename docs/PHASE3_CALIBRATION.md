# Phase 3 — ambiguity-gate calibration and policy evidence

Every threshold in `localize.py` was calibrated on **seed 20260810** and evaluated on seeds it
had never seen. Two of them ended up set *against* what the calibration set said, because the
holdout contradicted it. Both records are kept here so neither is rediscovered the hard way.

## 1. Which signal can gate ambiguity

Three candidates were measured. Two were rejected on evidence.

| signal | standard fields | `--pure-lattice` | verdict |
|---|---|---|---|
| peak NCC | hits 0.705–0.913 | misses 0.800–0.879 | **rejected** (Phase 2, fully overlapping) |
| score gap | p10 = 0.001 | max = 0.002 | **rejected** (does not separate) |
| PSR | min 2.655 | max 2.151 | **separates** |
| candidate count | median 3, max 10 | median 243, min 108 | **separates hardest** |

Shipped gate: `psr < 2.5 or n_candidates > 20`. Both thresholds sit in the empty band between
the two populations.

**Trigger rates:** 0% on standard v1.4 fields, 100% on `--pure-lattice` fields. That is exactly
the requested calibration — degenerate fields always announce themselves, normal fields never
false-alarm.

### The honest limit

The gate separates *populations*, not *individual failures*. Within standard fields, pairs the
localizer got wrong had a median PSR of **3.73** against **3.71** for pairs it got right — no
separation at all. The algorithm can say "this entire field is degenerate"; it cannot say "this
particular answer is wrong". Anyone reading a low-PSR flag as per-answer confidence would be
over-reading it.

## 2. Centre-rule policy — calibration was wrong, holdout corrected it

PROJECT_SPEC §5.2 states the tie-break unconditionally; the Phase 2 measurement (ungated ran
20/24 → 18/24) argued for gating it. Phase 3 re-opened the question because the sweep changes
the correlation surface.

On the calibration seed the ungated rule looked clearly better:

```
seed 20260810, n=40      never fires   32/40
                         gated         32/40
                         always fires  37/40      <- would have shipped this
```

On three unseen seeds the ordering reversed, and held in every one:

```
seed 20260820   gated 27/30   ungated 24/30
seed 20260822   gated 29/30   ungated 26/30
seed 20260823   gated 27/30   ungated 27/30
pooled (n=90)   gated 92.2%   ungated 85.6%    difference -6.7% +/- 9.1% (95% CI)
```

The difference is **inside the noise band** — the policy is not distinguishable at these sample
sizes. Gated ships because it never lost on holdout and because it is what the drift-prior
physics argues for, not because the data proved it superior. The single-seed calibration result
was overfitting.

Consequence worth stating plainly: since the gate's trigger rate on standard fields is 0%, the
gated centre rule is a **no-op** there. Before/after accuracy on standard fields is identical.
The rule's real work is on degenerate fields, and the ambiguity flag's real value is as the
§5.4 confidence report, not as a selection mechanism.

## 3. The rotation trim was a self-inflicted loss

Rotating a square template leaves undefined corners, so the first implementation trimmed 5% from
each edge. Measured on 40 calibration pairs:

```
trim    0.00   0.02   0.05   0.08
<5px   37/40  35/40  32/40  32/40
```

Monotonic — every pixel trimmed cost accuracy. At 2° the corner triangles are ~1.7% of the
linear extent and `BORDER_REFLECT_101` fills them with plausible texture, while a 5% trim
discards 19% of the template area, and that area carries the aperiodic superstructure the match
depends on. `ROTATION_CROP_FRAC` is now 0.0 and kept as a named knob so the finding stays
reproducible.

## 4. Acceptance (shipped configuration)

Seeds fixed before any result was inspected.

```
condition                               <5px  rate   <10px  med err      p95    ambig      ms
30 medium-noise (seed 20260820)       27/30     90%   27/30      0.06   196.43       0%     657  PASS
10 high-noise  (seed 20260821)        10/10    100%   10/10      0.07     0.15       0%     655
10 pure-lattice (honest failure)       0/10      0%    0/10     78.42   305.60     100%     656

full sweep  7 scales x 5 rot          27/30     90%   27/30      0.06   196.43       0%     654
--fast      7 scales, no rot          26/30     87%   26/30      0.07   229.83       0%     156
no sweep    single scale (Phase 2)    25/30     83%   25/30      0.14   284.44       0%      23
```

The sweep buys +2/30 over the single-scale baseline and halves the median error (0.14 → 0.06 px)
at 28x the cost — still 3x inside the 2 s/pair budget. `--fast` recovers most of the accuracy at
4x the speed.

90% is exactly the acceptance threshold, so it should be read as a boundary result: pooled over
the 90 fresh pairs of §2 the rate is 92.2%, which is the better estimate of true performance.

The 10-pair high-noise result (100%) is a small sample and almost certainly optimistic — it
should not be quoted as evidence that high noise is easier than medium.
