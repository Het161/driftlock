# SPEC AMENDMENT v1.6 — alignment with the official starter package

**Status:** adopted
**Supersedes:** nothing; extends v1.1–v1.5
**Trigger:** the organisers released the official starter package as a Hugging Face Space
(`aayushraina21/drift-sense-synthetic-data`), containing their own generator, a ZNCC
baseline solution, tests and a 30-slide methodology deck. SPEC_AMENDMENT_v1.1 §B.5 said
our provisional constants would be re-aligned when that landed. This is that amendment.

Everything below was measured before it was adopted. Where a change was proposed and the
measurement then refused it, the refusal is recorded rather than the proposal deleted.

---

## §0. What the official package actually specifies

Read off their code, not their prose:

| Contract item | Official value |
|---|---|
| Image sizes | both 1000×1000, reference 1 nm/px, search 10 nm/px |
| Scale | **exactly 10.0 by construction** — falls out of the pixel-size ratio |
| Rotation | **none.** No rotation term anywhere in their pipeline |
| Search-only geometry | progressive row shear + per-row jitter; optional barrel |
| Ground truth | `gt_x`/`gt_y` = float **centre** of a 100×100 box, plus `gt_box` top-left |
| Manifest | `manifest.csv`, 29 columns, not JSON |
| Architectures | 12 named presets — 6 DRAM, **6 FinFET** |
| Scoring | accuracy at tolerance (default **5 px**) + AP from a PR curve ranked by confidence |
| Submission | a script taking `--reference`/`--search`, printing the centre to stdout |

Their noise model is the **same** Poisson–Gaussian form as ours, in 0–255 units:
their `dose` ≡ our `N_e`, their `detector_noise_sigma/255` ≡ our `b`. That mapping is what
makes §E below possible.

## §A. Adaptive sweep — `localize.py`

**Problem.** The Phase-3 bank evaluates 7 scales × 5 rotations = 35 templates at ~660 ms.
On the official generator, where the true rotation is exactly 0, it selected θ≠0 on 14.4%
of pairs and scored **4.3%** within 5 px on those against **65.0%** when it selected θ=0.
Extra degrees of freedom buy extra chances to be spuriously wrong.

**Change.** Coarse-to-fine: stage A ranks scales at θ=0; stage B sweeps rotation over the
top `ADAPTIVE_TOP_K = 3` scales. 7 + 4·3 = 19 templates.

**Why K=3 and not 1.** Strict separability is unsafe — a −2° rotation depresses the true
scale's stage-A score enough that stage A picks a neighbour and stage B is locked to it.
Reproduced on dram/medium seed 42 pair 1: K=1 returned 263 px where the full bank returned
0.08 px. K=3 recovers it.

**Measured**, 120 our-pairs + 100 official-pairs, adaptive vs the full bank:

| | acc@5 ours | acc@5 official | ms/pair |
|---|---|---|---|
| full bank (Phase 3) | 92.5% | 72.0% | 656 |
| **adaptive (v1.6)** | **92.5%** | **72.0%** | **~360** |

Identical accuracy on every individual group of both generators, at 46% of the runtime.
`--exhaustive` restores the full bank.

## §B. The ambiguity gate does not transfer — and what we did about it

**Problem.** `AMBIGUITY_PSR_MIN = 2.5` was calibrated where PSR runs 2.1–6.5. On the
official generator the same algorithm produces PSR 4.5–16.0 (median 8.17), so **0 of 160
pairs** ever reach the threshold and the gate is silent on a dataset where 44% of answers
are wrong. PSR divides by the standard deviation of all non-peak pixels, so its absolute
scale is set by how much of the correlation map is flat background — a property of the
imaged structure, not of match quality.

**Change.** Added `_peak_margin()`: `(peak − best_rival) / (peak − median_local_max)`,
computed over local maxima rather than raw pixels, so it is invariant to any affine
rescale of the correlation surface. Reported in `--json` as `peak_margin`.

**Calibration**, three populations:

| population | min | p10 | median | max |
|---|---|---|---|---|
| our standard fields | 0.066 | 0.536 | 1.000 | — |
| our pure-lattice | 0.006 | 0.020 | 0.099 | 0.252 |
| official default | 0.109 | 0.245 | 1.000 | — |

At 0.30 the arm fires on 100% of pure-lattice, 6.7% of our standard fields, and 15% of
official pairs — catching 33% of official >5 px misses that PSR caught 0% of.

**What the measurement refused.** Letting this arm drive the centre rule cost **2 pairs of
220 (−0.9pp)**: it fires on near-ties that the centre rule then resolves wrongly. So the
margin feeds `ambiguous` (what we *report*) but not `ambig_x`/`ambig_y` (what we *act on*).
Unlike PSR on our own data, the two of our populations **overlap** on this axis
(0.066 vs 0.252), which is why it supplements PSR rather than replacing it.

**What it is not.** Because the candidate set is pre-filtered to near-ties, the ratio
saturates at 1.0 below about five candidates — exactly 1.0 by construction at one or two.
Measured over the 210-pair report: 100% saturated at n≤2 (143 pairs), 90.9% at n=3–4, then
informative from n≥5. So it is **not** an independent confidence axis; it is a finer reading
of `n_candidates` that resolves the 5–20 band `AMBIGUITY_CANDIDATES_MAX` steps over. It
cannot rank the 78% of pairs that saturate, and the docstring says so.

**Unchanged:** PSR remains the §5.4 confidence number, and the closest-to-centre tie-break
remains mandatory — it is quoted from the problem statement, not inferred, and §F does not
touch it.

## §C. stdout convention — `--format`

PROJECT_SPEC.md §4 fixes stdout as `"x y"` with rounded ints, but that was **our reading**;
the spec itself flags the exact invocation as unknown. The organisers' `infer.py` instead
prints `"746.00,326.00"` and claims to mirror the required interface.

**Measured:** rounding to integers flipped **zero** of 160 official pairs at either the 5 px
or 10 px tolerance (worst single-pair penalty 0.58 px). Emitting floats therefore buys no
accuracy and would break a grader parsing with `int()`.

**Change.** `--format {spec,official}`, defaulting to `spec`. The alternative is one flag
away if the grader turns out to want it. No default behaviour changed.

## §D. Sparse-landmark tier — the finding that matters most

**Problem.** `STRIPE_BASE_RANGE = (500, 700)` world px is **smaller than the 1000 px
reference crop**, so v1.2 §A.2 guarantees — and asserts — that every reference contains at
least one stripe of each family. The official generator's blocks are 2600 nm against a
1000 nm reference, so a crop can land entirely inside uniform periodic array.

Measured on official data, split by how much non-array material the reference contains:

| strip coverage of reference | share | acc@5 px | median err |
|---|---|---|---|
| 0% — pure array interior | 16% | **18.8%** | 416 px |
| 5–20% | 14% | 50.0% | 7.80 px |
| >20% | 67% | **89.6%** | 1.05 px |

That single split accounts for essentially all of the gap between our 94% on our own data
and 70% on theirs. **We had been solving a strictly easier problem**, and the ablation gate
could not detect it because the gate's worlds carry the same guarantee.

**Change.** `--sparse-landmarks` widens the stripe pitch to `(2400, 3400)` world px —
bracketing the official 2920 nm period — and relaxes the §A.2 assertion in that mode only
(both the per-world check and the per-crop check). This is now the fourth ablation tier.

**Measured**, 40 dram/medium pairs: 38% landmark-free crops (official: 16%), scoring 13.3%
within 5 px on those (official: 18.8%) — the failure mode transfers closely. The
landmark-*bearing* half does not (32.0% vs official 89.6%), because our landmarks are 1-D
stripes that pin one axis where official strips are 2-D regions with orthogonal routing
that pin both. It is a deliberate stress tier, not a calibrated replica.

## §E. Noise alignment

- **`official` noise preset**: `N_e_ref 2000, N_e_search 200, b_ref 2/255, b_search 5/255`,
  transcribed from their defaults via the unit mapping in §0. It shows our own reference
  captures run **5–13× noisier** than the official nominal, and that our `low` search tier
  is already noisier than their default.
- **`--speckle-sigma` / `--salt-pepper-prob`**: multiplicative and impulse noise, default
  `0.0` so every existing seed stays bit-identical. Added because their own `evaluate.py`
  enables speckle at its `high` tier and speckle + salt-pepper at `severe` — these are part
  of the distribution they score against, not demo-only knobs.

Not adopted: astigmatism, vignette, gamma, barrel and charging. They appear only in the
organisers' deck and Streamlit app, never in `evaluate.py`. Measured on official data with
all of them enabled at the authors' own demo values, both our localizer and their baseline
collapse (10% / 5% within 5 px) — worth knowing, but not evidence they are graded.

## §F. Uniform placement control

`--uniform-placement` drops the drift prior and places the target uniformly in the legal
frame, as the official generator does (its crop origins are uniform outside a 35%
boundary bias; there is no centre prior at all). This is the control that shows how much
our results lean on `DRIFT_SIGMA`/`DRIFT_CAP`.

**This changes only where the target is placed.** The closest-to-centre *tie-break* in
`localize.py` is a problem-statement requirement (PROJECT_SPEC.md §1, quoted: "if multiple
matching regions are found, return the one closest to the center of the Search Image") and
is untouched.

## §G. Attribution

`ablation_gate.baseline_localize` and `localize.py` both predate the starter package and
independently arrived at ZNCC with a scale sweep; the convergence is not a borrowing. What
this amendment takes from the organisers' package is recorded in CITATIONS.md as [S14]:
the parameter values in §E, the block geometry in §D, the stdout convention offered in §C,
and the evaluation contract in §0. No technique was copied.

## §H. Results were not reproducible from the documented command

Found while re-running the suite. `docs/results/results.csv` recorded no seed, and its
ground-truth values do not match what `--seed 42` — the seed the README documents —
produces from the same code. The label-only commit 30e6a7f is not the cause; the pixel data
is unchanged. The committed numbers were simply generated from inputs the file could not
name, so the README's "94% across 180 standard pairs" never reproduced.

**Verified not a v1.6 regression.** Running the pre-v1.6 `localize.py` and the v1.6 one over
the same 180 pairs gives **zero disagreements** — identical hit/miss on every pair, at 46%
of the runtime.

**Change.** `seed` and `sparse_landmarks` are now first-class `results.csv` columns
(`peak_margin` too), `docs/results/` is regenerated from the documented command, and the
README carries the reproducible figure: **87.8% ± 4.8%** across 180 standard pairs, 90% on
the mixed medium-noise row. A results file that cannot name its own inputs is not evidence.

## Open items

- Our stripe landmarks are 1-D; official strips are 2-D textured regions (§D). Making ours
  2-D would close most of the remaining realism gap.
- 65 of 160 official pairs are missed by our localizer *and* theirs. All are periodic
  aliasing on landmark-free crops. §D makes that case measurable; it does not fix it.
- No PS number appears anywhere in the official package. Our README's "PS02" is the only
  source for it.
