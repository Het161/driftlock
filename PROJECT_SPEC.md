# PROJECT SPEC — Drift-Sense: Navigation-Error Recovery
### SEMICON India Hackathon 2026 · Applied Materials Track (PS2) · Team submission

You are building a complete, judge-ready solution for a semiconductor computer-vision challenge.
Read this ENTIRE spec before writing any code. Implement it **phase by phase** — stop after each
phase, show me sample outputs, and wait for my confirmation before continuing.

---

## 1. Problem Summary (context — do not skip)

A wafer inspection tool must relocate the exact same site on a chip die repeatedly. A **Reference
Image** is captured at 100x magnification. A **Search Image** is captured at 10x magnification and
covers a much larger area (1000×1000 px). The reference pattern therefore appears **shrunk ~10x**
inside the search image, occupying roughly **100×100 px (~1% of the area)**.

The task: given a reference image and a search image, output the **center (x, y) in pixels** of
where the reference pattern appears inside the search image.

Why it's hard:
1. **High periodicity** — DRAM/FinFET layouts repeat; template matching produces a lattice of
   near-identical false-positive peaks.
2. **10x scale difference** with small magnification jitter (~±3%).
3. **Independent sensor noise** — reference and search are separate physical captures; the search
   image is significantly noisier (lower magnification = more noise).
4. **No dataset provided** — we must generate our own synthetic-but-physically-realistic data.

Official rule from the problem statement: **if multiple matching regions are found, return the one
closest to the center of the Search Image.** This rule MUST be encoded in the algorithm.

The organizers will run our `localize.py` **as-is** on their own hidden test pairs. If it doesn't
run without manual edits, we score zero. Their test set is noisier than typical training data and
deliberately includes at least one highly periodic region to test **failure-mode awareness**.

---

## 2. Repository Layout (flat, judge-friendly)

```
drift-sense/
├── README.md               # quickstart a stranger can follow — see §8
├── requirements.txt        # pinned, minimal (numpy, opencv-python, scipy, matplotlib, pillow)
├── common.py               # shared helpers (io, seeding, transforms) — keep small
├── generate_dataset.py     # Phase 1 — standalone CLI
├── localize.py             # Phase 2+3 — standalone CLI — THE SCORED SCRIPT
├── evaluate.py             # Phase 4 — standalone CLI
├── smoke_test.py           # Phase 5 — 60-second end-to-end sanity check
├── CITATIONS.md            # physics justification ledger — see §7
└── data/                   # generated output (gitignored except one sample pair)
```

Hard constraints for ALL code:
- Python 3.10+. **No GPU, no torch, no notebooks, no network calls, no hardcoded absolute paths.**
- Every script: `argparse` CLI with `--help`, docstrings, type hints, meaningful errors.
- **Determinism**: every script accepts `--seed` (default 42); per-pair seeds derived from it.
- Grayscale single-channel images throughout, saved as 8-bit PNG.
- Performance target: `localize.py` completes in **< 2 seconds per pair on laptop CPU**.

---

## 3. PHASE 1 — `generate_dataset.py` (synthetic wafer image generator)

### CLI
```
python generate_dataset.py --style dram --num-pairs 30 --output-dir data \
    --seed 42 --noise-level medium
```
- `--style {dram,finfet}` — default `dram` (primary). FinFET implemented behind the same
  interface, simpler version, flagged experimental.
- `--noise-level {low,medium,high}` — maps to the (N_e, b, blur σ) presets in §3.4.
- `--preview` — additionally save annotated composite images for each pair.

### 3.1 Geometry model (get this exactly right)

1. **World canvas**: `10000×10000` float32 in [0,1], representing the die area at
   100x-equivalent resolution. This is the clean "physical truth."
2. **DRAM structure** on the world canvas — parameters justified by real DRAM architecture
   (6F² open-bitline cell: word-line pitch 3F, bit-line pitch 2F, line width 1F,
   contact/via dot at each intersection — see CITATIONS.md):
   - Pick base feature size `F` ~ Uniform(40, 80) px per pair.
   - Horizontal word-lines: pitch `3F`, width `1F`, intensity ~0.75 with per-line jitter ±0.05
     (process variation).
   - Vertical bit-lines: pitch `2F`, width `1F`, intensity ~0.85 with per-line jitter.
   - Via dots: radius `0.4F` at every WL×BL intersection, intensity ~0.95.
   - Background ~0.15. Small global brightness/contrast jitter per pair.
3. **FinFET structure** (secondary): dense parallel vertical fin lines (pitch `1.5F`, width
   `0.5F`) crossed by 1–2 horizontal gate bars (width `2F`, higher intensity at crossings).
4. **Reference capture**: crop a `1000×1000` window from the world at full resolution.
   Crop center `(cx, cy)` chosen uniformly with ≥ 1500 px margin from world edges.
5. **Search capture** (models the tool's stage/magnification error):
   - Build affine `A` = rotation θ ~ Uniform(−2°, +2°) about world center, composed with
     isotropic scale `s` ~ Uniform(0.97, 1.03).
   - Warp the world with `A` (cv2.warpAffine, INTER_LINEAR), then downsample ×0.1 with
     `cv2.resize(..., interpolation=cv2.INTER_AREA)` → `1000×1000` search image.
6. **Ground truth**: `p_search = A(cx, cy) / 10`. Assert it lands ≥ 80 px inside the search
   frame; if not, resample the crop location. Record exact float coordinates.

### 3.2 SEM edge-brightening (mandatory per problem statement)

Physical basis: secondary-electron emission increases sharply at feature edges, producing
bright rims (the SEM "edge effect" — see CITATIONS.md). Implementation, applied to the CLEAN
structure **before** noise, independently for each capture:
```
edges = normalize(sobel_magnitude(clean))
img   = clip(clean + k * edges, 0, 1)      # k_ref ≈ 0.25, k_search ≈ 0.20
```

### 3.3 Blur (optics PSF)

Gaussian blur before noise: reference σ ≈ 0.8; search σ ~ Uniform(1.2, 2.0). Search is always
blurrier (lower magnification optics).

### 3.4 Sensor noise — mixed Poisson-Gaussian, INDEPENDENT per image

This is a hard requirement: **never reuse the same noise array on both images** — they are
separate physical captures. Use independently seeded RNG streams for reference vs search.

Physical model (see CITATIONS.md — shot noise in SEM is Poisson, signal-dependent; readout
noise is Gaussian): for clean image `x` in [0,1]:
```
counts = rng.poisson(x * N_e) / N_e          # shot noise; N_e = electrons/pixel scale
y      = clip(counts + rng.normal(0, b, x.shape), 0, 1.05)   # + readout noise
```
Presets (reference / search):
- low:    N_e 400/150,  b 0.01/0.02
- medium: N_e 250/80,   b 0.015/0.03
- high:   N_e 150/40,   b 0.02/0.05

Note the clip ceiling **1.05, not 1.0** — noise may push values slightly beyond the true
signal range, which mirrors real behaviour; rescale to 8-bit on save.

**Scan-line correlated noise (search image only — our realism differentiator):** SEM images
show row-correlated noise from line scanning. Add per-row offsets: generate 1D Gaussian noise
of length 1000, smooth with a 1D Gaussian (σ=3), scale by 0.015, add to every row.

### 3.5 Outputs
```
data/
├── pairs/pair_0001_ref.png,  pair_0001_search.png,  ... 
├── ground_truth.json    # per pair: id, true_x, true_y (floats), style,
│                        #   F, theta_deg, scale, noise_level, N_e_ref, N_e_search, seeds
└── previews/            # only with --preview: search image with GT crosshair + inset ref
```

### Phase 1 acceptance
- `--num-pairs 5 --preview` runs in < 30 s; previews visually show a plausible DRAM grid with
  bright edges, a noisier/blurrier search image, and the GT crosshair sitting on the correct
  structure. Show me 2 preview images before proceeding.

---

## 4. PHASE 2 — `localize.py` baseline (the scored script)

### CLI contract (Applied Materials runs this — keep it bulletproof)
```
python localize.py --reference path/to/ref.png --search path/to/search.png
```
- **stdout: exactly one line: `x y`** (rounded ints). Nothing else on stdout.
- `--json out.json` (optional): float coords, confidence, ambiguity flag, timing.
- `--debug debug.png` (optional): heatmap + predicted crosshair overlay.
- Also accept the two paths positionally (defensive — we don't know their exact invocation).
- Any diagnostic prints go to stderr only.

### Baseline pipeline
1. Load both grayscale float32, normalize to zero mean / unit std.
2. Light denoise on the search image only: `cv2.GaussianBlur(σ=1.0)`.
3. Downscale the reference by exactly 10 (INTER_AREA) → ~100×100 template.
4. `cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)` → correlation map.
5. Global peak → convert template top-left to center coords → print.

### Phase 2 acceptance
- On 10 freshly generated `low`-noise pairs: ≥ 8/10 predictions within 10 px of GT.
- Runtime < 0.5 s/pair. Show me the numbers.

---

## 5. PHASE 3 — robustness upgrades (this is where we win)

Upgrade `localize.py` (same CLI, no breaking changes):

1. **Multi-scale × multi-rotation sweep**: scales `np.linspace(9.6, 10.4, 7)` (resize the
   reference by 1/s each) × rotations `np.linspace(-2, 2, 5)` degrees (rotate the template).
   Track the best (score, scale, θ, corr-map). All matchTemplate calls are FFT-backed and fast;
   keep total < 2 s.
2. **Periodic-ambiguity resolution**: on the best correlation map, find ALL local maxima with
   score ≥ best − 0.02 (local maxima via dilate-compare, min separation ~ half template size).
   If more than one candidate → **select the one closest to the search-image center** (this is
   the official tie-break rule from the problem statement) and set `ambiguous = true`.
3. **Sub-pixel refinement**: fit a 2D quadratic to the 3×3 correlation neighborhood around the
   chosen peak; add the fractional offset. (Standard parabolic peak interpolation.)
4. **Confidence score — PSR** (peak-to-sidelobe ratio): mask an 11×11 window around the peak;
   `psr = (peak − mean(rest)) / std(rest)`. Report in `--json`. Low PSR ⇒ the algorithm KNOWS
   it is uncertain — this feeds our failure-mode-awareness story (the official test set
   explicitly includes a genuinely ambiguous periodic case).
5. Optional flag `--fast` that skips the rotation sweep (for speed benchmarking).

### Phase 3 acceptance
- On 30 `medium`-noise pairs: ≥ 90% within 5 px; report mean error and mean runtime.
- On 10 `high`-noise pairs: report honestly, no tuning-to-the-test. Show me the table.

---

## 6. PHASE 4 — `evaluate.py` + figures (feeds Slide 6 directly)

### CLI
```
python evaluate.py --data-dir data --tolerance 5 --report-dir report
```
Imports the localization function directly (no subprocess) but ALSO includes a `--subprocess`
mode that shells out to `localize.py` exactly the way a judge would — proving the CLI works.

Computes and writes to `report/`:
- `results.csv` — per pair: predicted x/y, true x/y, error px, runtime, psr, ambiguous flag.
- Summary printed: hit-rate @5 px and @10 px, mean/median error, p95 error, mean runtime.
- `robustness_noise.png` — accuracy vs noise level (uses params logged in ground_truth.json).
- `robustness_rotation.png` — error vs |θ|.
- `error_hist.png`.
- `success_case.png` — side-by-side: reference | search with predicted ✚ (green) and GT ✚
  (blue) | correlation heatmap.
- `failure_case.png` — the WORST pair, same layout, annotated with its PSR value.
  (The submission template demands one honest failure example — this generates it.)

### Phase 4 acceptance
- Full run on 30 pairs produces all figures; show me `success_case.png` and the summary line.

---

## 7. CITATIONS.md — the 30% score ledger

30% of the judging score is "augmentation justification": every physics choice must map to
2–3 credible references. Create `CITATIONS.md` with this structure, and **tag the code**: every
physics-motivated constant carries a comment like `# CITE: [S2] Poisson shot noise`.

```
[S1] SEM noise is dominated by Poisson shot noise (signal-dependent), not pure Gaussian.
     → used in: generate_dataset.py Poisson term
     refs: Timischl et al., SEM SNR estimation lit.; Sim/Kamel SEM signal-noise model;
           (TODO Het: pull exact titles/DOIs — placeholders below each entry)
[S2] Mixed Poisson-Gaussian model y = a·Poisson(x/a) + N(0, b²); Var = a·x + b².
     → used in: §3.4 noise implementation
[S3] Row/line-scan correlated noise is present in SEM acquisitions.
     → used in: scan-line noise term (search image)
[S4] SEM edge effect: secondary-electron yield rises at edges/protrusions → bright rims;
     exploited industrially in CD-SEM metrology.
     → used in: §3.2 edge-brightening; refs: JEOL SEM glossary (edge effect), ETH Zürich
       SE-imaging notes, CD-SEM measurement patents
[S5] DRAM 6F² open-bitline geometry: WL pitch 3F / BL pitch 2F / width 1F, orthogonal
     grid, contacts at intersections.
     → used in: §3.1 structure ratios; refs: DRAM design overviews, US patents on 6F² layout
[S6] NCC score degrades under small rotation/scale change → motivates multi-scale/rotation
     sweep; refs: machine-vision pattern matching literature/patents
[S7] Phase correlation / parabolic peak interpolation for sub-pixel registration.
[S8] Multi-scale template matching + edge-preserving filtering is the industrial approach
     for wafer alignment.
```
Leave `TODO Het:` markers where exact bibliographic details must be filled in manually — do
NOT fabricate DOIs or paper titles you are not certain of.

---

## 8. PHASE 5 — README.md, requirements.txt, smoke_test.py, sample data

**README.md** must let a stranger go from clone → result in 4 commands:
```
pip install -r requirements.txt
python generate_dataset.py --num-pairs 30 --output-dir data --seed 42
python localize.py --reference data/pairs/pair_0001_ref.png \
                   --search data/pairs/pair_0001_search.png
python evaluate.py --data-dir data --tolerance 5 --report-dir report
```
Plus: 3-sentence problem summary, algorithm overview diagram (ASCII is fine), results table
placeholder, repo map, citations pointer, team section placeholder.

**requirements.txt**: minimal pinned versions (generate from the working environment).

**smoke_test.py**: generates 3 low-noise pairs into a temp dir, runs localization via the real
CLI (subprocess), asserts all errors < 10 px and total runtime < 60 s, prints PASS/FAIL.
This is our "runs as-is on a fresh machine" insurance.

**Sample data**: commit exactly one pair + its GT entry under `data/sample/` so the repo
demonstrates the format even before generation is run. Gitignore the rest of `data/`.

### Phase 5 acceptance
- `smoke_test.py` passes from a **fresh virtualenv**. Paste the full console output.

---

## 9. What NOT to do
- ❌ No web app, no Streamlit, no frontend of any kind (zero marks in Rounds 1–2).
- ❌ No deep learning in this build (classical first; a learned re-ranker is a possible
  Round-2 addition only if evaluation data justifies it).
- ❌ No fabricated citations — placeholders with TODO markers only.
- ❌ Do not tune thresholds against a single fixed dataset; keep constants named and
  documented at the top of each script.
- ❌ Do not print anything except `x y` on localize.py stdout.

## 10. Working agreement
- Implement Phase 1 → STOP → show outputs → wait for my go-ahead. Then 2, 3, 4, 5.
- Commit at the end of each phase with a message like `phase-1: dataset generator (dram)`.
- After each phase, give me a 5-line plain-English explanation of the key logic decisions —
  I need to be able to defend every block in front of the jury without notes.
