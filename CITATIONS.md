# CITATIONS — physics justification ledger

Every physics-motivated constant and modelling choice in this repo carries an inline
`# CITE: [Sn]` comment pointing at an entry below. This file is the 30%-of-score
"augmentation justification" artifact (PROJECT_SPEC.md §7).

> **Status:** the *claims* and the *code they justify* are final. The bibliographic
> details are deliberately left as `TODO Het:` placeholders — no DOI, title, or author
> list appears here unless it has been verified by hand. Fabricated references are worse
> than missing ones in front of a jury (PROJECT_SPEC.md §9).

---

## Noise model

**[S1] SEM noise is dominated by Poisson shot noise (signal-dependent), not pure Gaussian.**
The number of secondary electrons collected per pixel is a counting process, so variance
scales with the signal rather than being constant across the image.
→ used in: `generate_dataset.py` → `apply_sensor_noise()`, the `rng.poisson(x * N_e)` term;
constants `NOISE_PRESETS[*]["N_e_ref"|"N_e_search"]`
→ refs: Timischl et al., SEM signal-to-noise estimation literature; Sim/Kamel SEM
signal-noise model
*(TODO Het: pull exact titles / venues / DOIs)*

**[S2] Mixed Poisson-Gaussian sensor model: `y = a·Poisson(x/a) + N(0, b²)`, giving `Var = a·x + b²`.**
Shot noise (Poisson) and readout/amplifier noise (Gaussian) are separate physical
mechanisms and must be modelled as separate terms, not folded into one Gaussian.
→ used in: PROJECT_SPEC.md §3.4 implementation; constants `NOISE_PRESETS[*]["b_ref"|"b_search"]`
→ verified in-code: measured variance 0.003379 vs predicted 0.003400 at x=0.2;
0.010662 vs 0.010900 at x=0.8
→ refs: Foi et al. Poissonian-Gaussian noise modelling for raw-data imaging; standard
CCD/CMOS noise-model treatments
*(TODO Het: confirm exact Foi et al. title/year, add DOI)*

**[S3] Row / line-scan correlated noise is present in SEM acquisitions.**
An SEM builds the image by raster scanning, so beam-current and amplifier drift within a
line produce noise that is correlated *along* a row and independent between rows.
→ used in: `apply_sensor_noise(scanline=True)`, search image only; constants
`SCANLINE_SIGMA`, `SCANLINE_AMPLITUDE`
→ note: the spec-literal recipe (white noise → Gaussian smooth σ=3 → scale ×0.015) yields a
realised per-row offset std of ≈0.0047 (≈1.2 DN of 255), measured row-to-row
autocorrelation 0.55
→ refs: SEM image restoration / scan-noise correction literature
*(TODO Het: refs)*

## Signal formation

**[S4] SEM edge effect: secondary-electron yield rises at edges and protrusions → bright rims.**
More escape paths exist for secondary electrons near a topographic edge, so edges image
brighter than flat faces. This is exploited industrially in CD-SEM metrology, where the
rim position *is* the measurement.
→ used in: `apply_edge_brightening()`; constants `EDGE_GAIN_REF = 0.25`, `EDGE_GAIN_SEARCH = 0.20`
→ refs: JEOL SEM glossary ("edge effect"); ETH Zürich SE-imaging lecture notes;
CD-SEM measurement patents
*(TODO Het: exact URLs / patent numbers)*

## Structure geometry

**[S5] DRAM 6F² open-bitline geometry: word-line pitch 3F, bit-line pitch 2F, line width 1F, orthogonal grid, contacts at intersections.**
The 6F² cell area follows directly from the 3F × 2F pitch product; the via/contact sits at
each word-line × bit-line crossing.
→ used in: `render_dram_world()`; constants `WL_PITCH_F=3`, `BL_PITCH_F=2`, `WL_WIDTH_F=1`,
`BL_WIDTH_F=1`, `VIA_RADIUS_F=0.4`
→ refs: DRAM cell-architecture overviews; US patents covering 6F² open-bitline layout
*(TODO Het: refs)*

**[S9] DRAM arrays are organised as subarray mats separated by sense-amplifier stripes and wordline-driver / decoder regions; block boundaries are visible structure at field scale.**
A memory array is not a uniform infinite lattice. Mats are bounded by sense-amp stripes
(horizontal) and wordline-driver/decoder columns (vertical), and a 1000×1000 px field at
10× spans many mats — so those boundaries are in-frame structure. This is what makes
localization well-posed while individual cells stay locally periodic.
→ used in: `apply_superstructure()` (SPEC_AMENDMENT_v1.1 §A.1–2); constants
`SA_PITCH_WL_RANGE`, `SA_WIDTH_F_RANGE`, `SA_INTENSITY`, `DR_PITCH_BL_RANGE`,
`DR_WIDTH_F_RANGE`, `DR_INTENSITY`
→ why it matters: the v1.0 uniform-lattice world was measurably unlocalizable — a
noiseless, unwarped, exact-scale match ranked the true site ~2079th of 855k. The fix is
this superstructure, **not** raising `LINE_INTENSITY_JITTER` (values ≥0.15 are unphysical).
→ refs: DRAM architecture literature (memory-systems texts; subarray-level DRAM
organization papers)
*(TODO Het: exact titles/DOIs)*

**[S12] Real DRAM floorplans are not exactly periodic at field scale: redundancy / spare rows and columns, edge mats, and bank boundaries break mat periodicity.**
A floorplan that repeated exactly would make every mat cell interchangeable. Real arrays
carry spare rows/columns for repair, irregular edge mats, and bank boundaries, so the
sequence of mat spacings is effectively a unique code across the die.
→ used in: `_stripe_starts()` irregular spacing (`MAT_JITTER`, `--mat-jitter`), per-stripe
width/intensity draws, and the bank-boundary stripe (`BANK_PROB`, `BANK_WIDTH_F_RANGE`,
`BANK_INTENSITY`) — SPEC_AMENDMENT_v1.2 §A
→ measured: with v1.1's strictly regular mats *pitched as an integer multiple of the lattice*,
false peaks landed on exact integer multiples of the stripe pitch (rank median 30). Moving to
absolute pitch bases incommensurate with the lattice pitch is what removes the joint
periodicity; the spacing jitter hardens it further.
→ refs: DRAM redundancy / repair literature; memory floorplanning texts
*(TODO Het: exact titles/DOIs)*

**[S13] Standard-cell logic is organised into fixed-height rows separated by row-boundary bands, with diffusion breaks between cells and dummy gates at cell edges; the contacted poly pitch (CPP) is regular by construction.**
A fin field on its own is a 1-D grating -- fins repeat along one axis and are uniform along the
other -- so it carries no information at all about position along the fin direction. Real logic is
not like that: cells are stacked in rows of fixed height, the row boundaries carry power rails and
n-well edges, the active region is cut by diffusion breaks between cells, and dummy gates sit at
cell edges. The rows, breaks and dummies vary; the gate pitch does not.
→ used in: `render_finfet_world()` row boundaries (`ROW_BASE_RANGE`, `ROW_JITTER`,
`ROW_WIDTH_RANGE_PX`, `ROW_INTENSITY`), diffusion breaks (`DIFF_BREAK_RATE`,
`DIFF_BREAK_WIDTH_RANGE_PX`), dummy-gate doublets (`DUMMY_GATE_PROB`) -- SPEC_AMENDMENT_v1.5
→ measured: before v1.5, FinFET trailed DRAM by ~10 points at every noise level, every failure was
pure-y, and the correlation surface degenerated into horizontal ridges spanning the frame. After
v1.5 the y-ambiguity flag rate on FinFET fields is 0% and FinFET outperforms DRAM.
→ refs: standard-cell layout / place-and-route texts; FinFET design-rule literature on diffusion
breaks and dummy-gate insertion
*(TODO Het: exact titles/DOIs -- this one needs real references before submission)*

**[S10] Contamination particles are a standard artifact in SEM-based wafer inspection.**
→ used in: `add_defects()` (SPEC_AMENDMENT_v1.1 §A.3); constants `DEFECT_RATE`,
`DEFECT_SIGMA_RANGE`, `DEFECT_AMPLITUDE_RANGE`, `DEFECT_MIN_CROSSING_DIST`
*(TODO Het: refs)*

**[S11] Stage navigation lands a short drift from the intended site, so the target sits near the frame centre.**
The tool drives to the nominal site and captures the search image where it actually landed;
the residual is stage/navigation error, not an arbitrary offset. This is the physical basis
for the official rule "if multiple matching regions are found, return the one closest to the
centre of the Search Image" — it is a prior, not a tie-break of convenience.
→ used in: `_place_by_drift()` (SPEC_AMENDMENT_v1.1 §B); constants `DRIFT_SIGMA = 120`,
`DRIFT_CAP = 350` (search px)
→ **provisional:** the organizers' own generator parameters are unknown until the official
starter code drops (4 Aug webinar). Re-align these defaults then — one-line change — and
note it in README.
*(TODO Het: refs on stage positioning repeatability / navigation error in wafer inspection tools)*

## Algorithm (Phase 2–3)

**[S6] NCC score degrades under small rotation / scale change → motivates the multi-scale, multi-rotation sweep.**
→ used in: `localize.py` scale × rotation sweep (Phase 3)
→ refs: machine-vision pattern-matching literature and patents
*(TODO Het: refs)*

**[S7] Phase correlation / parabolic peak interpolation for sub-pixel registration.**
Fitting a quadratic to the 3×3 neighbourhood of the correlation peak recovers the
fractional offset.
→ used in: `localize.py` sub-pixel refinement (Phase 3)
*(TODO Het: refs)*

**[S8] Multi-scale template matching plus edge-preserving filtering is the industrial approach for wafer alignment.**
→ used in: overall `localize.py` pipeline design
*(TODO Het: refs)*

---

### Index

| Tag | Claim | Code |
|-----|-------|------|
| S1  | Poisson shot noise dominates | `apply_sensor_noise` |
| S2  | Mixed Poisson-Gaussian, Var = a·x + b² | `apply_sensor_noise` |
| S3  | Line-scan correlated noise | `apply_sensor_noise(scanline=True)` |
| S4  | SEM edge-brightening | `apply_edge_brightening` |
| S5  | DRAM 6F² lattice ratios | `render_dram_world` |
| S6  | NCC degrades under rotation/scale | `localize.py` (Phase 3) |
| S7  | Parabolic sub-pixel peak fit | `localize.py` (Phase 3) |
| S8  | Multi-scale matching for wafer alignment | `localize.py` (Phase 2–3) |
| S9  | Subarray mats / sense-amp + driver stripes | `apply_superstructure` |
| S10 | SEM contamination particles | `add_defects` |
| S11 | Navigation drift centres the target | `_place_by_drift` |
| S12 | Floorplans are aperiodic at field scale | `_stripe_starts`, bank stripe |
| S13 | Standard-cell rows, diffusion breaks, dummy gates | `render_finfet_world` |
