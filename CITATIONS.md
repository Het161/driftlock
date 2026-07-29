# CITATIONS.md — physics justification ledger · Team DriftLock
### Every generator constant and algorithm choice maps to an [S#] tag in code.
### Rule from the problem statement: each augmentation/noise/structural choice needs 2–3
### credible references. This file is the master list; the deck's Slide 9 mirrors it.

---

## [S1] SEM noise is Poisson-dominated shot noise, not plain Gaussian
Used in: `generate_dataset.py` — Poisson term of the sensor model.
1. Mulapudi S., Joy D., "Is SEM Noise Gaussian?", Microscopy and Microanalysis 9(S02), 2003.
2. Timischl F. et al., "A statistical model of signal–noise in scanning electron microscopy," Scanning 34(3), 2012.
3. Goldstein J. et al., "Scanning Electron Microscopy and X-Ray Microanalysis," Springer, 2003 (textbook).

## [S2] Mixed Poisson–Gaussian detector model: y = a·Poisson(x/a) + N(0, b²)
Used in: `generate_dataset.py` §3.4 — shot noise + Gaussian readout, Var = a·x + b².
1. Mannam V., Zhang Y. et al., "Real-time image denoising of mixed Poisson–Gaussian noise in fluorescence microscopy images using ImageJ," Optica 9(4):335, 2022. https://opg.optica.org/optica/fulltext.cfm?uri=optica-9-4-335
2. Luisier F., Blu T., Unser M., "Image denoising in mixed Poisson–Gaussian noise" (PURE-LET), IEEE Trans. Image Processing 20(3), 2011.
3. Roels J. et al., "Bayesian deconvolution of scanning electron microscopy images…," 2018. https://arxiv.org/pdf/1810.09739 — states SEM noise = signal-dependent Poisson + signal-independent Gaussian.

## [S3] Scan-line correlated noise in SEM acquisition
Used in: `generate_dataset.py` — row-correlated noise term (search image).
1. Roels J. et al. (2018, above) — model y = Hx + D(x)Cn where circulant C explicitly models line-scanning noise correlation.
2. Sim K.S., Nia M.E., Tso C.P., "Image noise cross-correlation for signal-to-noise ratio estimation in scanning electron microscope images," Scanning 33(2):82–93, 2011.
3. (Supporting) Maraghechi S. et al., "Correction of scanning electron microscope imaging artifacts…," Exp. Mechanics, 2019 — scan-line shift artifacts in SEM. https://pmc.ncbi.nlm.nih.gov/articles/PMC6541586/

## [S4] Edge effect: secondary-electron yield rises at edges → bright rims
Used in: `generate_dataset.py` §3.2 — gradient-based edge-brightening before noise.
1. JEOL SEM Glossary, "Edge Effect" — SE emission increases remarkably at edges/protrusions.
2. Goldstein et al. 2003 (textbook, topographic contrast chapter).
3. Reimer L., "Scanning Electron Microscopy: Physics of Image Formation and Microanalysis," Springer, 1998.
4. (Industry relevance) CD-SEM metrology exploits this edge signal for critical-dimension measurement — CD-SEM measurement patents. [link-verify: pick one US patent]

## [S5] DRAM 6F² geometry: WL pitch 3F, BL pitch 2F, orthogonal grid, contacts at intersections
Used in: `generate_dataset.py` §3.1 — lattice pitch ratios.
1. US Patent 7,349,232 B2 (Micron), "6F² DRAM cell design with 3F-pitch folded digitline sense amplifier." https://patents.google.com/patent/US7349232B2/en
2. "Under the Hood: DRAM architectures: 8F² vs. 6F²," EDN/EE Times, 2008. https://www.edn.com/under-the-hood-dram-architectures-8f2-vs-6f2/
3. US Patent 4,888,732, "Dynamic random access memory having open bit line architecture."

## [S6] Plain NCC degrades under small rotation/scale change → multi-scale × rotation sweep
Used in: `localize.py` — the sweep over scale 9.6–10.4 and rotation ±2°.
1. Lewis J.P., "Fast Normalized Cross-Correlation," Vision Interface, 1995.
2. Brown L.G., "A Survey of Image Registration Techniques," ACM Computing Surveys 24(4), 1992.

## [S7] Phase correlation, sub-pixel peak refinement, PSR confidence
Used in: `localize.py` — sub-pixel fit and peak-to-sidelobe confidence.
1. Kuglin C.D., Hines D.C., "The Phase Correlation Image Alignment Method," Proc. IEEE Int. Conf. Cybernetics and Society, 1975.
2. Guizar-Sicairos M., Thurman S.T., Fienup J.R., "Efficient subpixel image registration algorithms," Optics Letters 33(2):156–158, 2008.
3. Bolme D.S. et al., "Visual Object Tracking using Adaptive Correlation Filters" (MOSSE — defines peak-to-sidelobe ratio), CVPR 2010.

## [S8] Multi-scale template matching + noise-robust filtering is the industrial wafer-alignment approach
Used in: overall localizer design rationale.
1. Basler AG, machine-vision application literature on wafer alignment via multi-scale template matching with edge-preserving filtering. [link-verify]
2. Phase-only correlation alignment patents in semiconductor positioning. [link-verify: one US/JP patent]

## [S9] DRAM arrays are organized as subarray mats separated by sense-amp and driver regions
Used in: `generate_dataset.py` — superstructure stripes (sense-amp horizontal, driver vertical).
1. "A multigigabit DRAM technology with 6F² open-bitline cell, distributed overdriven sensing, and stacked-flash fuse," IEEE J. Solid-State Circuits / IEDM, 2001 — 512 kb arrays of 1024 WLs × 512 BLs, tiled 16×16 per quadrant.
2. EDN/EE Times 2008 (above) — array block = cell array + bitline sense amps; 320 wordlines per block.
3. Kim Y. et al., "A Case for Exploiting Subarray-Level Parallelism (SALP) in DRAM," ISCA 2012.
4. Jacob B., Ng S., Wang D., "Memory Systems: Cache, DRAM, Disk," Morgan Kaufmann, 2007 (textbook).

## [S10] Contamination particles are a standard artifact in SEM-based wafer inspection
Used in: `--defects` blobs.
1. Goldstein et al. 2003 — specimen contamination in SEM.
2. Wafer-inspection defect-review literature (particle defects). [link-verify: one review/appnote]

## [S11] Drift prior: the tool lands near the target; drift is small relative to the field
Used in: `generate_dataset.py` §B — drift-centred placement; also why the official centre rule exists.
1. Primary source: official problem statement (Applied Materials, SEMICON India Hackathon 2026) — "motion stages accumulate tiny errors… it may land several pixels away," and the closest-to-centre tie-break rule.
2. Maraghechi et al. 2019 (above) — drift distortion as a standard SEM artifact category.

## [S12] Real DRAM floorplans are not exactly periodic at field scale (redundancy, bank edges)
Used in: `generate_dataset.py` — incommensurate/irregular mat pitches, bank-boundary stripe.
Measured basis: our own ablation — commensurate pitches (integer × lattice pitch) make the joint
pattern shift-invariant (median rank 78); incommensurate pitches solve it (rank 0, 0.43 px).
1. EE Times 2008 (above) — row redundancy in 6F² designs (spare rows exist and vary block content).
2. Keeth B., Baker R.J., "DRAM Circuit Design: Fundamental and High-Speed Topics," IEEE Press/Wiley — redundancy and array-edge structures.
3. Jacob et al. 2007 (above) — bank/subarray organization.

## [S13] FinFET logic fields carry standard-cell-row structure: rows, diffusion breaks, dummy gates
Used in: v1.5 FinFET generator — row-boundary bands, irregular diffusion-break gaps, dummy-gate
doublets; the gate pitch (CPP) itself stays regular, as in real logic.
1. Clark L.T. et al., "ASAP7: A 7-nm finFET predictive process design kit," Microelectronics Journal 53:105–115, 2016. https://doi.org/10.1016/j.mejo.2016.04.006 — standard-cell architecture, double diffusion breaks with fin cuts under dummy gates, regular CPP, 27 nm fin pitch.
2. Vashishtha V., Clark L.T., "Design Flows and Collateral for the ASAP7 7-nm FinFET PDK" — dummy cells complete the gate grid; tap cells on a 2-CPP pitch. https://pages.hmc.edu/harris/research/asap7.pdf
3. Arm, "Standard Cell Design and Optimization Methodology," ICCAD 2017 tutorial — ASAP7 geometry: 54 nm gate pitch, 27 nm fin pitch.
