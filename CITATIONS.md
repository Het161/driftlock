# CITATIONS.md — physics justification ledger · Team DriftLock
### Every generator constant and algorithm choice maps to an [S#] tag in code.
### Rule from the problem statement: each augmentation/noise/structural choice needs 2–3
### credible references. This file is the master list; the deck's Slide 9 mirrors it.

> **Verification status.** Every reference below was fetched and checked against its
> publisher, DOI or patent record — titles, authors, years, volumes and pages. Entries that
> did not check out were replaced, not kept; what changed is recorded in
> "Verification log" at the foot of this file. Four items marked *(unverified)* are
> well-known textbooks and one tutorial whose bibliographic details were not independently
> confirmed; treat those as the weakest links.

---

## [S1] SEM noise is Poisson-dominated shot noise, not plain Gaussian
Used in: `generate_dataset.py` — Poisson term of the sensor model.
1. Timischl F., Date M., Nemoto S., "A statistical model of signal–noise in scanning electron microscopy," *Scanning* 34(3):137–144, 2012. DOI [10.1002/sca.20282](https://doi.org/10.1002/sca.20282) — models the detector signal as a cascade of five conversion stages, every one of which is a quantum fluctuation described by Poisson statistics.
2. Roels J., Aelterman J., De Vylder J., Luong H., Saeys Y., Philips W., "Bayesian Deconvolution of Scanning Electron Microscopy Images Using Point-spread Function Estimation and Non-local Regularization," 2018. [arXiv:1810.09739](https://arxiv.org/abs/1810.09739) — states directly: *"noise should be modeled as a composition of signal-dependent (i.e. Poisson) and signal-independent (Gaussian) noise."*
3. Reimer L., *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*, 2nd ed., Springer Series in Optical Sciences 45, 1998, 527 pp. DOI [10.1007/978-3-540-38967-5](https://doi.org/10.1007/978-3-540-38967-5) — shot noise in the primary and secondary electron signal.

> **Contested, and we say so.** Prasad M.S. & Joy D.C., "Is SEM Noise Gaussian?", *Microscopy and Microanalysis* 9(S02):982–983, 2003, DOI [10.1017/S1431927603444917](https://doi.org/10.1017/S1431927603444917), reports that in a thermionic-gun SEM the *final image* noise follows Gaussian rather than Poisson statistics. That is a genuine counterpoint: the electron-counting process is Poisson, but the detector chain that follows it can drive the output toward Gaussian. Our model contains both terms, so it spans the disagreement rather than picking a side.

## [S2] Mixed Poisson–Gaussian detector model: y = a·Poisson(x/a) + N(0, b²)
Used in: `generate_dataset.py` §3.4 — shot noise + Gaussian readout, Var = a·x + b².
1. Mannam V., Zhang Y., Zhu Y., Nichols E., Wang Q., Sundaresan V. et al., "Real-time image denoising of mixed Poisson–Gaussian noise in fluorescence microscopy images using ImageJ," *Optica* 9(4):335–345, 2022. DOI [10.1364/OPTICA.448287](https://doi.org/10.1364/OPTICA.448287)
2. Luisier F., Blu T., Unser M., "Image Denoising in Mixed Poisson–Gaussian Noise" (PURE-LET), *IEEE Transactions on Image Processing* 20(3):696–708, 2011. DOI [10.1109/TIP.2010.2073477](https://doi.org/10.1109/TIP.2010.2073477) *(published online 2010, appears in the 2011 issue)*
3. Roels J. et al. 2018 (above) — the observation model is **y = Hx + D(x)Cn** with `(D(x))ᵢᵢ = √(σ² + αxᵢ)`, which is exactly the signal-dependent-plus-constant variance we implement.

## [S3] Scan-line correlated noise in SEM acquisition
Used in: `generate_dataset.py` — row-correlated noise term (search image only).
1. Roels J. et al. 2018 (above) — in **y = Hx + D(x)Cn**, **C** is a circulant matrix modelling *"the effects of line scanning noise correlation"*. This is the direct warrant for a row-correlated term.
2. Sim K.S., Nia M.E., Tso C.P., "Image noise cross-correlation for signal-to-noise ratio estimation in scanning electron microscope images," *Scanning* 33:82–93, 2011. DOI [10.1002/sca.20223](https://doi.org/10.1002/sca.20223)
3. Maraghechi S., Hoefnagels J.P.M., Peerlings R.H.J., Rokoš O., Geers M.G.D., "Correction of Scanning Electron Microscope Imaging Artifacts in a Novel Digital Image Correlation Framework," *Experimental Mechanics* 59(4):489–516, 2019. DOI [10.1007/s11340-018-00469-w](https://doi.org/10.1007/s11340-018-00469-w) — treats **scan line shifts** as one of the three dominant SEM artifact classes, alongside spatial and drift distortion.

## [S4] Edge effect: secondary-electron yield rises at edges → bright rims
Used in: `generate_dataset.py` §3.2 — gradient-based edge-brightening before noise.
1. JEOL Ltd., SEM Glossary, ["edge effect"](https://www.jeol.com/words/semterms/20121024.012800.php) — *"the tip of a protrusion and the edge of a step on a specimen surface become extremely bright… secondary electrons are emitted much more from the tip of a protrusion and the edge of a step than those from flat regions."*
2. Reimer L. 1998 (above) — topographic contrast and edge brightening, physics of SE escape.
3. Goldstein J. et al., *Scanning Electron Microscopy and X-Ray Microanalysis*, 3rd ed., Springer, 2003. DOI [10.1007/978-1-4615-0215-9](https://doi.org/10.1007/978-1-4615-0215-9) *(4th ed. 2018, ISBN 978-1-4939-6674-5)*
4. **Industrial use — CD-SEM.** US Patent 9,200,896 B2 (Hitachi High-Tech; Hitomi, Nakayama, Tanaka; granted 1 Dec 2015), "Pattern dimension measurement method and charged particle beam microscope used in same": *"Since the amount of secondary electrons emitted is increased at pattern edge portions, which is called edge effect, a band-like region called a white band is observed at a location corresponding to a pattern edge portion in an image obtained by the CD-SEM"* — and that white band **is** the measurement. See also Hitachi High-Tech, ["What is a Critical Dimension SEM?"](https://www.hitachi-hightech.com/global/en/knowledge/semiconductor/room/manufacturing/cd-sem.html)

## [S5] DRAM 6F² geometry: WL pitch 3F, BL pitch 2F, orthogonal grid, contacts at intersections
Used in: `generate_dataset.py` §3.1 — lattice pitch ratios.
1. US Patent 7,349,232 B2 (Micron Technology; Fei Wang, Anton P. Eppich; granted 25 Mar 2008), ["6F2 DRAM cell design with 3F-pitch folded digitline sense amplifier"](https://patents.google.com/patent/US7349232B2/en)
2. "Under the Hood: DRAM architectures: 8F² vs. 6F²," *EDN*, 22 Feb 2008. https://www.edn.com/under-the-hood-dram-architectures-8f2-vs-6f2/ — 6F² gives ~25% cell-area improvement and forces open-bitline architecture because of the tight sense-amplifier pitch.
3. US Patent 4,888,732 (Matsushita Electric / Panasonic; Michihiro Inoue, Toshio Yamada; granted 19 Dec 1989), ["Dynamic random access memory having open bit line architecture"](https://patents.google.com/patent/US4888732A/en)

## [S6] Plain NCC degrades under small rotation/scale change → multi-scale × rotation sweep
Used in: `localize.py` — the sweep over scale 9.6–10.4 and rotation ±2°.
1. Lewis J.P., "Fast Normalized Cross-Correlation," *Vision Interface*, 1995, pp. 120–123. [PDF](https://scribblethink.org/Work/nvisionInterface/nip.pdf)
2. Brown L.G., "A Survey of Image Registration Techniques," *ACM Computing Surveys* 24(4):325–376, 1992. DOI [10.1145/146370.146374](https://doi.org/10.1145/146370.146374)

## [S7] Phase correlation, sub-pixel peak refinement, PSR confidence
Used in: `localize.py` — sub-pixel fit and peak-to-sidelobe confidence.
1. Kuglin C.D., Hines D.C., "The Phase Correlation Image Alignment Method," *Proc. IEEE Int. Conf. on Cybernetics and Society*, 1975, pp. 163–165.
2. Guizar-Sicairos M., Thurman S.T., Fienup J.R., "Efficient subpixel image registration algorithms," *Optics Letters* 33(2):156–158, 2008. DOI [10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156)
3. Bolme D.S., Beveridge J.R., Draper B.A., Lui Y.M., "Visual Object Tracking using Adaptive Correlation Filters" (MOSSE), *CVPR* 2010. DOI [10.1109/CVPR.2010.5539960](https://doi.org/10.1109/CVPR.2010.5539960) — defines the peak-to-sidelobe ratio and uses it to detect occlusion, i.e. as an "am I still tracking the right thing" test. That is precisely our use.

## [S8] Multi-scale template matching is the industrial wafer-alignment approach
Used in: overall localizer design rationale.
1. US Patent 8,538,168 B2 (Raintree Scientific Instruments Shanghai; Lisong Liu; granted 17 Sep 2013), ["Image pattern matching systems and methods for wafer alignment"](https://patents.google.com/patent/US8538168B2/en) — applies *"a normalized cross-correlation (NCC) algorithm"* and a coarse-to-fine search: a rough search on down-sampled images to find a candidate area, then a fine search at full resolution.
2. US Patent 9,057,873 B2 (Hitachi High-Tech; Miyamoto, Hosoya, Kawahara, Onizawa; granted 16 Jun 2015), ["Global alignment using multiple alignment pattern candidates"](https://patents.google.com/patent/US9057873B2/en) — ranks multiple template candidates and falls back through them: *"if the matching fails in any one of the alignment pattern candidates but succeeds in any other alignment pattern candidate, the global alignment can succeed."* The industrial answer to ambiguity is more candidates plus a tie-break, which is the shape of our Phase-3 design.
3. Lewis J.P. 1995 (above) — the NCC formulation both patents rest on.

## [S9] DRAM arrays are organized as subarray mats separated by sense-amp and driver regions
Used in: `generate_dataset.py` — superstructure stripes (sense-amp horizontal, driver vertical).
1. Takahashi T., Sekiguchi T. et al., "A multigigabit DRAM technology with 6F² open-bitline cell, distributed overdriven sensing, and stacked-flash fuse," *IEEE Journal of Solid-State Circuits*, 2001. [IEEE Xplore 962294](https://ieeexplore.ieee.org/document/962294/) · [free PDF](https://www.bioee.ee.columbia.edu/courses/ee6321/papers/00962294.pdf) — a 6F² open-bitline array tiled into blocks, with edge arrays fully utilised.
2. Kim Y., Seshadri V., Lee D., Liu J., Mutlu O., "A Case for Exploiting Subarray-Level Parallelism (SALP) in DRAM," *ISCA* 2012 — a DRAM bank is physically many subarrays, each with its own local sense amplifiers.
3. *EDN* 2008 (above) — an array block comprises the cell array plus its bitline sense amplifiers.
4. Jacob B., Ng S., Wang D., *Memory Systems: Cache, DRAM, Disk*, Morgan Kaufmann, 2007 *(unverified)*.

## [S10] Contamination particles are a standard artifact in SEM-based wafer inspection
Used in: `--defects` blobs.
1. Hitachi High-Tech, ["What is a Review SEM?"](https://www.hitachi-hightech.com/global/en/knowledge/semiconductor/room/manufacturing/review-sem.html) — defect inspection detects particles and pattern anomalies; the review SEM re-images them at high magnification for automatic defect review and classification.
2. "A Review on Machine and Deep Learning for Semiconductor Defect Classification in Scanning Electron Microscope Images," *Applied Sciences* 11(20):9508, 2021. DOI [10.3390/app11209508](https://doi.org/10.3390/app11209508) — particle contamination as a primary defect class in SEM wafer inspection *(record from index; publisher page blocked automated fetch)*.
3. Goldstein J. et al. 2003 (above) — specimen contamination in the SEM.

## [S11] Drift prior: the tool lands near the target; drift is small relative to the field
Used in: `generate_dataset.py` §B — drift-centred placement; also why the official centre rule exists.
1. **Primary source:** official problem statement (Applied Materials, SEMICON India Hackathon 2026) — motion stages accumulate small errors and may land several pixels away, and the stated tie-break is to return the match closest to the centre of the search image.
2. Maraghechi S. et al. 2019 (above) — **drift distortion** as one of the three dominant SEM artifact classes.

## [S12] Real DRAM floorplans are not exactly periodic at field scale (redundancy, bank edges)
Used in: `generate_dataset.py` — incommensurate/irregular mat pitches, bank-boundary stripe.
**Measured basis (our own ablation, `ablation_gate.py --n-pairs 24`):** commensurate pitches — stripe
pitch an integer multiple of the lattice pitch — make the joint pattern shift-invariant and the true
site drops to **median rank 46**; incommensurate pitches recover **rank 0 at 0.42 px median error**.
A pure lattice with no superstructure at all sits at median rank 583.
1. *EDN* 2008 (above) — row redundancy in 6F² designs, so block content is not identical across the die.
2. Keeth B., Baker R.J., *DRAM Circuit Design: Fundamental and High-Speed Topics*, IEEE Press/Wiley *(unverified)* — redundancy and array-edge structures.
3. Jacob B. et al. 2007 (above) *(unverified)* — bank/subarray organization.

## [S13] FinFET logic fields carry standard-cell-row structure: rows, diffusion breaks, dummy gates
Used in: v1.5 FinFET generator — row-boundary bands, irregular diffusion-break gaps, dummy-gate
doublets; the gate pitch (CPP) itself stays regular, as in real logic.
1. Clark L.T., Vashishtha V., Shifren L., Gujja A., Sinha S., Cline B., Ramamurthy C., Yeric G., "ASAP7: A 7-nm finFET predictive process design kit," *Microelectronics Journal* 53:105–115, 2016. DOI [10.1016/j.mejo.2016.04.006](https://doi.org/10.1016/j.mejo.2016.04.006) — standard-cell architecture, double diffusion breaks with fin cuts under dummy gates, regular CPP, 27 nm fin pitch.
2. Clark L.T., Vashishtha V., Harris D.M., Dietrich S., Wang Z., "Design flows and collateral for the ASAP7 7nm FinFET predictive process design kit," *IEEE Int. Conf. on Microelectronic Systems Education (MSE)*, 2017, pp. 1–4. DOI [10.1109/MSE.2017.7945071](https://doi.org/10.1109/MSE.2017.7945071) · [PDF](https://pages.hmc.edu/harris/research/asap7.pdf) — dummy cells complete the gate grid; tap cells on a 2-CPP pitch.
3. Arm, "Standard Cell Design and Optimization Methodology," ICCAD 2017 tutorial *(unverified)* — ASAP7 geometry: 54 nm gate pitch, 27 nm fin pitch.

---

## Verification log

Checked on 2026-07-29 by fetching each publisher, DOI or patent record.

**Replaced — did not support the claim:**

| Was | Problem | Now |
|---|---|---|
| [S1].1 "Mulapudi S., Joy D., *Is SEM Noise Gaussian?*" cited **for** Poisson dominance | Author is **M. Satya Prasad**, not "Mulapudi S."; and the paper reports image noise is **Gaussian, not Poisson** — the opposite of the claim it was supporting | Timischl 2012 and Roels 2018 promoted; the Prasad & Joy paper retained as an explicitly labelled **counterpoint** |
| [S4].4 `[link-verify]` "CD-SEM measurement patents" | First candidate found, US 7,335,880, **explicitly avoids** the edge signal (it measures area fraction *because* "the edge effect also may give rise to image artifacts") | US 9,200,896 B2, which states the white band from the edge effect **is** what CD-SEM measures |
| [S8].1 `[link-verify]` "Basler AG application literature" | Vendor marketing, no locatable specific document | US 8,538,168 B2 — verified NCC plus coarse-to-fine wafer alignment |
| [S8].2 `[link-verify]` "phase-only correlation alignment patents" | Too vague to verify | US 9,057,873 B2 — multiple template candidates with fallback |
| [S10].2 `[link-verify]` "defect-review literature" | Placeholder | Hitachi High-Tech Review SEM page + *Applied Sciences* 11(20):9508 |

**Corrected:**

- **[S13].2** author order and title were wrong — cited as "Vashishtha V., Clark L.T., *Design Flows and Collateral for the ASAP7 7-nm FinFET PDK*"; the record is **Clark, Vashishtha, Harris, Dietrich, Wang**, MSE 2017, pp. 1–4.
- **[S12]** measured figures were stale — said "median rank 78 … rank 0, 0.43 px" from an early n=8 run; the final matrix at n=24 gives **rank 46** and **0.42 px**.
- **[S1].3 / [S4].3** Goldstein "Springer, 2003" is the **3rd edition**; the 4th is 2018. Both now stated.
- Volumes, pages and DOIs added throughout where the original gave only a journal and year.

**Verified exactly as cited:** Timischl (Scanning 34(3):137–144); Mannam (Optica 9(4):335); Luisier (IEEE TIP 20(3):696–708); Roels (arXiv:1810.09739, including the quoted noise model); Sim/Nia/Tso (Scanning 33:82–93); Maraghechi (Exp. Mech. 59(4):489–516); US 7,349,232 B2; US 4,888,732; EDN 22 Feb 2008; Lewis (Vision Interface 1995, 120–123); Brown (ACM CSUR 24(4):325–376); Kuglin & Hines (1975, 163–165); Guizar-Sicairos (Opt. Lett. 33(2):156–158); Bolme (CVPR 2010); Clark ASAP7 (Microelectronics Journal 53:105–115); Kim SALP (ISCA 2012); Takahashi (JSSC 2001); Reimer (1998, 2nd ed.); JEOL edge-effect glossary.

**Not independently verified** — flagged inline as *(unverified)*: Jacob/Ng/Wang *Memory Systems* (2007); Keeth & Baker *DRAM Circuit Design*; Arm ICCAD 2017 tutorial; and the *Applied Sciences* record, whose publisher page rejects automated fetching.
