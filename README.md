<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&height=190&color=0:0B1533,50:1E3A8A,100:F0A93B&text=DriftLock&fontColor=FFFFFF&fontSize=64&animation=fadeIn&desc=Navigation-Error%20Recovery%20for%20Wafer%20Inspection%20Tools&descSize=18&descAlignY=72" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=17&duration=2800&pause=900&color=F0A93B&center=true&vCenter=true&width=640&lines=Find+one+site+in+a+sea+of+identical+patterns;CPU-only+%C2%B7+sub-second+%C2%B7+no+model+weights;SEM+physics+in%2C+(x%2C+y)+%2B+confidence+out" alt="typing"/>

![Python](https://img.shields.io/badge/Python-3.10+-1E3A8A?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-classical_CV-F0A93B)
![NumPy](https://img.shields.io/badge/NumPy-physics_sim-1E3A8A)
![CPU only](https://img.shields.io/badge/GPU-not_needed-2ea44f)
![SEMICON India 2026](https://img.shields.io/badge/SEMICON_India-Hackathon_2026-F0A93B)

**Applied Materials PS02 · Team DriftLock · Gandhinagar University**

</div>

---

## What this is

A wafer inspection tool drifts. It lands near a target, not on it. Our job: given a clean
**reference** image (100x) and a noisy **search** image (10x), return the exact centre `(x, y)`
where the reference sits inside the search frame.

No dataset exists for this. So we build our own, with real SEM physics in it. Then we localize
with classical multi-scale matching. No deep learning. Nothing that can break.

<p align="center"><img src="docs/assets/pair_preview.png" width="720" alt="generated pair preview"/></p>

## Quickstart

```bash
pip install -r requirements.txt
python generate_dataset.py --num-pairs 30 --output-dir data --seed 42
python localize.py --reference data/pairs/pair_0001_ref.png --search data/pairs/pair_0001_search.png
python evaluate.py --data-dir data --tolerance 5 --report-dir report
```

`localize.py` prints one line: `x y`. That is the whole contract.

## Architecture

```mermaid
flowchart LR
    G["generate_dataset.py<br/>SEM-physics generator"] -->|"ref + search + ground truth"| D[("data/")]
    D --> L["localize.py<br/>the scored script"]
    L -->|"(x, y) + PSR confidence"| E["evaluate.py"]
    E --> R["report/<br/>metrics + figures"]
    classDef amber fill:#F0A93B,stroke:#F0A93B,color:#0B1533
    classDef navy fill:#0B1533,stroke:#7A8BB8,color:#fff
    class L amber
    class G,D,E,R navy
```

**How one pair is made** (each capture gets independent noise):

```mermaid
flowchart LR
    W["10,000 px<br/>die world"] --> C["crop 1000 px<br/>= reference"]
    W --> A["rotate ±2° · scale ±3%<br/>shrink 10x = search"]
    C & A --> P["edge-brighten → blur →<br/>Poisson + Gaussian noise"]
    P --> O["pair + true (x, y)"]
    classDef navy fill:#0B1533,stroke:#7A8BB8,color:#fff
    class W,C,A,P,O navy
```

**Localization pipeline:**

```mermaid
flowchart LR
    S1[clean up] --> S2["ZNCC sweep<br/>scale 9.6–10.4 · rot ±2°"] --> S3[peak set] --> S4["centre rule<br/>(official tie-break)"] --> S5[sub-pixel fit] --> S6["(x, y) + PSR"]
    classDef amber fill:#F0A93B,stroke:#F0A93B,color:#0B1533
    classDef navy fill:#0B1533,stroke:#7A8BB8,color:#fff
    class S6 amber
    class S1,S2,S3,S4,S5 navy
```

## We measured our data before trusting it

| World model | Median rank of truth | Result |
|---|---|---|
| Pure lattice | 583 | lost |
| Regular mats (pitch = lattice multiple) | 46 | lost |
| **Incommensurate mats (ours)** | **0** | **0.42 px error** |

Reproduce it: `python ablation_gate.py --n-pairs 24`

<details>
<summary><b>Why did regular mats fail?</b></summary>

Stripe pitch was a whole multiple of the lattice pitch. Shift the frame by one stripe and the
picture is identical, so NCC cannot tell the difference. Our fix uses incommensurate pitches:
the two periods never line up again inside the frame. `--pure-lattice` regenerates the broken
world on demand. It is our honest-failure exhibit.
</details>

<details>
<summary><b>Why classical, not deep learning?</b></summary>

The judges run `localize.py` as-is. A deterministic CPU method has no weights to download, no
CUDA to match, nothing to break. It also answers in under a second.
</details>

## Repo map

| File | Job |
|---|---|
| `generate_dataset.py` | make pairs + ground truth (`--style dram/finfet`, `--pure-lattice`, `--commensurate-mats`, `--seed`) |
| `localize.py` | **scored script** — two image paths in, `x y` out |
| `evaluate.py` | accuracy @5/@10 px, runtime, success + honest-failure figures |
| `ablation_gate.py` | the three-tier data ablation, exit 0 = pass |
| `smoke_test.py` | fresh-machine sanity, 60 s |
| `common.py` | shared I/O, seeding and affine helpers |
| `requirements.txt` | pinned runtime dependencies |
| `CITATIONS.md` | every physics constant → 2–3 references ([S1]–[S13]) |
| `docs/` | spec + amendment trail (how this design was reached) |
| `docs/results/` | committed figures and `results.csv` from the final run |

## Results

About **91%** of predictions land within 5 px on mixed DRAM + FinFET pairs at medium noise
(94% across all 180 standard pairs · median error **0.08 px** · ~**660 ms** per pair on a
laptop CPU). On deliberately degenerate pure-lattice fields the system degrades gracefully
to the drift prior (median error 73 px ≈ the prior's median) and flags **both axes 100% of
the time**. It identifies degenerate fields; it does not claim to know when an individual
answer is wrong.

<p align="center"><img src="docs/results/success_case.png" width="420" alt="success"/> <img src="docs/results/failure_case.png" width="420" alt="honest failure"/></p>

Full tables: `docs/results/results.csv` · robustness curves: `docs/results/robustness_noise.png`

## Team

**DriftLock** — Het Sanjaykumar Patel (algorithm & generator) · Eklavya DilipBhai Jha
(evaluation & visualization) · Gandhinagar University

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&height=100&color=0:F0A93B,100:0B1533&section=footer" width="100%"/>
</div>
