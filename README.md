# Typhoon Extreme-Rainfall Structure Analysis & Rainfall-Field Generation

[English](README.md) | [中文](README.zh-CN.md)

> Mathematical-modeling competition project. Built around the **track–intensity–environment–rainfall** relationship, it combines rainfall-structure analysis, rainfall-field generation for unobserved typhoons, and virtual-scenario simulation into one pipeline.

Using **CMA best-track data (CMABST)** and **GPM IMERG half-hourly precipitation**, the project abstracts typhoon rainfall from raw gridded fields into a set of interpretable structural metrics, then applies machine learning and analog-template transfer to go from *explaining historical patterns* → *generating rainfall for unobserved storms* → *simulating future virtual scenarios*.

---

## The Three Problems

### Problem 1 — Drivers of rainfall structure
- Build a **storm-relative coordinate system** anchored on the typhoon center and motion direction, rotating every rainfall field onto a common reference frame.
- Extract a rainfall-structure metric system: peak intensity, P95/P99 quantile intensity, heavy-rain area, rainfall-centroid offset, front/back & left/right asymmetry, rainband anisotropy, rainband width, four-quadrant contributions, etc.
- Explanatory variables: max wind, central pressure, translation speed and direction, distance to coast, land-cover fraction, terrain relief, etc.
- Identify key drivers with **Spearman correlation + an enhanced Random Forest**.
- **Result**: test-set R² reaches **0.684 / 0.640** for rainfall-centroid offset and rainband anisotropy; rainfall is systematically biased toward the **front-left** of the motion direction (front-left quadrant carries the largest share).

### Problem 2 — Rainfall generation for unobserved typhoons
The 2024 typhoons **KONG-REY** and **MAN-YI** have no matching GPM record, so rainfall simulation is reframed as a **conditional rainfall-field generation** task:
1. Retrieve historical time steps with similar track, intensity, motion and near-coast conditions (Top-K analog templates);
2. Perform **analog-template transfer and weighted fusion** in the storm-relative frame;
3. Correct large-scale spatial structure with **EOF / PCA**;
4. Restore the heavy-rain tail with **extreme-quantile calibration**.
- **Result**: complete half-hourly rainfall sequences for both storms (KONG-REY 421 steps, MAN-YI 553 steps), peak grid intensities of **53.7 / 54.4 mm·h⁻¹**, with the main rain area on the front-left of the track.

### Problem 3 — Virtual-typhoon scenario simulation
Using KONG-REY as the baseline, construct virtual scenarios — **intensified, shifted closer to coast, slowed translation, compound perturbations** — and feed the perturbed variables back into the Problem-2 generator.
- **Result**: the slowed-translation scenario (S4) amplifies fixed-location accumulation and duration the most — peak accumulated rainfall rises from **723.5 mm → 1208.1 mm** and maximum heavy-rain duration from **24.5 h → 39.0 h**, highlighting how a stalling typhoon magnifies extreme-rainfall risk.

### Model validation
A **historical pseudo-missing experiment** treats real GPM-observed typhoons as unknown events and reconstructs them with the same pipeline. After extreme-quantile calibration, P95 error, P99 error, and 10 mm·h⁻¹ heavy-rain-area error drop by **39.7% / 39.4% / 37.1%** relative to the raw template field.

---

## Repository Structure

```
.
├── scripts/        Main analysis pipeline (44 scripts, run in numeric order)
│   ├── 00a–00e_*   Stage 0: parse CMABST tracks, extract target-typhoon tracks, coast features, track plots
│   ├── 02–16_*     Preprocessing: GPM feature extraction, track–rainfall spatiotemporal matching, feature engineering, leakage checks
│   ├── 17–25_*     Problems 1/2: storm-relative frame, Random Forest, analog retrieval, template generation, PCA correction, quantile calibration
│   ├── 26–31_*     Problem 3: virtual-scenario construction, scenario rainfall fields, event metrics & footprint maps
│   └── 32–35_*     Rainband-width diagnostics, typical-typhoon evolution visualization
│   └── rainband_width_utils.py   Shared rainband-width utilities
├── output/         Stage-0 (track preprocessing) artifacts: target-track CSVs and path figures
├── outputs/        Main-pipeline artifacts: figures/, tables/, reports/, and *_qc_report.md QC reports
├── paper.tex       LaTeX source of the modeling paper
├── requirements.txt
└── .gitignore
```

> Note: `output/` (track preprocessing) and `outputs/` (main pipeline) are two separate stage directories, named as they were during development.

---

## Tech Stack

Python 3.9+ · numpy · pandas · scipy · **scikit-learn** (Random Forest) · **rasterio** (GPM / terrain raster I/O) · shapely (land–sea / coast-distance geometry) · matplotlib · tqdm

Methods involved: storm-relative coordinate transforms, Spearman correlation, Random Forest with feature importance, KNN analog retrieval, EOF/PCA structural decomposition, quantile-mapping calibration.

---

## Data (not included in the repo)

Raw and intermediate data total several GB (GPM half-hourly precipitation rasters, ETOPO terrain, generated `.npz` fields, etc.) and are **excluded via `.gitignore`** — they are not distributed with the repository. To reproduce, obtain:

- **CMABST best-track data** (Shanghai Typhoon Institute, CMA)
- **GPM IMERG half-hourly precipitation** (NASA GES DISC)
- **ETOPO 2022** terrain data (NOAA)

Place them under `data/raw/` and `data/processed/` following the path conventions used in the scripts.

## Reproduce

```bash
pip install -r requirements.txt
# After preparing data/ with the raw inputs, run the scripts in order:
python scripts/00a_parse_cmabst.py
python scripts/02_extract_gpm_features.py
# ... continue through scripts/35_*
```

Each script runs standalone and is invoked from the project root; paths are resolved relative to the root.

---

## Paper

`paper.tex` is the full LaTeX source of the modeling paper. Figures are referenced from `outputs/figures` and `outputs/tables`, and it compiles with XeLaTeX. The paper is written in Chinese.

## Note

This is a competition entry. AI-assisted tools were used during development to help with code and document drafting; the model design, method selection, and result analysis were done by the author (see the "AI Tool Usage" section of the paper).
