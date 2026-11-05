# HSA-CP: Hierarchical Sector-Adaptive Conformal Prediction

This repository contains the corrected IEEE BigData 2026 manuscript, the causal
experimental pipeline, locked outputs, and supporting workbooks for HSA-CP.

## Main corrected result

On 121,744 valid one-day-ahead intervals for a 501-security fixed S&P 500
snapshot cohort in 2025, HSA-CP obtains:

- coverage: 0.900455 at a nominal 0.90 level;
- mean interval score: 0.096016 (lowest among eight primary methods);
- ALCE20: 0.047270 (lowest);
- XLCE20: 0.017636 (lowest);
- sector coverage range: 0.897662--0.904000.

Newey--West tests with Holm adjustment and moving-block bootstrap intervals at
block lengths 5, 10, and 20 support an improvement over RegimeCP, ACI,
RollingCP, and GraphCP. Evidence against SplitCP, PControl, and SectorCP is not
robust at all block lengths.

Frozen-configuration sensitivity gives empirical coverage 0.799588, 0.900455,
and 0.951883 at nominal 80%, 90%, and 95%. The full 501-asset calibration
layer requires 2.725 ms per session in the recorded reference benchmark.

## Leakage controls

- The analysis cohort is locked at May 31, 2024 using only January--May 2024
  return availability (at least 99%).
- No 2025 acquisition, delisting, quotation, or missingness field is used for
  security selection.
- Natural post-lock missing observations are retained and never imputed.
- The sector-return feature is leave-one-out for the target asset.
- Standardization and the volatility floor are estimated from training data.
- The ridge penalty is tuned on calendar 2023 only.
- Baseline rates and the HSA-CP architecture/rates are selected on
  June--December 2024 only.
- Date-t outcomes are used only after every date-t interval has been issued.

## Reproduce

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tune_baselines.py
python run_ablations.py
python run_experiments.py
python run_extended_experiments.py
python make_artifacts.py
```

The scripts use `HSA_CP_Analysis_Data_2019_2025.xlsx` in the repository root
and write outputs to `results/`, `tables/`, and `figures/`. The primary run is
deterministic. Moving-block bootstrap seeds are fixed in `run_experiments.py`.

## Manuscript

Upload `paper.tex`, `references.bib`, and the `figures/` directory to the
official IEEE conference template in Overleaf. IEEE BigData 2026 uses
single-blind review, so the author name and affiliation remain visible. The
regular full paper is limited to 10 pages including references. The source
declares:

```latex
\documentclass[conference]{IEEEtran}
```

The included PDF was compiled with a local IEEE-like layout shim because the
official class was unavailable in the local runtime. It is a visual/page-limit
check, not a replacement for the final Overleaf compile with official
`IEEEtran`.

## Public repository and data

This release includes the fixed analysis workbook, the corrected results
workbook, the complete experimental code, and the locked machine-readable
outputs. The source licences and redistribution permissions were reviewed by
the dataset depositor before this public release. Source attribution and
provenance are documented in `DATA_ACCESS_NOTICE.md` and in the workbook.

After creating the GitHub repository, add this sentence to the reproducibility
paragraph in `paper.tex`:

```latex
Code and documented experimental outputs are available at
\url{https://github.com/ACCOUNT/hsa-cp}.
```

Replace `ACCOUNT` with the actual account and compile again. Do not insert a
placeholder URL in the submitted PDF.

## File map

- `paper.tex`, `references.bib`: IEEE manuscript source.
- `run_experiments.py`: primary methods, metrics, HAC tests, and block bootstrap.
- `tune_baselines.py`: validation-only ACI and PControl rate selection.
- `run_ablations.py`: validation/test ablations and validation-only sensitivity.
- `run_extended_experiments.py`: nominal levels, scale ablations, cohort-lock sensitivity, and scalability.
- `make_artifacts.py`: manuscript figures and descriptive tables.
- `HSA_CP_Analysis_Data_2019_2025.xlsx`: corrected analysis panel and audit sheets.
- `HSA_CP_Corrected_Experimental_Results.xlsx`: readable result workbook.
- `results/`: machine-readable locked outputs.
- `tables/`, `figures/`: publication artifacts.

## Scope

The theoretical results are deterministic sector-calibration and clipped-loss
Hedge statements. They do not claim finite-sample conditional coverage for
dependent financial returns. The fixed 2024 constituent snapshot also creates
historical-selection and survivorship limitations; this is not a point-in-time
tradable-index backtest or an investment strategy.
