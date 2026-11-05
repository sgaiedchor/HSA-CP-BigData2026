# Data availability, provenance, and reuse

The public reproducibility package includes
`HSA_CP_Analysis_Data_2019_2025.xlsx`. The workbook contains a documented,
fixed-snapshot research panel of log returns, metadata, market-regime
variables, membership changes, quality checks, and chronological data splits.
The source licences and redistribution permissions were reviewed by the
dataset depositor before release.

The workbook preserves source attribution and should not be interpreted as
transferring ownership of third-party source material. Users must retain the
source notices and comply with the applicable source terms when redistributing
or extending the dataset. The research code and author-generated results are
released separately under the repository licence.

The scripts expect the analysis workbook in the project root. To reproduce the
reported outputs, run:

```bash
python tune_baselines.py
python run_ablations.py
python run_experiments.py
python run_extended_experiments.py
python make_artifacts.py
```

The research outputs are not an investment strategy or trading recommendation.

## Workbook contents

- `README`: dataset scope and cohort-lock protocol.
- `Metadata`: security identifiers, sector information, and availability.
- `Analysis Universe`: the leakage-controlled 501-security cohort.
- `Log Returns`: daily log-return panel from 1 April 2019 to 31 December 2025.
- `Market Regimes`: lagged market and volatility variables used by the study.
- `Membership Changes`: documented post-snapshot index changes.
- `Quality Checks`: completeness and provider-download diagnostics.
- `Data Splits`: training, calibration, validation, and final-test dates.

## Citation

When using the data, code, or protocol, cite the associated HSA-CP paper and
the archived release. A DOI can be added to `CITATION.cff` after the GitHub
release is archived on Zenodo.
