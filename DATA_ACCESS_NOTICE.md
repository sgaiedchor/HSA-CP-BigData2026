# Data access notice

The scripts expect `HSA_CP_Analysis_Data_2019_2025.xlsx` in the project root.
That workbook contains a documented fixed-snapshot panel derived from market
data and should be redistributed only when the applicable provider terms allow
it.

For a public GitHub repository, the safest default is to publish the code,
configuration, derived aggregate result tables, figures, and schema while
excluding the analysis workbook. A user with lawful access can reconstruct or
place the workbook locally and run:

```bash
python tune_baselines.py
python run_ablations.py
python run_experiments.py
python run_extended_experiments.py
python make_artifacts.py
```

The research outputs are not an investment strategy or trading recommendation.
