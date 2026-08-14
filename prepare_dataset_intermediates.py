"""Prepare auditable CSV inputs for the corrected analysis-data workbook."""

from pathlib import Path
import pandas as pd


ROOT = Path("/workspace/scratch/51583e3576dd/outputs/hsa_cp_bigdata2026_final")
SOURCE = ROOT.parent / "sp500_conformal_dataset_2019_2025.xlsx"
OUT = ROOT / "tmp" / "dataset_csv"
OUT.mkdir(parents=True, exist_ok=True)

returns = pd.read_excel(SOURCE, sheet_name="Log_Returns", index_col=0, parse_dates=True)
metadata = pd.read_excel(SOURCE, sheet_name="Metadata")
regimes = pd.read_excel(SOURCE, sheet_name="Market_Regimes", index_col=0, parse_dates=True)
membership = pd.read_excel(SOURCE, sheet_name="Membership_Changes")
quality = pd.read_excel(SOURCE, sheet_name="Quality_Checks")

calibration = returns.loc["2024-01-01":"2024-05-31"]
coverage = calibration.notna().mean(axis=0)
metadata["Coverage_PreLock_2024H1"] = metadata["Original_Ticker"].map(coverage)
metadata["Eligible_PreLock_2024H1"] = metadata["Coverage_PreLock_2024H1"].ge(0.99)
metadata["Analysis_Status"] = metadata["Eligible_PreLock_2024H1"].map(
    {True: "Included at 2024-05-31 lock", False: "Excluded: insufficient pre-lock history"}
)
metadata["Future_Filter_Used_In_Analysis"] = False

analysis_columns = [
    "Original_Ticker", "Provider_Ticker", "Security", "GICS_Sector", "GICS_SubIndustry",
    "SP500_Date_Added", "Coverage_PreLock_2024H1", "Eligible_PreLock_2024H1",
    "First_Price_Date", "Last_Price_Date", "Quoted_Near_2025_12_31",
    "Data_Quality_Flag", "Analysis_Status", "Notes",
]
analysis_universe = metadata.loc[metadata["Eligible_PreLock_2024H1"], analysis_columns].copy()

readme = pd.DataFrame([
    ("Dataset", "HSA-CP leakage-controlled S&P 500 fixed-snapshot analysis panel"),
    ("Purpose", "One-day-ahead online conformal interval experiments"),
    ("Price/return period", "2019-04-01 to 2025-12-31"),
    ("Constituent snapshot", "503 securities in the S&P 500 snapshot dated 2024-05-23"),
    ("Analysis cohort", f"{len(analysis_universe)} securities locked before validation"),
    ("Cohort lock rule", "At least 99% observed log returns from 2024-01-01 through 2024-05-31"),
    ("Excluded at lock", ", ".join(metadata.loc[~metadata["Eligible_PreLock_2024H1"], "Original_Ticker"].astype(str))),
    ("Anti-leakage rule", "No availability, delisting, acquisition, or quotation information from 2025 is used to select securities"),
    ("Post-lock missingness", "Natural post-acquisition/delisting blanks are retained and never imputed"),
    ("Return definition", "Log_Return(i,t) = ln(AdjustedClose(i,t) / AdjustedClose(i,t-1))"),
    ("Individual-price imputation", "None: no interpolation, forward fill, or backward fill"),
    ("Causal regime", "For date t, VIX(t-1) is compared with rolling lagged-VIX terciles; all inputs are available by t-1"),
    ("Composition source", "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1225357006"),
    ("Price source", "https://query2.finance.yahoo.com/v8/finance/chart/{SYMBOL}"),
    ("Redistribution note", "Review provider terms before placing raw/derived market data in a public repository"),
    ("Code expectation", "run_experiments.py derives the cohort from pre-lock returns; it never reads the legacy future-balanced flag"),
    ("Generated", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
], columns=["Field", "Value"])

splits = pd.DataFrame([
    ("Point-model tuning train", "2019-04-01", "2022-12-30", "Ridge candidates"),
    ("Point-model tuning validation", "2023-01-03", "2023-12-29", "Select ridge penalty"),
    ("Initial online calibration", "2024-01-02", "2024-05-31", "Cohort lock and controller burn-in"),
    ("Architecture/rate validation", "2024-06-03", "2024-12-31", "No 2025 outcomes"),
    ("Final evaluation", "2025-01-02", "2025-12-31", "No parameter selection"),
], columns=["Partition", "Start", "End", "Role"])

readme.to_csv(OUT / "README.csv", index=False)
metadata.to_csv(OUT / "Metadata.csv", index=False)
analysis_universe.to_csv(OUT / "Analysis_Universe.csv", index=False)
returns.rename_axis("Date").to_csv(OUT / "Log_Returns.csv")
regimes.rename_axis("Date").to_csv(OUT / "Market_Regimes.csv")
membership.to_csv(OUT / "Membership_Changes.csv", index=False)
quality.to_csv(OUT / "Quality_Checks.csv", index=False)
splits.to_csv(OUT / "Data_Splits.csv", index=False)

print(f"Prepared {len(analysis_universe)}-security cohort in {OUT}")
