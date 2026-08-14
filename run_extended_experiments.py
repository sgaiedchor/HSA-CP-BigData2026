"""Extended, leakage-controlled sensitivity experiments for HSA-CP.

The primary 90% experiment remains frozen.  This script adds:
  * nominal-level sensitivity at 80%, 90%, and 95%;
  * volatility-normalization ablations at the primary 90% level;
  * pre-validation cohort-lock sensitivity; and
  * a single-core reference scalability benchmark.

All predictive evaluations retain the original chronology.  The point model is
fit through 2023, online states burn in over 2024, and 2025 is evaluated without
any re-selection.  Rates and windows are held at the primary configuration.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import (
    CALIB_START,
    CALIB_END,
    CONTROLLER_GAMMA,
    HEDGE_ETA,
    TEST_END,
    TEST_START,
    TRAIN_END,
    build_feature_cube,
    compute_scale,
    conformal_quantile,
    fit_point_model,
    interval_score,
    load_panel,
    safe_mean,
)


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "HSA_CP_Analysis_Data_2019_2025.xlsx"
OUTPUT = ROOT / "results"
PRIMARY_ALPHA = 0.10
LEVELS = (0.80, 0.90, 0.95)


def hsa_quantiles(
    score_history: np.ndarray,
    date_pos: int,
    sector_ids: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Asset, sector, and market quantiles used by the selected architecture."""
    n_assets = score_history.shape[1]
    out = np.full((n_assets, 3), np.nan, dtype=float)
    asset_start = max(0, date_pos - 126)
    sector_start = max(0, date_pos - 63)
    market_start = max(0, date_pos - 5)
    for i in range(n_assets):
        out[i, 0] = conformal_quantile(score_history[asset_start:date_pos, i], alpha)
    for sector_id in np.unique(sector_ids):
        members = np.where(sector_ids == sector_id)[0]
        q_sector = conformal_quantile(
            score_history[sector_start:date_pos, :][:, members].ravel(), alpha
        )
        out[members, 1] = q_sector
    out[:, 2] = conformal_quantile(score_history[market_start:date_pos].ravel(), alpha)
    fallback = np.nanmedian(out[:, 0])
    return np.where(np.isfinite(out), out, fallback)


def evaluate_hsa(panel, pred: np.ndarray, scale: np.ndarray, alpha: float) -> dict[str, float]:
    """Run the selected HSA-CP architecture at one alpha without re-tuning."""
    dates = panel.returns.index
    y = panel.returns.to_numpy(dtype=float)
    scores = np.abs(y - pred) / scale
    categories = pd.Categorical(panel.sectors.reindex(panel.eligible))
    sector_ids = categories.codes
    sector_names = list(categories.categories)
    n_sectors = len(sector_names)
    n_assets = len(panel.eligible)
    burn_positions = np.where((dates >= CALIB_START) & (dates < TEST_START))[0]
    test_positions = np.where((dates >= TEST_START) & (dates <= TEST_END))[0]
    all_positions = np.concatenate([burn_positions, test_positions])
    test_set = set(int(pos) for pos in test_positions)

    log_weights = np.zeros((n_sectors, 3), dtype=float)
    controller = np.zeros(n_sectors, dtype=float)
    lo_rows: list[np.ndarray] = []
    hi_rows: list[np.ndarray] = []

    for pos in all_positions:
        qs = hsa_quantiles(scores, int(pos), sector_ids, alpha)
        q_final = np.full(n_assets, np.nan, dtype=float)
        for sector_id in range(n_sectors):
            members = sector_ids == sector_id
            weights = np.exp(log_weights[sector_id] - np.max(log_weights[sector_id]))
            weights /= weights.sum()
            q_final[members] = np.maximum(
                qs[members] @ weights + controller[sector_id], 0.0
            )

        if int(pos) in test_set:
            half_width = q_final * scale[pos]
            lo_rows.append(pred[pos] - half_width)
            hi_rows.append(pred[pos] + half_width)

        valid = np.isfinite(y[pos]) & np.isfinite(pred[pos]) & np.isfinite(scale[pos])
        for sector_id in range(n_sectors):
            members = (sector_ids == sector_id) & valid
            if not members.any():
                continue
            current_scores = scores[pos, members]
            expert_q = qs[members]
            expert_losses = np.nanmean(
                2 * expert_q
                + (2 / alpha) * np.maximum(current_scores[:, None] - expert_q, 0),
                axis=0,
            )
            loss_scale = max(float(np.nanmedian(expert_losses)), 1e-6)
            log_weights[sector_id] -= HEDGE_ETA * np.clip(
                expert_losses / loss_scale, 0, 10
            )
            log_weights[sector_id] -= log_weights[sector_id].max()
            mean_miss = float(np.mean(current_scores > q_final[members]))
            controller[sector_id] += CONTROLLER_GAMMA * (mean_miss - alpha)

    lo = np.asarray(lo_rows)
    hi = np.asarray(hi_rows)
    y_test = y[test_positions]
    valid = np.isfinite(y_test) & np.isfinite(lo) & np.isfinite(hi)
    covered = np.where(valid, (y_test >= lo) & (y_test <= hi), np.nan)
    width = np.where(valid, hi - lo, np.nan)
    score = np.where(valid, interval_score(y_test, lo, hi, alpha), np.nan)
    rolling_asset = pd.DataFrame(covered).rolling(20, min_periods=10).mean().to_numpy()
    daily_cross = np.nanmean(covered, axis=1)
    rolling_cross = pd.Series(daily_cross).rolling(20, min_periods=10).mean().to_numpy()
    counts = np.sum(np.isfinite(covered), axis=0)
    asset_coverage = np.divide(
        np.nansum(covered, axis=0),
        counts,
        out=np.full(n_assets, np.nan),
        where=counts > 0,
    )
    sector_coverage = [
        safe_mean(covered[:, sector_ids == sector_id])
        for sector_id in range(n_sectors)
    ]
    target = 1 - alpha
    return {
        "Nominal_Coverage": target,
        "Alpha": alpha,
        "Coverage": safe_mean(covered),
        "Coverage_Gap": abs(safe_mean(covered) - target),
        "Mean_Width": safe_mean(width),
        "Median_Width": float(np.nanmedian(width)),
        "Interval_Score": safe_mean(score),
        "ALCE20": safe_mean(np.abs(rolling_asset - target)),
        "XLCE20": safe_mean(np.abs(rolling_cross - target)),
        "Asset_Coverage_P10": float(np.nanquantile(asset_coverage, 0.10)),
        "Worst_Sector_Coverage": float(np.nanmin(sector_coverage)),
        "Best_Sector_Coverage": float(np.nanmax(sector_coverage)),
        "N_Valid_Intervals": int(valid.sum()),
    }


def cohort_sensitivity(workbook: Path) -> pd.DataFrame:
    """Vary only the pre-validation cutoff and availability threshold."""
    excel = pd.ExcelFile(workbook)
    returns_sheet = "Log_Returns" if "Log_Returns" in excel.sheet_names else "Log Returns"
    returns = pd.read_excel(workbook, sheet_name=returns_sheet, index_col=0, parse_dates=True)
    metadata = pd.read_excel(workbook, sheet_name="Metadata")
    known = set(metadata["Original_Ticker"].astype(str))

    def selected(cutoff: pd.Timestamp, threshold: float) -> set[str]:
        availability = returns.loc[CALIB_START:cutoff].notna().mean(axis=0)
        return {
            str(ticker)
            for ticker in availability[availability >= threshold].index
            if str(ticker) in known
        }

    primary = selected(pd.Timestamp("2024-05-31"), 0.99)
    rows = []
    for cutoff in ("2024-04-30", "2024-05-23", "2024-05-31"):
        cutoff_ts = pd.Timestamp(cutoff)
        actual_end = returns.loc[:cutoff_ts].index.max()
        sessions = int(len(returns.loc[CALIB_START:cutoff_ts]))
        for threshold in (0.98, 0.99, 1.00):
            cohort = selected(cutoff_ts, threshold)
            overlap = len(cohort & primary)
            union = len(cohort | primary)
            rows.append({
                "Requested_Cutoff": cutoff,
                "Last_Trading_Session": str(actual_end.date()),
                "Availability_Threshold": threshold,
                "Sessions": sessions,
                "N_Selected": len(cohort),
                "Overlap_With_Primary": overlap,
                "Added_Versus_Primary": len(cohort - primary),
                "Removed_Versus_Primary": len(primary - cohort),
                "Jaccard_With_Primary": overlap / union if union else np.nan,
                "Primary_Configuration": cutoff == "2024-05-31" and threshold == 0.99,
            })
    return pd.DataFrame(rows)


def stratified_subset(sector_ids: np.ndarray, n_assets: int, seed: int = 2026) -> np.ndarray:
    """Deterministic approximately proportional subset retaining every sector."""
    rng = np.random.default_rng(seed + n_assets)
    sectors, counts = np.unique(sector_ids, return_counts=True)
    exact = counts / counts.sum() * n_assets
    allocation = np.floor(exact).astype(int)
    allocation = np.maximum(allocation, 1)
    while allocation.sum() > n_assets:
        candidates = np.where(allocation > 1)[0]
        idx = candidates[np.argmin(exact[candidates] - allocation[candidates])]
        allocation[idx] -= 1
    while allocation.sum() < n_assets:
        room = np.where(allocation < counts)[0]
        idx = room[np.argmax(exact[room] - allocation[room])]
        allocation[idx] += 1
    chosen = []
    for sector, take in zip(sectors, allocation):
        members = np.where(sector_ids == sector)[0]
        chosen.extend(rng.choice(members, size=int(take), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=int)


def benchmark_scalability(panel, pred: np.ndarray, scale: np.ndarray) -> pd.DataFrame:
    """Benchmark one causal HSA prediction/update pass on fixed 2025 sessions."""
    dates = panel.returns.index
    y_all = panel.returns.to_numpy(dtype=float)
    full_scores = np.abs(y_all - pred) / scale
    full_sector_ids = pd.Categorical(panel.sectors.reindex(panel.eligible)).codes
    positions = np.where(
        (dates >= pd.Timestamp("2025-01-02"))
        & (dates <= pd.Timestamp("2025-03-31"))
    )[0]
    repeats = 5
    rows = []

    for n_assets in (50, 100, 250, len(panel.eligible)):
        subset = stratified_subset(full_sector_ids, n_assets)
        y = y_all[:, subset]
        scores = full_scores[:, subset]
        sub_scale = scale[:, subset]
        sector_ids = pd.Categorical(panel.sectors.reindex(panel.eligible).iloc[subset]).codes
        n_sectors = len(np.unique(sector_ids))

        def run_once() -> None:
            log_weights = np.zeros((n_sectors, 3), dtype=float)
            controller = np.zeros(n_sectors, dtype=float)
            for pos in positions:
                qs = hsa_quantiles(scores, int(pos), sector_ids, PRIMARY_ALPHA)
                q_final = np.full(n_assets, np.nan, dtype=float)
                for sector_id in range(n_sectors):
                    members = sector_ids == sector_id
                    weights = np.exp(log_weights[sector_id] - np.max(log_weights[sector_id]))
                    weights /= weights.sum()
                    q_final[members] = np.maximum(
                        qs[members] @ weights + controller[sector_id], 0.0
                    )
                valid = np.isfinite(y[pos]) & np.isfinite(sub_scale[pos])
                for sector_id in range(n_sectors):
                    members = (sector_ids == sector_id) & valid
                    if not members.any():
                        continue
                    current = scores[pos, members]
                    expert_q = qs[members]
                    losses = np.nanmean(
                        2 * expert_q
                        + (2 / PRIMARY_ALPHA) * np.maximum(current[:, None] - expert_q, 0),
                        axis=0,
                    )
                    loss_scale = max(float(np.nanmedian(losses)), 1e-6)
                    log_weights[sector_id] -= HEDGE_ETA * np.clip(
                        losses / loss_scale, 0, 10
                    )
                    log_weights[sector_id] -= log_weights[sector_id].max()
                    controller[sector_id] += CONTROLLER_GAMMA * (
                        float(np.mean(current > q_final[members])) - PRIMARY_ALPHA
                    )

        run_once()  # warm-up outside the timed repetitions
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            run_once()
            elapsed.append(time.perf_counter() - start)
        elapsed = np.asarray(elapsed)
        median_ms = float(np.median(elapsed) * 1000 / len(positions))
        rows.append({
            "N_Assets": n_assets,
            "N_Sectors": n_sectors,
            "Sessions_Per_Repeat": len(positions),
            "Repeats": repeats,
            "Median_Milliseconds_Per_Session": median_ms,
            "IQR_Milliseconds_Per_Session": float(
                (np.quantile(elapsed, 0.75) - np.quantile(elapsed, 0.25))
                * 1000
                / len(positions)
            ),
            "Asset_Intervals_Per_Second": n_assets / (median_ms / 1000),
        })

    result = pd.DataFrame(rows)
    result["Runtime_Relative_To_N50"] = (
        result["Median_Milliseconds_Per_Session"]
        / result.loc[result["N_Assets"] == 50, "Median_Milliseconds_Per_Session"].iloc[0]
    )
    slope = float(np.polyfit(
        np.log(result["N_Assets"]),
        np.log(result["Median_Milliseconds_Per_Session"]),
        1,
    )[0])
    result["Empirical_LogLog_Slope"] = slope
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    panel = load_panel(WORKBOOK)
    cube, target, dates, _ = build_feature_cube(panel)
    _, _, pred, _ = fit_point_model(panel, cube, target, dates)
    ewm_scale = compute_scale(panel)

    level_rows = []
    for level in LEVELS:
        row = evaluate_hsa(panel, pred, ewm_scale, 1 - level)
        row["Rates_And_Windows_Retuned"] = False
        level_rows.append(row)
    multi_level = pd.DataFrame(level_rows)
    multi_level.to_csv(OUTPUT / "multi_coverage_results.csv", index=False)

    training_floor = panel.returns.loc[:TRAIN_END].stack().std() * 0.02
    rolling_scale = (
        panel.returns.shift(1)
        .rolling(21, min_periods=10)
        .std()
        .clip(lower=max(float(training_floor), 1e-5))
        .to_numpy(dtype=float)
    )
    no_scale = np.ones_like(ewm_scale, dtype=float)
    scale_rows = []
    for name, scale in (
        ("EWM21 lagged (primary)", ewm_scale),
        ("Rolling21 lagged", rolling_scale),
        ("No volatility normalization", no_scale),
    ):
        row = evaluate_hsa(panel, pred, scale, PRIMARY_ALPHA)
        row["Scale"] = name
        scale_rows.append(row)
    scale_results = pd.DataFrame(scale_rows)
    scale_results.to_csv(OUTPUT / "scale_ablation_results.csv", index=False)

    cohort = cohort_sensitivity(WORKBOOK)
    cohort.to_csv(OUTPUT / "cohort_lock_sensitivity.csv", index=False)

    scaling = benchmark_scalability(panel, pred, ewm_scale)
    scaling.to_csv(OUTPUT / "scalability_results.csv", index=False)

    primary = pd.read_csv(OUTPUT / "main_results.csv").set_index("Method").loc["HSA-CP"]
    recomputed = multi_level.loc[np.isclose(multi_level["Nominal_Coverage"], 0.90)].iloc[0]
    verification = {
        "primary_recomputed_absolute_differences": {
            key: abs(float(primary[key]) - float(recomputed[key]))
            for key in (
                "Coverage", "Coverage_Gap", "Mean_Width", "Median_Width",
                "Interval_Score", "ALCE20", "XLCE20", "Worst_Sector_Coverage",
            )
        },
        "chronology": {
            "point_fit_end": str(TRAIN_END.date()),
            "online_burn_start": str(CALIB_START.date()),
            "online_burn_end": "2024-12-31",
            "evaluation": [str(TEST_START.date()), str(TEST_END.date())],
        },
        "fixed_configuration": {
            "asset_window": 126,
            "sector_window": 63,
            "market_window": 5,
            "hedge_rate": HEDGE_ETA,
            "controller_rate": CONTROLLER_GAMMA,
        },
        "benchmark_environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "note": "Machine-specific single-process reference timing; not a hardware-independent claim.",
        },
        "runtime_seconds": time.time() - started,
    }
    (OUTPUT / "extended_experiment_info.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print("MULTI-COVERAGE")
    print(multi_level.to_string(index=False))
    print("\nSCALE ABLATION")
    print(scale_results.to_string(index=False))
    print("\nCOHORT SENSITIVITY")
    print(cohort.to_string(index=False))
    print("\nSCALABILITY")
    print(scaling.to_string(index=False))
    print("\nVERIFICATION")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
