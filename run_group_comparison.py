"""Causal overlapping-group feedback comparison for HSA-CP.

The comparator augments the sector feedback used by HSA-CP with one
asset-level state.  All intervals for a date are formed before any state is
updated, so contemporaneous returns never leak across assets.  Hyperparameters
are selected on the 2024 validation year and frozen for the 2025 test.

This is deliberately described as an overlapping-group feedback comparator,
not as an implementation of POGO: POGO is defined for a scalar online stream,
whereas this experiment observes a cross-section simultaneously each day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import (
    ALPHA,
    CALIB_END,
    CALIB_START,
    CONTROLLER_GAMMA,
    EXPERTS,
    HEDGE_ETA,
    TEST_END,
    TEST_START,
    VALID_END,
    VALID_START,
    build_feature_cube,
    build_graph,
    compute_scale,
    expert_quantiles,
    fit_point_model,
    interval_score,
    load_panel,
    safe_mean,
)


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "HSA_CP_Analysis_Data_2019_2025.xlsx"
OUTPUT = ROOT / "results" / "group_comparison"


def summarize(name, y, lo, hi, sector_ids):
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    covered = np.where(valid, (y >= lo) & (y <= hi), np.nan)
    width = np.where(valid, hi - lo, np.nan)
    loss = np.where(valid, interval_score(y, lo, hi, ALPHA), np.nan)
    rolling_asset = pd.DataFrame(covered).rolling(20, min_periods=10).mean().to_numpy()
    rolling_cross = pd.Series(np.nanmean(covered, axis=1)).rolling(20, min_periods=10).mean().to_numpy()
    counts = np.sum(np.isfinite(covered), axis=0)
    asset_cov = np.divide(
        np.nansum(covered, axis=0), counts,
        out=np.full(covered.shape[1], np.nan), where=counts > 0,
    )
    sector_cov = [safe_mean(covered[:, sector_ids == s]) for s in np.unique(sector_ids)]
    return {
        "Method": name,
        "Coverage": safe_mean(covered),
        "Coverage_Gap": abs(safe_mean(covered) - (1 - ALPHA)),
        "Mean_Width": safe_mean(width),
        "Interval_Score": safe_mean(loss),
        "ALCE20": safe_mean(np.abs(rolling_asset - (1 - ALPHA))),
        "XLCE20": safe_mean(np.abs(rolling_cross - (1 - ALPHA))),
        "Asset_Coverage_P10": float(np.nanquantile(asset_cov, 0.10)),
        "Asset_Coverage_Min": float(np.nanmin(asset_cov)),
        "Worst_Sector_Coverage": float(np.nanmin(sector_cov)),
        "N_Valid_Intervals": int(valid.sum()),
    }


def run(panel, pred, scale, dates, scores, sector_ids, sector_names, regime_labels,
        neighbors, burn_positions, eval_positions, asset_gamma=0.0, decay=1.0,
        q_cache=None):
    """Run HSA plus a causal asset correction; return evaluation arrays."""
    all_positions = np.concatenate([burn_positions, eval_positions])
    eval_set = set(int(p) for p in eval_positions)
    selected = np.array([EXPERTS.index(x) for x in ("asset", "sector", "global")])
    log_weights = np.zeros((len(sector_names), len(selected)))
    sector_state = np.zeros(len(sector_names))
    asset_state = np.zeros(len(panel.eligible))
    lo, hi = [], []

    for pos in all_positions:
        qs_all = (q_cache[int(pos)] if q_cache is not None else
                  expert_quantiles(scores, int(pos), sector_ids, neighbors, regime_labels))
        qs = qs_all[:, selected]
        q_base = np.full(len(panel.eligible), np.nan)
        for s in range(len(sector_names)):
            members = sector_ids == s
            w = np.exp(log_weights[s] - np.max(log_weights[s]))
            w /= w.sum()
            q_base[members] = qs[members] @ w + sector_state[s]
        q_final = np.maximum(q_base + asset_state, 0.0)

        if int(pos) in eval_set:
            half = q_final * scale[pos]
            lo.append(pred[pos] - half)
            hi.append(pred[pos] + half)

        # Synchronous end-of-day update: q_final for every asset is already fixed.
        valid = np.isfinite(scores[pos]) & np.isfinite(q_final)
        asset_state *= decay
        asset_state[valid] += asset_gamma * ((scores[pos, valid] > q_final[valid]) - ALPHA)
        for s in range(len(sector_names)):
            members = (sector_ids == s) & valid
            if not members.any():
                continue
            expert_loss = np.nanmean(
                2 * qs[members]
                + (2 / ALPHA) * np.maximum(scores[pos, members, None] - qs[members], 0),
                axis=0,
            )
            loss_scale = max(float(np.nanmedian(expert_loss)), 1e-6)
            log_weights[s] -= HEDGE_ETA * np.clip(expert_loss / loss_scale, 0, 10)
            log_weights[s] -= log_weights[s].max()
            sector_state[s] += CONTROLLER_GAMMA * (
                np.mean(scores[pos, members] > q_final[members]) - ALPHA
            )

    return np.asarray(lo), np.asarray(hi)


def run_binary_pogo(pred, scale, scores, burn_positions, eval_positions):
    """Apply the authors' binary-group POGO update independently by asset.

    Each asset is one scalar stream and all assets for a date are predicted
    before the synchronized state update.  This preserves the information
    timing while giving a parameter-free asset-conditional comparator.
    """
    n_assets = scores.shape[1]
    wealth = np.ones(n_assets)
    miscover = np.zeros(n_assets)
    counts = np.zeros(n_assets)
    eval_set = set(int(p) for p in eval_positions)
    lo, hi = [], []
    for pos in np.concatenate([burn_positions, eval_positions]):
        lambd = (miscover + 0.5) / (counts + 1.0)
        beta = (lambd - ALPHA) / (ALPHA * (1 - ALPHA))
        radius = np.maximum(wealth * beta, 0.0)
        if int(pos) in eval_set:
            half = radius * scale[pos]
            lo.append(pred[pos] - half)
            hi.append(pred[pos] + half)
        valid = np.isfinite(scores[pos])
        covered = scores[pos, valid] <= radius[valid]
        gradient = covered.astype(float) - (1 - ALPHA)
        wealth[valid] *= 1.0 - beta[valid] * gradient
        miscover[valid] += ~covered
        counts[valid] += 1.0
    return np.asarray(lo), np.asarray(hi)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(WORKBOOK)
    cube, target, dates, _ = build_feature_cube(panel)
    _, _, pred, _ = fit_point_model(panel, cube, target, dates)
    scale = compute_scale(panel)
    neighbors = build_graph(panel, k=5)
    scores = np.abs(panel.returns.to_numpy(float) - pred) / scale
    sector_categories = pd.Categorical(panel.sectors.reindex(panel.eligible))
    sector_ids = sector_categories.codes
    sector_names = list(sector_categories.categories)
    regime_labels = panel.regimes["Regime_Causal_3State"].reindex(dates).fillna("Warm-up").to_numpy()

    calib = np.where((dates >= CALIB_START) & (dates <= CALIB_END))[0]
    valid = np.where((dates >= VALID_START) & (dates <= VALID_END))[0]
    pretest = np.where((dates >= CALIB_START) & (dates < TEST_START))[0]
    test = np.where((dates >= TEST_START) & (dates <= TEST_END))[0]
    y = panel.returns.to_numpy(float)
    required = np.unique(np.concatenate([calib, valid, pretest, test]))
    q_cache = {
        int(pos): expert_quantiles(scores, int(pos), sector_ids, neighbors, regime_labels)
        for pos in required
    }

    grid = [(g, d) for g in (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
            for d in (1.0, 0.999, 0.995, 0.99)]
    validation_rows = []
    for gamma, decay in grid:
        lo, hi = run(panel, pred, scale, dates, scores, sector_ids, sector_names,
                     regime_labels, neighbors, calib, valid, gamma, decay, q_cache)
        row = summarize(f"gamma={gamma:g},decay={decay:g}", y[valid], lo, hi, sector_ids)
        row.update({"Asset_Gamma": gamma, "Decay": decay})
        validation_rows.append(row)
    validation = pd.DataFrame(validation_rows)

    # Predeclared selection: require near-target marginal coverage, then maximize
    # lower-tail asset coverage, breaking ties by interval score.  No 2025 outcome
    # enters this choice.
    eligible = validation[validation.Coverage_Gap <= 0.005].copy()
    if eligible.empty:
        eligible = validation.copy()
    eligible = eligible.sort_values(
        ["Asset_Coverage_P10", "Interval_Score"], ascending=[False, True]
    )
    chosen = eligible.iloc[0]
    gamma, decay = float(chosen.Asset_Gamma), float(chosen.Decay)
    validation["Selected_On_Validation"] = (
        (validation.Asset_Gamma == gamma) & (validation.Decay == decay)
    )

    test_rows = []
    for name, g, d in (("HSA-CP", 0.0, 1.0), ("Overlapping-group feedback", gamma, decay)):
        lo, hi = run(panel, pred, scale, dates, scores, sector_ids, sector_names,
                     regime_labels, neighbors, pretest, test, g, d, q_cache)
        row = summarize(name, y[test], lo, hi, sector_ids)
        row.update({"Asset_Gamma": g, "Decay": d})
        test_rows.append(row)
    lo, hi = run_binary_pogo(pred, scale, scores, pretest, test)
    row = summarize("Binary POGO by asset", y[test], lo, hi, sector_ids)
    row.update({"Asset_Gamma": np.nan, "Decay": np.nan})
    test_rows.append(row)
    test_results = pd.DataFrame(test_rows)

    validation.to_csv(OUTPUT / "validation_grid.csv", index=False)
    test_results.to_csv(OUTPUT / "test_comparison.csv", index=False)
    pd.DataFrame([{
        "selection_rule": "Coverage gap <= 0.005; maximize validation asset P10; break ties by interval score",
        "selected_asset_gamma": gamma,
        "selected_decay": decay,
    }]).to_csv(OUTPUT / "selection.csv", index=False)
    print("Selected:", gamma, decay)
    print(test_results.to_string(index=False))


if __name__ == "__main__":
    main()
