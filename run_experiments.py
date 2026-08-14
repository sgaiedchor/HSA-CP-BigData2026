"""Leakage-controlled experiments for HSA-CP on the S&P 500 panel.

The pipeline is strictly chronological:
  point-model train: 2019-04-01 .. 2023-12-29
  initial conformal calibration: 2024-01-02 .. 2024-05-31
  architecture validation: 2024-06-03 .. 2024-12-31
  final evaluation: 2025-01-02 .. 2025-12-31

No target-day return is used before its interval is issued.  The analysis
universe is locked using information available by 2024-05-31; availability in
the 2025 evaluation period is never used for security selection.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ALPHA = 0.10
TRAIN_END = pd.Timestamp("2023-12-31")
CALIB_START = pd.Timestamp("2024-01-01")
CALIB_END = pd.Timestamp("2024-05-31")
VALID_START = pd.Timestamp("2024-06-03")
VALID_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-31")
EXPERTS = ("asset", "sector", "graph", "regime", "global")
PRELOCK_MIN_COVERAGE = 0.99
ACI_GAMMA = 0.005
PCONTROL_ETA = 0.05
HEDGE_ETA = 0.04
CONTROLLER_GAMMA = 0.05
RIDGE_ALPHA_GRID = (1.0, 10.0, 25.0, 100.0)
PROJECT_ROOT = Path(__file__).resolve().parent


def conformal_quantile(values: np.ndarray, alpha: float = ALPHA) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    # Exact split-conformal finite-sample correction: take the kth order
    # statistic directly.  Passing k/n to np.quantile(method="higher") would
    # be one rank too conservative for most sample sizes because NumPy maps
    # quantile levels to indices using n-1.
    rank = min(values.size, max(1, math.ceil((values.size + 1) * (1 - alpha))))
    return float(np.partition(values, rank - 1)[rank - 1])


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    width = hi - lo
    return width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def moving_block_bootstrap_ci(
    values: np.ndarray,
    block_length: int = 10,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Percentile CI for a mean while preserving short-range time dependence."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < block_length:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    starts = np.arange(values.size - block_length + 1)
    blocks_needed = math.ceil(values.size / block_length)
    means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[s:s + block_length] for s in chosen])[: values.size]
        means[b] = np.mean(sample)
    return tuple(np.quantile(means, [0.025, 0.975]))


def newey_west_mean_test(values: np.ndarray, max_lag: int = 10) -> tuple[float, float, float]:
    """One-sided HAC test that the mean paired loss difference is positive."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan
    centered = x - x.mean()
    long_run = float(np.dot(centered, centered) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2.0 * (1.0 - lag / (max_lag + 1.0)) * gamma
    se = math.sqrt(max(long_run, 0.0) / n)
    z = float(x.mean() / se) if se > 0 else (np.inf if x.mean() > 0 else 0.0)
    p_one_sided = float(norm.sf(z))
    return se, z, p_one_sided


@dataclass
class Panel:
    returns: pd.DataFrame
    metadata: pd.DataFrame
    regimes: pd.DataFrame
    sectors: pd.Series
    eligible: list[str]


def load_panel(workbook: Path) -> Panel:
    sheet_names = pd.ExcelFile(workbook).sheet_names
    returns_sheet = "Log_Returns" if "Log_Returns" in sheet_names else "Log Returns"
    regimes_sheet = "Market_Regimes" if "Market_Regimes" in sheet_names else "Market Regimes"
    returns = pd.read_excel(workbook, sheet_name=returns_sheet, index_col=0, parse_dates=True)
    metadata = pd.read_excel(workbook, sheet_name="Metadata")
    regimes = pd.read_excel(workbook, sheet_name=regimes_sheet, index_col=0, parse_dates=True)
    # Lock the cohort before architecture validation.  This deliberately does
    # not use the legacy future-informed 2024H2--2025 balanced-panel flag.
    prelock = returns.loc[CALIB_START:CALIB_END]
    availability = prelock.notna().mean(axis=0)
    eligible = availability[availability >= PRELOCK_MIN_COVERAGE].index.astype(str).tolist()
    known_tickers = set(metadata["Original_Ticker"].astype(str))
    eligible = [ticker for ticker in eligible if ticker in known_tickers]
    sectors = metadata.set_index("Original_Ticker")["GICS_Sector"].reindex(eligible)
    returns = returns[eligible].astype(float)
    returns.index = pd.to_datetime(returns.index)
    regimes.index = pd.to_datetime(regimes.index)
    return Panel(returns=returns, metadata=metadata, regimes=regimes, sectors=sectors, eligible=eligible)


def build_feature_cube(panel: Panel) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    r = panel.returns
    tickers = panel.eligible
    dates = r.index
    sector_names = sorted(panel.sectors.dropna().unique())
    market = panel.regimes["GSPC_LogReturn"].reindex(dates)
    vix_lag1 = panel.regimes["VIX_Lag1"].reindex(dates)
    mvol_lag1 = panel.regimes["MarketVol21_Annualized_Lag1"].reindex(dates)
    regime = panel.regimes["Regime_Causal_3State"].reindex(dates)

    sector_return = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for sector in sector_names:
        members = panel.sectors[panel.sectors == sector].index.tolist()
        sector_block = r[members]
        sector_sum = sector_block.sum(axis=1, skipna=True)
        sector_count = sector_block.notna().sum(axis=1)
        # Leave-one-out sector return avoids mechanically feeding an asset's
        # own lagged return back through the sector feature.
        for ticker in members:
            denom = sector_count - sector_block[ticker].notna().astype(int)
            numer = sector_sum - sector_block[ticker].fillna(0.0)
            sector_return[ticker] = numer.div(denom.where(denom > 0))

    features = [
        r.shift(1).to_numpy(),
        r.shift(2).to_numpy(),
        r.shift(5).to_numpy(),
        r.shift(1).rolling(5, min_periods=3).mean().to_numpy(),
        r.shift(1).rolling(21, min_periods=10).mean().to_numpy(),
        r.shift(1).rolling(5, min_periods=3).std().to_numpy(),
        r.shift(1).rolling(21, min_periods=10).std().to_numpy(),
        np.repeat(market.shift(1).to_numpy()[:, None], len(tickers), axis=1),
        sector_return.shift(1).to_numpy(),
        np.repeat(vix_lag1.to_numpy()[:, None], len(tickers), axis=1),
        np.repeat(mvol_lag1.to_numpy()[:, None], len(tickers), axis=1),
    ]
    for label in ("Low", "Normal", "High"):
        flag = (regime == label).astype(float).to_numpy()
        features.append(np.repeat(flag[:, None], len(tickers), axis=1))
    for sector in sector_names:
        flag = (panel.sectors.reindex(tickers) == sector).astype(float).to_numpy()
        features.append(np.repeat(flag[None, :], len(dates), axis=0))
    cube = np.stack(features, axis=2).astype(np.float32)
    target = r.to_numpy(dtype=np.float32)
    feature_names = [
        "ret_lag1", "ret_lag2", "ret_lag5", "mean5_lag1", "mean21_lag1",
        "vol5_lag1", "vol21_lag1", "market_lag1", "sector_lag1", "vix_lag1",
        "market_vol21_lag1", "regime_low", "regime_normal", "regime_high",
    ] + [f"sector_{s}" for s in sector_names]
    return cube, target, dates, feature_names


def fit_point_model(panel: Panel, cube: np.ndarray, target: np.ndarray, dates: pd.DatetimeIndex):
    train_date_mask = dates <= TRAIN_END
    x_train = cube[train_date_mask].reshape(-1, cube.shape[2])
    y_train = target[train_date_mask].reshape(-1)
    valid = np.isfinite(y_train) & np.isfinite(x_train).all(axis=1)
    x_train, y_train = x_train[valid], y_train[valid]
    # Select the common point-model penalty without using conformal calibration,
    # architecture-validation, or 2025 outcomes: fit through 2022 and validate
    # on calendar 2023, then refit the selected model through 2023.
    tune_train_mask = dates <= pd.Timestamp("2022-12-31")
    tune_valid_mask = (dates >= pd.Timestamp("2023-01-01")) & (dates <= TRAIN_END)
    x_tune = cube[tune_train_mask].reshape(-1, cube.shape[2])
    y_tune = target[tune_train_mask].reshape(-1)
    ok_tune = np.isfinite(y_tune) & np.isfinite(x_tune).all(axis=1)
    x_tune, y_tune = x_tune[ok_tune], y_tune[ok_tune]
    x_valid = cube[tune_valid_mask].reshape(-1, cube.shape[2])
    y_valid = target[tune_valid_mask].reshape(-1)
    ok_valid = np.isfinite(y_valid) & np.isfinite(x_valid).all(axis=1)
    x_valid, y_valid = x_valid[ok_valid], y_valid[ok_valid]
    tune_scaler = StandardScaler()
    x_tune_scaled = tune_scaler.fit_transform(x_tune)
    x_valid_scaled = tune_scaler.transform(x_valid)
    tuning = []
    for candidate in RIDGE_ALPHA_GRID:
        candidate_model = Ridge(alpha=candidate, fit_intercept=True, solver="lsqr", tol=1e-6)
        candidate_model.fit(x_tune_scaled, y_tune)
        mae = float(np.mean(np.abs(y_valid - candidate_model.predict(x_valid_scaled))))
        tuning.append({"Alpha": candidate, "Validation_2023_MAE": mae})
    selected_alpha = min(tuning, key=lambda row: row["Validation_2023_MAE"])["Alpha"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = Ridge(alpha=selected_alpha, fit_intercept=True, solver="lsqr", tol=1e-6)
    model.fit(x_train_scaled, y_train)
    model.hsa_tuning_results_ = tuning
    model.hsa_selected_alpha_ = selected_alpha

    flat = cube.reshape(-1, cube.shape[2])
    pred = np.full(flat.shape[0], np.nan, dtype=float)
    valid_all = np.isfinite(flat).all(axis=1)
    pred[valid_all] = model.predict(scaler.transform(flat[valid_all]))
    pred = pred.reshape(target.shape)
    return model, scaler, pred, int(valid.sum())


def build_graph(panel: Panel, k: int = 5) -> np.ndarray:
    train = panel.returns.loc[:TRAIN_END]
    corr = train.corr(min_periods=252).fillna(-1.0)
    neighbors = np.empty((len(panel.eligible), k + 1), dtype=int)
    ticker_pos = {ticker: i for i, ticker in enumerate(panel.eligible)}
    for i, ticker in enumerate(panel.eligible):
        # The graph complements rather than duplicates GICS sectors: neighbors
        # are selected over the full universe using training-only correlations.
        candidates = pd.Index(panel.eligible)
        ranked = corr.loc[ticker, candidates].drop(index=ticker, errors="ignore").sort_values(ascending=False)
        chosen = [ticker] + ranked.head(k).index.tolist()
        while len(chosen) < k + 1:
            chosen.append(ticker)
        neighbors[i] = [ticker_pos[x] for x in chosen]
    return neighbors


def compute_scale(panel: Panel) -> np.ndarray:
    vol = panel.returns.ewm(span=21, adjust=False, min_periods=10).std().shift(1)
    # The floor is estimated from training observations only.
    floor = panel.returns.loc[:TRAIN_END].stack().std() * 0.02
    return vol.clip(lower=max(float(floor), 1e-5)).to_numpy(dtype=float)


def expert_quantiles(
    score_history: np.ndarray,
    date_pos: int,
    sector_ids: np.ndarray,
    neighbors: np.ndarray,
    regime_labels: np.ndarray,
) -> np.ndarray:
    n_assets = score_history.shape[1]
    out = np.full((n_assets, len(EXPERTS)), np.nan, dtype=float)
    asset_start = max(0, date_pos - 126)
    graph_start = max(0, date_pos - 63)
    regime_start = max(0, date_pos - 252)
    global_start = max(0, date_pos - 5)
    current_regime = regime_labels[date_pos]
    for i in range(n_assets):
        out[i, 0] = conformal_quantile(score_history[asset_start:date_pos, i], ALPHA)
    for sector_id in np.unique(sector_ids):
        members = np.where(sector_ids == sector_id)[0]
        q = conformal_quantile(score_history[graph_start:date_pos, :][:, members].ravel(), ALPHA)
        out[members, 1] = q
    for i in range(n_assets):
        out[i, 2] = conformal_quantile(score_history[graph_start:date_pos, neighbors[i]].ravel(), ALPHA)
        past_pos = np.arange(regime_start, date_pos)
        matched = past_pos[regime_labels[past_pos] == current_regime]
        values = score_history[matched, i] if matched.size >= 30 else score_history[regime_start:date_pos, i]
        out[i, 3] = conformal_quantile(values, ALPHA)
    global_q = conformal_quantile(score_history[global_start:date_pos].ravel(), ALPHA)
    out[:, 4] = global_q
    fallback = np.nanmedian(out[:, 0])
    out = np.where(np.isfinite(out), out, fallback)
    return out


def evaluate(panel: Panel, pred: np.ndarray, scale: np.ndarray, neighbors: np.ndarray):
    dates = panel.returns.index
    y = panel.returns.to_numpy(dtype=float)
    scores = np.abs(y - pred) / scale
    test_positions = np.where((dates >= TEST_START) & (dates <= TEST_END))[0]
    n_test, n_assets = len(test_positions), y.shape[1]
    sector_categories = pd.Categorical(panel.sectors.reindex(panel.eligible))
    sector_ids = sector_categories.codes
    sector_names = list(sector_categories.categories)
    regime_labels = panel.regimes["Regime_Causal_3State"].reindex(dates).fillna("Warm-up").to_numpy()

    methods = ["SplitCP", "RollingCP", "ACI", "PControl", "SectorCP", "GraphCP", "RegimeCP", "HSA-CP"]
    lo = {m: np.full((n_test, n_assets), np.nan) for m in methods}
    hi = {m: np.full((n_test, n_assets), np.nan) for m in methods}
    q_store = {m: np.full((n_test, n_assets), np.nan) for m in methods}

    calib_mask = (dates >= CALIB_START) & (dates <= CALIB_END)
    split_q = np.array([conformal_quantile(scores[calib_mask, i], ALPHA) for i in range(n_assets)])
    alpha_aci = np.full(n_assets, ALPHA, dtype=float)
    gamma_aci = ACI_GAMMA
    pre_calib_positions = np.where(dates < CALIB_START)[0][-126:]
    q_pcontrol = np.array([
        conformal_quantile(scores[pre_calib_positions, i], ALPHA) for i in range(n_assets)
    ])
    eta_pcontrol = PCONTROL_ETA
    n_sectors = len(sector_names)
    # This active architecture was selected on June--December 2024 only.
    # The graph expert remains a benchmark but was excluded after validation.
    proposed_experts = ("asset", "sector", "global")
    proposed_idx = np.array([EXPERTS.index(name) for name in proposed_experts], dtype=int)
    log_weights = np.zeros((n_sectors, len(proposed_experts)), dtype=float)
    controller = np.zeros(n_sectors, dtype=float)
    hedge_eta = HEDGE_ETA
    controller_gamma = CONTROLLER_GAMMA
    weight_path = []
    controller_path = []
    sector_miss_path = []

    def update_online_state(pos: int, qs: np.ndarray, q_aci: np.ndarray, q_hsa: np.ndarray) -> None:
        nonlocal alpha_aci, q_pcontrol, log_weights, controller
        valid = np.isfinite(y[pos]) & np.isfinite(pred[pos]) & np.isfinite(scale[pos])
        err_aci = (scores[pos] > q_aci).astype(float)
        alpha_aci[valid] = np.clip(alpha_aci[valid] + gamma_aci * (ALPHA - err_aci[valid]), 0.005, 0.40)
        err_pcontrol = (scores[pos] > q_pcontrol).astype(float)
        q_pcontrol[valid] += eta_pcontrol * (err_pcontrol[valid] - ALPHA)
        for sector_id in range(n_sectors):
            members = (sector_ids == sector_id) & valid
            if not members.any():
                continue
            current_scores = scores[pos, members]
            expert_q = qs[members][:, proposed_idx]
            expert_losses = np.nanmean(
                2 * expert_q + (2 / ALPHA) * np.maximum(current_scores[:, None] - expert_q, 0), axis=0
            )
            scale_loss = max(float(np.nanmedian(expert_losses)), 1e-6)
            log_weights[sector_id] -= hedge_eta * np.clip(expert_losses / scale_loss, 0, 10)
            log_weights[sector_id] -= log_weights[sector_id].max()
            miss = float(np.mean(current_scores > q_hsa[members]))
            controller[sector_id] += controller_gamma * (miss - ALPHA)

    # Burn in adaptive states using all observations strictly before the final
    # 2025 test. June-December 2024 is the architecture-validation period.
    burn_positions = np.where((dates >= CALIB_START) & (dates < TEST_START))[0]
    for pos in burn_positions:
        qs = expert_quantiles(scores, pos, sector_ids, neighbors, regime_labels)
        asset_start = max(0, pos - 126)
        q_aci = np.array([
            conformal_quantile(scores[asset_start:pos, i], alpha_aci[i]) for i in range(n_assets)
        ])
        q_hsa = np.full(n_assets, np.nan)
        for sector_id in range(n_sectors):
            weights = np.exp(log_weights[sector_id] - np.max(log_weights[sector_id]))
            weights /= weights.sum()
            members = sector_ids == sector_id
            q_hsa[members] = np.maximum(
                (qs[members][:, proposed_idx] @ weights) + controller[sector_id], 0.0
            )
        update_online_state(pos, qs, q_aci, q_hsa)

    controller_test_start = controller.copy()

    for out_t, pos in enumerate(test_positions):
        qs = expert_quantiles(scores, pos, sector_ids, neighbors, regime_labels)
        asset_start = max(0, pos - 126)
        q_aci = np.array([
            conformal_quantile(scores[asset_start:pos, i], alpha_aci[i]) for i in range(n_assets)
        ])
        q_map = {
            "SplitCP": split_q,
            "RollingCP": qs[:, 0],
            "ACI": q_aci,
            "PControl": q_pcontrol.copy(),
            "SectorCP": qs[:, 1],
            "GraphCP": qs[:, 2],
            "RegimeCP": qs[:, 3],
        }
        q_hsa = np.full(n_assets, np.nan)
        weights_today = np.full((n_sectors, len(proposed_experts)), np.nan)
        for sector_id in range(n_sectors):
            w = np.exp(log_weights[sector_id] - np.max(log_weights[sector_id]))
            w /= w.sum()
            weights_today[sector_id] = w
            members = sector_ids == sector_id
            q_hsa[members] = np.maximum(
                (qs[members][:, proposed_idx] @ w) + controller[sector_id], 0.0
            )
        q_map["HSA-CP"] = q_hsa

        for method, q in q_map.items():
            q = np.maximum(q, 0.0)
            q_store[method][out_t] = q
            half = q * scale[pos]
            lo[method][out_t] = pred[pos] - half
            hi[method][out_t] = pred[pos] + half

        misses_today = np.full(n_sectors, np.nan)
        valid_today = np.isfinite(y[pos]) & np.isfinite(pred[pos]) & np.isfinite(scale[pos])
        for sector_id in range(n_sectors):
            members = (sector_ids == sector_id) & valid_today
            if members.any():
                misses_today[sector_id] = float(np.mean(scores[pos, members] > q_hsa[members]))
        sector_miss_path.append(misses_today)
        update_online_state(pos, qs, q_aci, q_hsa)
        weight_path.append(weights_today.copy())
        controller_path.append(controller.copy())

    test_dates = dates[test_positions]
    y_test = y[test_positions]
    records = []
    sector_records = []
    regime_records = []
    daily_losses = {}
    regime_test = regime_labels[test_positions]
    for method in methods:
        valid_test = (
            np.isfinite(y_test) & np.isfinite(lo[method]) & np.isfinite(hi[method])
        )
        covered = (y_test >= lo[method]) & (y_test <= hi[method])
        width = hi[method] - lo[method]
        loss = interval_score(y_test, lo[method], hi[method], ALPHA)
        covered = np.where(valid_test, covered, np.nan)
        width = np.where(valid_test, width, np.nan)
        loss = np.where(valid_test, loss, np.nan)
        daily_losses[method] = np.nanmean(loss, axis=1)
        rolling_asset_cov = pd.DataFrame(covered).rolling(20, min_periods=10).mean().to_numpy()
        daily_cross_cov = np.nanmean(covered, axis=1)
        rolling_cross_cov = pd.Series(daily_cross_cov).rolling(20, min_periods=10).mean().to_numpy()
        asset_counts = np.sum(np.isfinite(covered), axis=0)
        asset_cov = np.divide(
            np.nansum(covered, axis=0),
            asset_counts,
            out=np.full(covered.shape[1], np.nan),
            where=asset_counts > 0,
        )
        regime_cov = {
            label: safe_mean(covered[regime_test == label]) for label in ("Low", "Normal", "High")
        }
        sector_cov = {
            sector_names[s]: safe_mean(covered[:, sector_ids == s]) for s in range(n_sectors)
        }
        for s, sector_name in enumerate(sector_names):
            members = sector_ids == s
            sector_records.append({
                "Method": method,
                "Sector": sector_name,
                "N_Assets": int(members.sum()),
                "Coverage": safe_mean(covered[:, members]),
                "Mean_Width": safe_mean(width[:, members]),
                "Interval_Score": safe_mean(loss[:, members]),
            })
        for regime_name in ("Low", "Normal", "High"):
            members = regime_test == regime_name
            regime_records.append({
                "Method": method,
                "Regime": regime_name,
                "N_Days": int(members.sum()),
                "Coverage": safe_mean(covered[members]),
                "Mean_Width": safe_mean(width[members]),
                "Interval_Score": safe_mean(loss[members]),
            })
        records.append({
            "Method": method,
            "Coverage": safe_mean(covered),
            "Coverage_Gap": abs(safe_mean(covered) - (1 - ALPHA)),
            "Mean_Width": safe_mean(width),
            "Median_Width": float(np.nanmedian(width)),
            "Interval_Score": safe_mean(loss),
            "ALCE20": safe_mean(np.abs(rolling_asset_cov - (1 - ALPHA))),
            "XLCE20": safe_mean(np.abs(rolling_cross_cov - (1 - ALPHA))),
            "Asset_Coverage_P10": float(np.nanquantile(asset_cov, 0.10)),
            "Worst_Regime_Coverage": float(np.nanmin(list(regime_cov.values()))),
            "Worst_Sector_Coverage": float(np.nanmin(list(sector_cov.values()))),
            "N_Valid_Intervals": int(valid_test.sum()),
            **{f"Coverage_Regime_{k}": v for k, v in regime_cov.items()},
        })
    summary = pd.DataFrame(records)

    tests = []
    proposed = daily_losses["HSA-CP"]
    for comparator_id, method in enumerate(methods):
        if method == "HSA-CP":
            continue
        valid = np.isfinite(proposed) & np.isfinite(daily_losses[method])
        diff = daily_losses[method][valid] - proposed[valid]
        hac_se, hac_z, p_value = newey_west_mean_test(diff, max_lag=10)
        ci_5 = moving_block_bootstrap_ci(diff, block_length=5, seed=2026 + comparator_id)
        ci_10 = moving_block_bootstrap_ci(diff, block_length=10, seed=3026 + comparator_id)
        ci_20 = moving_block_bootstrap_ci(diff, block_length=20, seed=4026 + comparator_id)
        tests.append({
            "Comparator": method,
            "Mean_Loss_Difference_ComparatorMinusHSA": float(np.mean(diff)),
            "HAC_SE_Lag10": float(hac_se),
            "HAC_Z": float(hac_z),
            "P_Value_OneSided_HAC": float(p_value),
            "MBB5_Lower": float(ci_5[0]),
            "MBB5_Upper": float(ci_5[1]),
            "MBB10_Lower": float(ci_10[0]),
            "MBB10_Upper": float(ci_10[1]),
            "MBB20_Lower": float(ci_20[0]),
            "MBB20_Upper": float(ci_20[1]),
            "Robust_All_Block_Lengths": bool(ci_5[0] > 0 and ci_10[0] > 0 and ci_20[0] > 0),
        })
    tests_df = pd.DataFrame(tests).sort_values("P_Value_OneSided_HAC")
    family_size = len(tests_df)
    tests_df["Holm_Adjusted_P"] = np.minimum(
        1.0,
        np.maximum.accumulate(
            tests_df["P_Value_OneSided_HAC"].to_numpy() * np.arange(family_size, 0, -1)
        ),
    )
    tests_df["Holm_Reject_5pct"] = tests_df["Holm_Adjusted_P"] <= 0.05

    long_rows = []
    for method in methods:
        valid_method = np.isfinite(y_test) & np.isfinite(lo[method]) & np.isfinite(hi[method])
        covered = (y_test >= lo[method]) & (y_test <= hi[method])
        width = hi[method] - lo[method]
        loss = interval_score(y_test, lo[method], hi[method], ALPHA)
        for t, date in enumerate(test_dates):
            long_rows.append({
                "Date": date,
                "Method": method,
                "Regime": regime_test[t],
                "N_Valid_Daily": int(valid_method[t].sum()),
                "Coverage_Daily": safe_mean(np.where(valid_method[t], covered[t], np.nan)),
                "Width_Daily": safe_mean(np.where(valid_method[t], width[t], np.nan)),
                "Interval_Score_Daily": safe_mean(np.where(valid_method[t], loss[t], np.nan)),
            })
    daily = pd.DataFrame(long_rows)
    paths = {
        "weights": np.asarray(weight_path),
        "controller": np.asarray(controller_path),
        "controller_test_start": controller_test_start,
        "sector_miss": np.asarray(sector_miss_path),
        "sector_names": sector_names,
        "expert_names": list(proposed_experts),
        "test_dates": test_dates,
    }
    return (
        summary,
        tests_df,
        daily,
        paths,
        pd.DataFrame(sector_records),
        pd.DataFrame(regime_records),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workbook",
        type=Path,
        default=PROJECT_ROOT / "HSA_CP_Analysis_Data_2019_2025.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    panel = load_panel(args.workbook)
    cube, target, dates, feature_names = build_feature_cube(panel)
    model, scaler, pred, n_train = fit_point_model(panel, cube, target, dates)
    scale = compute_scale(panel)
    neighbors = build_graph(panel, k=5)
    summary, tests, daily, paths, sector_results, regime_results = evaluate(panel, pred, scale, neighbors)

    summary.to_csv(args.output / "main_results.csv", index=False)
    tests.to_csv(args.output / "statistical_tests.csv", index=False)
    daily.to_csv(args.output / "daily_results.csv", index=False)
    sector_results.to_csv(args.output / "sector_results.csv", index=False)
    regime_results.to_csv(args.output / "regime_results.csv", index=False)
    np.savez_compressed(
        args.output / "online_paths.npz",
        weights=paths["weights"],
        controller=paths["controller"],
        controller_test_start=paths["controller_test_start"],
        sector_miss=paths["sector_miss"],
        sector_names=np.asarray(paths["sector_names"], dtype=str),
        expert_names=np.asarray(paths["expert_names"], dtype=str),
        test_dates=np.asarray(paths["test_dates"].astype(str), dtype=str),
    )
    run_info = {
        "method": "HSA-CP",
        "method_full_name": "Hierarchical Sector-Adaptive Conformal Prediction",
        "target_coverage": 1 - ALPHA,
        "tickers": len(panel.eligible),
        "universe_lock_rule": (
            f"At least {PRELOCK_MIN_COVERAGE:.0%} observed log returns on "
            f"{CALIB_START.date()} through {CALIB_END.date()}; no 2025 availability filter"
        ),
        "train_rows": n_train,
        "train_end": str(TRAIN_END.date()),
        "calibration_period": [str(CALIB_START.date()), str(CALIB_END.date())],
        "validation_period": [str(VALID_START.date()), str(VALID_END.date())],
        "test_period": [str(TEST_START.date()), str(TEST_END.date())],
        "features": feature_names,
        "point_model": f"Pooled standardized ridge regression, alpha={model.hsa_selected_alpha_:g}",
        "point_model_tuning": model.hsa_tuning_results_,
        "online_rates": {
            "aci_gamma": ACI_GAMMA,
            "pcontrol_eta": PCONTROL_ETA,
            "hsa_hedge_eta": HEDGE_ETA,
            "hsa_controller_gamma": CONTROLLER_GAMMA,
        },
        "graph": "Top-5 training-period correlations across the full eligible universe",
        "architecture_selection": (
            "Asset, sector, and global experts with sector-level Hedge aggregation and feedback control; "
            "selected on 2024-06-03 through 2024-12-31 before the 2025 evaluation."
        ),
        "runtime_seconds": time.time() - started,
        "randomness": "Deterministic predictions; block-bootstrap seeds are fixed and recorded",
    }
    miss = paths["sector_miss"]
    controller_end = paths["controller"][-1]
    diagnostics = []
    for sector_id, sector_name in enumerate(paths["sector_names"]):
        observed = np.isfinite(miss[:, sector_id])
        t_count = int(observed.sum())
        mean_miss = float(np.mean(miss[observed, sector_id]))
        identity_rhs = float(
            (controller_end[sector_id] - paths["controller_test_start"][sector_id])
            / (CONTROLLER_GAMMA * t_count)
        )
        diagnostics.append({
            "Sector": sector_name,
            "Updated_Days": t_count,
            "Mean_Daily_Sector_Miscoverage": mean_miss,
            "Mean_Miscoverage_Minus_Alpha": mean_miss - ALPHA,
            "Controller_Identity_RHS": identity_rhs,
            "Absolute_Identity_Error": abs((mean_miss - ALPHA) - identity_rhs),
        })
    pd.DataFrame(diagnostics).to_csv(args.output / "controller_identity_diagnostics.csv", index=False)
    (args.output / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nStatistical tests\n", tests.to_string(index=False))
    print("\n", json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
