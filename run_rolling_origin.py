"""Two-fold rolling-origin robustness analysis for HSA-CP.

This script evaluates the same fixed-snapshot research panel with two strictly
chronological origins. It is a temporal robustness analysis, not a
point-in-time index-membership backtest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import run_experiments as core
import run_ablations as abl
import tune_baselines as baseline_tuning


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "HSA_CP_Analysis_Data_2019_2025.xlsx"
OUTPUT = ROOT / "results" / "rolling_origin"


FOLDS = (
    {
        "fold": "test_2024",
        "tune_train_end": "2021-12-31",
        "tune_valid_start": "2022-01-01",
        "train_end": "2022-12-31",
        "calib_start": "2023-01-01",
        "calib_end": "2023-05-31",
        "valid_start": "2023-06-01",
        "valid_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "fold": "test_2025",
        "tune_train_end": "2022-12-31",
        "tune_valid_start": "2023-01-01",
        "train_end": "2023-12-31",
        "calib_start": "2024-01-01",
        "calib_end": "2024-05-31",
        "valid_start": "2024-06-03",
        "valid_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
)


def fit_point_model_for_fold(panel, cube, target, dates, fold):
    train_end = pd.Timestamp(fold["train_end"])
    tune_train_end = pd.Timestamp(fold["tune_train_end"])
    tune_valid_start = pd.Timestamp(fold["tune_valid_start"])

    tune_train_mask = dates <= tune_train_end
    tune_valid_mask = (dates >= tune_valid_start) & (dates <= train_end)
    final_train_mask = dates <= train_end

    def rows(mask):
        x = cube[mask].reshape(-1, cube.shape[2])
        y = target[mask].reshape(-1)
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        return x[ok], y[ok]

    x_tune, y_tune = rows(tune_train_mask)
    x_valid, y_valid = rows(tune_valid_mask)
    x_train, y_train = rows(final_train_mask)

    tune_scaler = StandardScaler()
    x_tune_scaled = tune_scaler.fit_transform(x_tune)
    x_valid_scaled = tune_scaler.transform(x_valid)
    tuning = []
    for alpha in core.RIDGE_ALPHA_GRID:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-6)
        model.fit(x_tune_scaled, y_tune)
        mae = float(np.mean(np.abs(y_valid - model.predict(x_valid_scaled))))
        tuning.append({"alpha": alpha, "validation_mae": mae})
    selected = min(tuning, key=lambda x: x["validation_mae"])["alpha"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = Ridge(alpha=selected, fit_intercept=True, solver="lsqr", tol=1e-6)
    model.fit(x_train_scaled, y_train)

    flat = cube.reshape(-1, cube.shape[2])
    pred = np.full(flat.shape[0], np.nan, dtype=float)
    valid = np.isfinite(flat).all(axis=1)
    pred[valid] = model.predict(scaler.transform(flat[valid]))
    return pred.reshape(target.shape), selected, tuning, len(y_train)


def configure(fold):
    core.TRAIN_END = pd.Timestamp(fold["train_end"])
    core.CALIB_START = pd.Timestamp(fold["calib_start"])
    core.CALIB_END = pd.Timestamp(fold["calib_end"])
    core.VALID_START = pd.Timestamp(fold["valid_start"])
    core.VALID_END = pd.Timestamp(fold["valid_end"])
    core.TEST_START = pd.Timestamp(fold["test_start"])
    core.TEST_END = pd.Timestamp(fold["test_end"])


def tune_online_rates(panel, pred, scale, dates, target, neighbors):
    """Select all adaptive rates using the fold's validation period only."""
    scores = np.abs(target - pred) / scale
    burn = np.where((dates >= core.CALIB_START) & (dates <= core.CALIB_END))[0]
    valid = np.where((dates >= core.VALID_START) & (dates <= core.VALID_END))[0]
    positions = np.concatenate([burn, valid])

    baseline_tuning.CALIB_START = core.CALIB_START
    aci_rows = []
    for gamma in baseline_tuning.ACI_GRID:
        q = baseline_tuning.simulate_aci(scores, target, pred, scale, positions, gamma)
        aci_rows.append(
            baseline_tuning.summarize("ACI", gamma, target, pred, scale, q, valid)
        )
    _, selected_aci = baseline_tuning.choose(aci_rows)

    pcontrol_rows = []
    for eta in baseline_tuning.PCONTROL_GRID:
        q = baseline_tuning.simulate_pcontrol(
            scores, target, pred, scale, dates, positions, eta
        )
        pcontrol_rows.append(
            baseline_tuning.summarize("PControl", eta, target, pred, scale, q, valid)
        )
    _, selected_pcontrol = baseline_tuning.choose(pcontrol_rows)

    sector_categories = pd.Categorical(panel.sectors.reindex(panel.eligible))
    sector_ids = sector_categories.codes
    sector_names = list(sector_categories.categories)
    regime_labels = (
        panel.regimes["Regime_Causal_3State"].reindex(dates).fillna("Warm-up").to_numpy()
    )
    q_cache = {
        int(pos): core.expert_quantiles(scores, int(pos), sector_ids, neighbors, regime_labels)
        for pos in positions
    }
    joint_grid = {
        f"controller_{controller:.3f}_hedge_{hedge:.3f}": {
            "experts": ["asset", "sector", "global"],
            "hedge": True,
            "controller": True,
            "controller_gamma": controller,
            "hedge_eta": hedge,
        }
        for controller in (0.05, 0.08, 0.12, 0.18, 0.25)
        for hedge in (0.01, 0.02, 0.04, 0.08, 0.16)
    }
    hsa_validation = abl.evaluate_variants(
        panel, pred, scale, dates, target, scores, sector_ids, sector_names,
        regime_labels, burn, valid, q_cache, joint_grid
    )
    feasible = hsa_validation[hsa_validation["Coverage_Gap"] <= 0.005]
    selected_row = (feasible if not feasible.empty else hsa_validation).iloc[0]
    selected_config = joint_grid[str(selected_row["Variant"])]
    return {
        "aci_gamma": float(selected_aci),
        "pcontrol_eta": float(selected_pcontrol),
        "controller_gamma": float(selected_config["controller_gamma"]),
        "hedge_eta": float(selected_config["hedge_eta"]),
        "hsa_validation_coverage": float(selected_row["Coverage"]),
        "hsa_validation_score": float(selected_row["Interval_Score"]),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    sectors = []
    metadata = []
    for fold in FOLDS:
        configure(fold)
        panel = core.load_panel(WORKBOOK)
        cube, target, dates, _ = core.build_feature_cube(panel)
        pred, selected_alpha, tuning, n_train = fit_point_model_for_fold(
            panel, cube, target, dates, fold
        )
        scale = core.compute_scale(panel)
        neighbors = core.build_graph(panel, k=5)
        rates = tune_online_rates(panel, pred, scale, dates, target, neighbors)
        core.ACI_GAMMA = rates["aci_gamma"]
        core.PCONTROL_ETA = rates["pcontrol_eta"]
        core.CONTROLLER_GAMMA = rates["controller_gamma"]
        core.HEDGE_ETA = rates["hedge_eta"]
        summary, _, _, _, sector_results, _ = core.evaluate(
            panel, pred, scale, neighbors
        )
        summary.insert(0, "Fold", fold["fold"])
        sector_results.insert(0, "Fold", fold["fold"])
        summaries.append(summary)
        sectors.append(sector_results)
        metadata.append(
            {
                **fold,
                "n_assets": len(panel.eligible),
                "n_train_rows": n_train,
                "selected_ridge_alpha": selected_alpha,
                "ridge_tuning": tuning,
                "validation_selected_online_rates": rates,
                "fixed_snapshot_caveat": True,
            }
        )

    summary_all = pd.concat(summaries, ignore_index=True)
    sector_all = pd.concat(sectors, ignore_index=True)
    summary_all.to_csv(OUTPUT / "rolling_origin_main_results.csv", index=False)
    sector_all.to_csv(OUTPUT / "rolling_origin_sector_results.csv", index=False)
    (OUTPUT / "rolling_origin_protocol.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary_all.to_string(index=False))


if __name__ == "__main__":
    main()
