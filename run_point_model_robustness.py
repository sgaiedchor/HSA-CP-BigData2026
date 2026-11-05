"""Point-model robustness: Ridge versus histogram gradient boosting.

All hyperparameter selection is chronological. Candidate gradient-boosting
models train through 2022 and are selected by 2023 MAE. The selected model is
refit through 2023, while HSA-CP rates remain those selected on 2024 validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import run_experiments as core


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "HSA_CP_Analysis_Data_2019_2025.xlsx"
OUTPUT = ROOT / "results" / "point_model_robustness"


CANDIDATES = (
    {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"learning_rate": 0.03, "max_iter": 150, "max_leaf_nodes": 31, "l2_regularization": 1.0},
)


def rows(cube, target, mask):
    x = cube[mask].reshape(-1, cube.shape[2])
    y = target[mask].reshape(-1)
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[ok], y[ok]


def fit_hist_gradient_boosting(cube, target, dates):
    tune_train = dates <= pd.Timestamp("2022-12-31")
    tune_valid = (dates >= pd.Timestamp("2023-01-01")) & (dates <= pd.Timestamp("2023-12-31"))
    final_train = dates <= pd.Timestamp("2023-12-31")
    x_train, y_train = rows(cube, target, tune_train)
    x_valid, y_valid = rows(cube, target, tune_valid)

    tuning = []
    for config in CANDIDATES:
        model = HistGradientBoostingRegressor(
            **config, loss="squared_error", min_samples_leaf=100,
            early_stopping=False, random_state=2026
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        tuning.append({**config, "validation_2023_mae": float(mean_absolute_error(y_valid, pred))})
    selected = min(tuning, key=lambda x: x["validation_2023_mae"])
    selected_config = {k: selected[k] for k in CANDIDATES[0]}

    x_final, y_final = rows(cube, target, final_train)
    model = HistGradientBoostingRegressor(
        **selected_config, loss="squared_error", min_samples_leaf=100,
        early_stopping=False, random_state=2026
    )
    model.fit(x_final, y_final)
    flat = cube.reshape(-1, cube.shape[2])
    pred = np.full(flat.shape[0], np.nan, dtype=float)
    valid = np.isfinite(flat).all(axis=1)
    pred[valid] = model.predict(flat[valid])
    return pred.reshape(target.shape), selected_config, tuning, len(y_final)


def point_metrics(name, y, pred, dates):
    mask_dates = (dates >= core.TEST_START) & (dates <= core.TEST_END)
    actual = y[mask_dates].reshape(-1)
    forecast = pred[mask_dates].reshape(-1)
    ok = np.isfinite(actual) & np.isfinite(forecast)
    actual, forecast = actual[ok], forecast[ok]
    sse = float(np.sum((actual - forecast) ** 2))
    sse_zero = float(np.sum(actual ** 2))
    return {
        "Point_Model": name,
        "N": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, forecast)),
        "RMSE": float(mean_squared_error(actual, forecast) ** 0.5),
        "Sign_Accuracy": float(np.mean(np.sign(actual) == np.sign(forecast))),
        "OOS_R2_vs_Zero": float(1.0 - sse / sse_zero),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = core.load_panel(WORKBOOK)
    cube, target, dates, _ = core.build_feature_cube(panel)
    _, _, ridge_pred, ridge_rows = core.fit_point_model(panel, cube, target, dates)
    hgb_pred, selected, tuning, hgb_rows = fit_hist_gradient_boosting(cube, target, dates)
    scale = core.compute_scale(panel)
    neighbors = core.build_graph(panel, k=5)

    point = pd.DataFrame([
        point_metrics("Ridge", target, ridge_pred, dates),
        point_metrics("HistGradientBoosting", target, hgb_pred, dates),
    ])
    summaries = []
    for name, pred in (("Ridge", ridge_pred), ("HistGradientBoosting", hgb_pred)):
        summary, _, _, _, _, _ = core.evaluate(panel, pred, scale, neighbors)
        summary.insert(0, "Point_Model", name)
        summaries.append(summary)
    conformal = pd.concat(summaries, ignore_index=True)
    point.to_csv(OUTPUT / "point_forecast_metrics.csv", index=False)
    conformal.to_csv(OUTPUT / "conformal_results_by_point_model.csv", index=False)
    (OUTPUT / "protocol.json").write_text(
        json.dumps({
            "selected_hist_gradient_boosting": selected,
            "tuning": tuning,
            "ridge_training_rows": ridge_rows,
            "hist_gradient_boosting_training_rows": hgb_rows,
        }, indent=2), encoding="utf-8"
    )
    print(point.to_string(index=False))
    print("\nHSA-CP results by point model")
    print(conformal[conformal["Method"] == "HSA-CP"].to_string(index=False))


if __name__ == "__main__":
    main()
