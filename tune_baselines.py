"""Validation-only tuning for adaptive baseline learning rates.

The grid is evaluated after a January--May 2024 burn-in and scored only on
June--December 2024.  No 2025 outcome participates in parameter selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import (
    ALPHA,
    CALIB_START,
    CALIB_END,
    VALID_START,
    VALID_END,
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
ACI_GRID = (0.001, 0.0025, 0.005, 0.01, 0.02)
PCONTROL_GRID = (0.01, 0.025, 0.05, 0.10, 0.20)


def summarize(name, rate, y, pred, scale, thresholds, eval_rows):
    half = thresholds[eval_rows] * scale[eval_rows]
    lo = pred[eval_rows] - half
    hi = pred[eval_rows] + half
    y_eval = y[eval_rows]
    valid = np.isfinite(y_eval) & np.isfinite(lo) & np.isfinite(hi)
    covered = np.where(valid, (y_eval >= lo) & (y_eval <= hi), np.nan)
    loss = np.where(valid, interval_score(y_eval, lo, hi, ALPHA), np.nan)
    width = np.where(valid, hi - lo, np.nan)
    coverage = safe_mean(covered)
    return {
        "Method": name,
        "Rate": rate,
        "Coverage": coverage,
        "Coverage_Gap": abs(coverage - (1 - ALPHA)),
        "Mean_Width": safe_mean(width),
        "Interval_Score": safe_mean(loss),
        "N_Valid_Intervals": int(valid.sum()),
    }


def simulate_aci(scores, y, pred, scale, positions, gamma):
    n_dates, n_assets = y.shape
    alpha_t = np.full(n_assets, ALPHA, dtype=float)
    thresholds = np.full((n_dates, n_assets), np.nan)
    for pos in positions:
        start = max(0, int(pos) - 126)
        q = np.array([
            conformal_quantile(scores[start:pos, i], alpha_t[i]) for i in range(n_assets)
        ])
        thresholds[pos] = np.maximum(q, 0.0)
        valid = np.isfinite(y[pos]) & np.isfinite(pred[pos]) & np.isfinite(scale[pos]) & np.isfinite(q)
        err = (scores[pos] > q).astype(float)
        alpha_t[valid] = np.clip(alpha_t[valid] + gamma * (ALPHA - err[valid]), 0.005, 0.40)
    return thresholds


def simulate_pcontrol(scores, y, pred, scale, dates, positions, eta):
    n_dates, n_assets = y.shape
    pre = np.where(dates < CALIB_START)[0][-126:]
    state = np.array([conformal_quantile(scores[pre, i], ALPHA) for i in range(n_assets)])
    thresholds = np.full((n_dates, n_assets), np.nan)
    for pos in positions:
        thresholds[pos] = np.maximum(state, 0.0)
        valid = np.isfinite(y[pos]) & np.isfinite(pred[pos]) & np.isfinite(scale[pos]) & np.isfinite(state)
        err = (scores[pos] > state).astype(float)
        state[valid] += eta * (err[valid] - ALPHA)
    return thresholds


def choose(rows):
    frame = pd.DataFrame(rows).sort_values("Interval_Score")
    feasible = frame[frame["Coverage_Gap"] <= 0.005]
    selected = feasible.iloc[0] if not feasible.empty else frame.iloc[0]
    return frame, float(selected["Rate"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(WORKBOOK)
    cube, y, dates, _ = build_feature_cube(panel)
    _, _, pred, _ = fit_point_model(panel, cube, y, dates)
    scale = compute_scale(panel)
    scores = np.abs(y - pred) / scale
    burn = np.where((dates >= CALIB_START) & (dates <= CALIB_END))[0]
    valid = np.where((dates >= VALID_START) & (dates <= VALID_END))[0]
    positions = np.concatenate([burn, valid])

    aci_rows = []
    for gamma in ACI_GRID:
        q = simulate_aci(scores, y, pred, scale, positions, gamma)
        aci_rows.append(summarize("ACI", gamma, y, pred, scale, q, valid))
    aci, selected_aci = choose(aci_rows)

    p_rows = []
    for eta in PCONTROL_GRID:
        q = simulate_pcontrol(scores, y, pred, scale, dates, positions, eta)
        p_rows.append(summarize("PControl", eta, y, pred, scale, q, valid))
    pcontrol, selected_pcontrol = choose(p_rows)

    tuning = pd.concat([aci, pcontrol], ignore_index=True)
    tuning["Selected_On_Validation"] = (
        ((tuning["Method"] == "ACI") & np.isclose(tuning["Rate"], selected_aci))
        | ((tuning["Method"] == "PControl") & np.isclose(tuning["Rate"], selected_pcontrol))
    )
    tuning.to_csv(OUTPUT / "baseline_tuning_results.csv", index=False)
    info = {
        "selection_rule": "Lowest validation interval score among rates with absolute coverage gap <= 0.005",
        "period": [str(VALID_START.date()), str(VALID_END.date())],
        "selected_aci_gamma": selected_aci,
        "selected_pcontrol_eta": selected_pcontrol,
        "aci_grid": ACI_GRID,
        "pcontrol_grid": PCONTROL_GRID,
    }
    (OUTPUT / "baseline_tuning_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(tuning.to_string(index=False))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
