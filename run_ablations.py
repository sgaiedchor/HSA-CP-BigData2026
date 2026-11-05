"""Chronological validation and 2025 ablations for HSA-CP."""

from __future__ import annotations

import json
import time
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
OUTPUT = ROOT / "results"


VARIANTS = {
    "HSA-CP": {"experts": ["asset", "sector", "global"], "hedge": True, "controller": True},
    "Add_Graph_Expert": {"experts": ["asset", "sector", "graph", "global"], "hedge": True, "controller": True},
    "Add_Regime_Expert": {"experts": ["asset", "sector", "regime", "global"], "hedge": True, "controller": True},
    "No_Sector": {"experts": ["asset", "global"], "hedge": True, "controller": True},
    "No_Global": {"experts": ["asset", "sector"], "hedge": True, "controller": True},
    "No_Hedge_EqualWeights": {"experts": ["asset", "sector", "global"], "hedge": False, "controller": True},
    "No_Controller": {"experts": ["asset", "sector", "global"], "hedge": True, "controller": False},
    "EqualWeights_NoController": {"experts": ["asset", "sector", "global"], "hedge": False, "controller": False},
    "Asset_Controller_Only": {"experts": ["asset"], "hedge": False, "controller": True},
    "Sector_Controller_Only": {"experts": ["sector"], "hedge": False, "controller": True},
}

SENSITIVITY = {
    **{
        f"Controller_{gamma:.2f}": {
            "experts": ["asset", "sector", "global"], "hedge": True, "controller": True,
            "controller_gamma": gamma, "hedge_eta": HEDGE_ETA,
        }
        for gamma in (0.05, 0.08, 0.12, 0.18, 0.25)
    },
    **{
        f"Hedge_{eta:.2f}": {
            "experts": ["asset", "sector", "global"], "hedge": True, "controller": True,
            "controller_gamma": CONTROLLER_GAMMA, "hedge_eta": eta,
        }
        for eta in (0.01, 0.02, 0.04, 0.08, 0.16)
    },
}


def evaluate_variants(
    panel,
    pred,
    scale,
    dates,
    y,
    scores,
    sector_ids,
    sector_names,
    regime_labels,
    burn_positions,
    eval_positions,
    q_cache,
    variants,
):
    """Evaluate every architecture after a strictly prior online burn-in."""
    all_positions = np.concatenate([burn_positions, eval_positions])
    eval_set = set(int(pos) for pos in eval_positions)
    rows = []
    for variant_name, config in variants.items():
        selected = np.array([EXPERTS.index(name) for name in config["experts"]], dtype=int)
        log_weights = np.zeros((len(sector_names), len(selected)), dtype=float)
        controller = np.zeros(len(sector_names), dtype=float)
        pred_lo, pred_hi = [], []
        for pos in all_positions:
            qs_all = q_cache[int(pos)]
            qs = qs_all[:, selected]
            q_final = np.full(len(panel.eligible), np.nan)
            for sector_id in range(len(sector_names)):
                members = sector_ids == sector_id
                if config["hedge"]:
                    weights = np.exp(log_weights[sector_id] - np.max(log_weights[sector_id]))
                    weights /= weights.sum()
                else:
                    weights = np.full(len(selected), 1 / len(selected))
                q = qs[members] @ weights
                if config["controller"]:
                    q = q + controller[sector_id]
                q_final[members] = np.maximum(q, 0.0)

            if int(pos) in eval_set:
                half = q_final * scale[pos]
                pred_lo.append(pred[pos] - half)
                pred_hi.append(pred[pos] + half)

            valid = np.isfinite(y[pos]) & np.isfinite(pred[pos]) & np.isfinite(scale[pos])
            for sector_id in range(len(sector_names)):
                members = (sector_ids == sector_id) & valid
                if not members.any():
                    continue
                current_scores = scores[pos, members]
                expert_q = qs[members]
                if config["hedge"]:
                    expert_losses = np.nanmean(
                        2 * expert_q + (2 / ALPHA) * np.maximum(current_scores[:, None] - expert_q, 0), axis=0
                    )
                    scale_loss = max(float(np.nanmedian(expert_losses)), 1e-6)
                    hedge_eta = float(config.get("hedge_eta", HEDGE_ETA))
                    log_weights[sector_id] -= hedge_eta * np.clip(expert_losses / scale_loss, 0, 10)
                    log_weights[sector_id] -= log_weights[sector_id].max()
                if config["controller"]:
                    miss = float(np.mean(current_scores > q_final[members]))
                    controller_gamma = float(config.get("controller_gamma", CONTROLLER_GAMMA))
                    controller[sector_id] += controller_gamma * (miss - ALPHA)

        lo = np.asarray(pred_lo)
        hi = np.asarray(pred_hi)
        y_eval = y[eval_positions]
        valid_eval = np.isfinite(y_eval) & np.isfinite(lo) & np.isfinite(hi)
        covered = np.where(valid_eval, (y_eval >= lo) & (y_eval <= hi), np.nan)
        width = np.where(valid_eval, hi - lo, np.nan)
        loss = np.where(valid_eval, interval_score(y_eval, lo, hi, ALPHA), np.nan)
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
        regime_eval = regime_labels[eval_positions]
        regime_cov = {label: safe_mean(covered[regime_eval == label]) for label in ("Low", "Normal", "High")}
        sector_cov = {
            sector_names[s]: safe_mean(covered[:, sector_ids == s]) for s in range(len(sector_names))
        }
        rows.append({
            "Variant": variant_name,
            "Coverage": safe_mean(covered),
            "Coverage_Gap": abs(safe_mean(covered) - (1 - ALPHA)),
            "Mean_Width": safe_mean(width),
            "Interval_Score": safe_mean(loss),
            "ALCE20": safe_mean(np.abs(rolling_asset_cov - (1 - ALPHA))),
            "XLCE20": safe_mean(np.abs(rolling_cross_cov - (1 - ALPHA))),
            "Asset_Coverage_P10": float(np.nanquantile(asset_cov, 0.10)),
            "Worst_Regime_Coverage": float(np.nanmin(list(regime_cov.values()))),
            "Worst_Sector_Coverage": float(np.nanmin(list(sector_cov.values()))),
            "N_Valid_Intervals": int(valid_eval.sum()),
        })
    return pd.DataFrame(rows).sort_values("Interval_Score")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    panel = load_panel(WORKBOOK)
    cube, target, dates, _ = build_feature_cube(panel)
    _, _, pred, _ = fit_point_model(panel, cube, target, dates)
    scale = compute_scale(panel)
    neighbors = build_graph(panel, k=5)
    y = panel.returns.to_numpy(dtype=float)
    scores = np.abs(y - pred) / scale
    sector_categories = pd.Categorical(panel.sectors.reindex(panel.eligible))
    sector_ids = sector_categories.codes
    sector_names = list(sector_categories.categories)
    regime_labels = panel.regimes["Regime_Causal_3State"].reindex(dates).fillna("Warm-up").to_numpy()

    calibration_positions = np.where((dates >= CALIB_START) & (dates <= CALIB_END))[0]
    validation_positions = np.where((dates >= VALID_START) & (dates <= VALID_END))[0]
    pretest_positions = np.where((dates >= CALIB_START) & (dates < TEST_START))[0]
    test_positions = np.where((dates >= TEST_START) & (dates <= TEST_END))[0]
    required_positions = np.unique(
        np.concatenate([calibration_positions, validation_positions, pretest_positions, test_positions])
    )

    q_cache = {
        int(pos): expert_quantiles(scores, int(pos), sector_ids, neighbors, regime_labels)
        for pos in required_positions
    }

    validation_results = evaluate_variants(
        panel,
        pred,
        scale,
        dates,
        y,
        scores,
        sector_ids,
        sector_names,
        regime_labels,
        calibration_positions,
        validation_positions,
        q_cache,
        VARIANTS,
    )
    test_results = evaluate_variants(
        panel,
        pred,
        scale,
        dates,
        y,
        scores,
        sector_ids,
        sector_names,
        regime_labels,
        pretest_positions,
        test_positions,
        q_cache,
        VARIANTS,
    )
    sensitivity_results = evaluate_variants(
        panel,
        pred,
        scale,
        dates,
        y,
        scores,
        sector_ids,
        sector_names,
        regime_labels,
        calibration_positions,
        validation_positions,
        q_cache,
        SENSITIVITY,
    )

    # The feedback controller is a constitutive component tied to Proposition 1,
    # not an optional expert chosen for its test score.  No_Controller is kept
    # as a diagnostic ablation but is excluded from architecture selection.
    controller_preserving = [name for name, cfg in VARIANTS.items() if cfg["controller"]]
    selection_pool = validation_results[validation_results["Variant"].isin(controller_preserving)]
    eligible = selection_pool[selection_pool["Coverage_Gap"] <= 0.005]
    if eligible.empty:
        eligible = selection_pool
    selected_variant = str(eligible.iloc[0]["Variant"])
    validation_results["Selected_On_Validation"] = validation_results["Variant"].eq(selected_variant)
    test_results["Selected_On_Validation"] = test_results["Variant"].eq(selected_variant)

    validation_results.to_csv(OUTPUT / "validation_ablation_results.csv", index=False)
    test_results.to_csv(OUTPUT / "test_ablation_results.csv", index=False)
    test_results.to_csv(OUTPUT / "ablation_results.csv", index=False)
    sensitivity_results.to_csv(OUTPUT / "validation_sensitivity_results.csv", index=False)
    metadata = {
        "variants": VARIANTS,
        "sensitivity_grid": SENSITIVITY,
        "runtime_seconds": time.time() - started,
        "selected_variant": selected_variant,
        "selection_policy": (
            "The feedback controller is constitutive and No_Controller is diagnostic only. Among "
            "controller-preserving variants, select the lowest validation interval score with absolute "
            "coverage gap <= 0.005; if none qualify, select the lowest score. Selection uses only "
            "2024-06-03 through 2024-12-31 after a 2024-01-02 through 2024-05-31 burn-in."
        ),
        "test_policy": "No architecture choice uses 2025; burn online states over all of 2024, then evaluate 2025.",
    }
    (OUTPUT / "ablation_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("VALIDATION ABRATIONS")
    print(validation_results.to_string(index=False))
    print("\nHELD-OUT TEST ABLATIONS")
    print(test_results.to_string(index=False))
    print("\nVALIDATION-ONLY SENSITIVITY")
    print(sensitivity_results.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
