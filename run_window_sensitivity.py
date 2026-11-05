"""Validation-only window sensitivity followed by one frozen 2025 check."""

from itertools import product
from pathlib import Path

import pandas as pd

from run_experiments import build_feature_cube, compute_scale, fit_point_model, load_panel
from run_extended_experiments import evaluate_hsa


ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "HSA_CP_Analysis_Data_2019_2025.xlsx"
OUTPUT = ROOT / "results" / "window_sensitivity"
PRIMARY = (126, 63, 5)
GRID = tuple(product((63, 126, 252), (21, 63, 126), (5, 10, 21)))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(WORKBOOK)
    cube, target, _, _ = build_feature_cube(panel)
    _, _, pred, _ = fit_point_model(panel, cube, target, panel.returns.index)
    scale = compute_scale(panel)

    validation_rows = []
    for windows in GRID:
        row = evaluate_hsa(
            panel, pred, scale, 0.10, windows=windows,
            eval_start=pd.Timestamp("2024-06-03"),
            eval_end=pd.Timestamp("2024-12-31"),
        )
        row["Primary"] = windows == PRIMARY
        validation_rows.append(row)
    validation = pd.DataFrame(validation_rows).sort_values("Interval_Score")
    feasible = validation[validation["Coverage_Gap"] <= 0.005]
    selected = (feasible if not feasible.empty else validation).iloc[0]
    selected_windows = (
        int(selected["Asset_Window"]),
        int(selected["Sector_Window"]),
        int(selected["Market_Window"]),
    )
    validation["Selected_On_Validation"] = (
        (validation["Asset_Window"] == selected_windows[0])
        & (validation["Sector_Window"] == selected_windows[1])
        & (validation["Market_Window"] == selected_windows[2])
    )
    test_rows = []
    for label, windows in (("Primary", PRIMARY), ("Validation_Selected", selected_windows)):
        row = evaluate_hsa(panel, pred, scale, 0.10, windows=windows)
        row["Configuration"] = label
        test_rows.append(row)
    test = pd.DataFrame(test_rows)
    validation.to_csv(OUTPUT / "window_validation_grid.csv", index=False)
    test.to_csv(OUTPUT / "window_frozen_test_comparison.csv", index=False)
    print("VALIDATION GRID")
    print(validation.to_string(index=False))
    print("\nFROZEN TEST COMPARISON")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
