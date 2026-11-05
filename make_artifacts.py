"""Create publication figures and descriptive tables from locked experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from run_experiments import TEST_END, TEST_START, build_feature_cube, fit_point_model, load_panel


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
WORKBOOK = ROOT / "HSA_CP_Analysis_Data_2019_2025.xlsx"

COLORS = {
    "HSA-CP": "#0072B2",
    "PControl": "#D55E00",
    "SectorCP": "#009E73",
    "ACI": "#CC79A7",
    "SplitCP": "#666666",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, subtitle, face, edge="#34495E", title_size=7.0, sub_size=6.3):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=1.15, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h * 0.68, title,
            ha="center", va="center", fontsize=title_size, weight="bold", color="#1F2D3D",
        )
        ax.text(
            x + w / 2, y + h * 0.34, subtitle,
            ha="center", va="center", fontsize=sub_size, linespacing=1.25, color="#2F3E4E",
        )
        return (x, y, w, h)

    def arrow(start, end, dashed=False, rad=0.0, color="#34495E", lw=1.05):
        style = (0, (3, 2)) if dashed else "solid"
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9.5,
            linewidth=lw, linestyle=style, color=color,
            connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5,
        ))

    ax.text(
        0.50, 0.965, r"Prediction at session $t$ (all solid-arrow quantities are $\mathcal{F}_{t-1}$-measurable)",
        ha="center", va="top", fontsize=7.5, weight="bold", color="#273746",
    )

    inputs = box(
        0.012, 0.43, 0.145, 0.32, "Causal inputs",
        r"returns $<t$" "\nsector map; VIX",
        "#EAF2F8", edge="#2874A6",
    )
    forecast = box(
        0.192, 0.43, 0.145, 0.32, "Forecast + scale",
        r"$\widehat r_{i,t}$" "\n" r"lagged $\widehat\sigma_{i,t}$",
        "#F2F4F4", edge="#566573",
    )
    asset = box(
        0.382, 0.70, 0.145, 0.16, "Asset expert",
        r"$q^A_{i,t}$; 126 sessions", "#E8F4FC", edge="#2E86C1", title_size=6.6, sub_size=5.9,
    )
    sector = box(
        0.382, 0.49, 0.145, 0.16, "Sector expert",
        r"$q^S_{i,t}$; 63 sessions", "#E9F7EF", edge="#239B56", title_size=6.6, sub_size=5.9,
    )
    market = box(
        0.382, 0.28, 0.145, 0.16, "Market expert",
        r"$q^M_{i,t}$; 5 sessions", "#FEF5E7", edge="#CA6F1E", title_size=6.6, sub_size=5.9,
    )
    hedge = box(
        0.575, 0.43, 0.13, 0.32, "Sector Hedge",
        r"$q^{mix}=\sum_jw_{g,t,j}q^j$" "\n" r"$w^+\!\propto w e^{-\lambda z}$",
        "#E8F8F5", edge="#148F77", title_size=6.8, sub_size=5.8,
    )
    control = box(
        0.750, 0.43, 0.12, 0.32, "Feedback",
        r"$q=q^{mix}+c_{g,t}$" "\n" r"$c^+=c+\eta(\bar e-\alpha)$",
        "#FFF5D9", edge="#B7950B", title_size=6.8, sub_size=5.7,
    )
    output = box(
        0.910, 0.43, 0.080, 0.32, "Issue",
        r"$C_{i,t}$" "\n90% interval",
        "#F4ECF7", edge="#7D3C98", title_size=6.8, sub_size=5.9,
    )

    arrow((inputs[0] + inputs[2], 0.59), (forecast[0], 0.59))
    for expert in (asset, sector, market):
        arrow((forecast[0] + forecast[2], 0.59), (expert[0], expert[1] + expert[3] / 2))
        arrow((expert[0] + expert[2], expert[1] + expert[3] / 2), (hedge[0], 0.59))
    arrow((hedge[0] + hedge[2], 0.59), (control[0], 0.59))
    arrow((control[0] + control[2], 0.59), (output[0], 0.59))

    observed = box(
        0.785, 0.055, 0.175, 0.17, "Observe after issue",
        r"$r_{i,t}$ and valid-asset mask",
        "#FDEDEC", edge="#B03A2E", title_size=6.5, sub_size=5.8,
    )
    arrow(
        (output[0] + output[2] / 2, output[1]),
        (observed[0] + observed[2] / 2, observed[1] + observed[3]),
        dashed=True, rad=0.10, color="#A93226",
    )
    arrow(
        (observed[0] + 0.02, observed[1] + observed[3]),
        (control[0] + control[2] / 2, control[1]),
        dashed=True, rad=-0.08, color="#A93226",
    )
    arrow(
        (observed[0] + 0.015, observed[1] + 0.08),
        (hedge[0] + hedge[2] / 2, hedge[1]),
        dashed=True, rad=-0.18, color="#A93226",
    )
    arrow(
        (observed[0], observed[1] + 0.055),
        (sector[0] + sector[2] / 2, market[1]),
        dashed=True, rad=-0.12, color="#A93226",
    )
    ax.text(
        0.39, 0.085,
        "Dashed arrows: append scores and update expert losses / sector miss\n"
        "using observed assets only",
        ha="center", va="center", fontsize=5.9, linespacing=1.25, color="#922B21",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )
    save_figure(fig, "fig1_hsa_architecture")


def rolling_coverage_figure() -> None:
    daily = pd.read_csv(RESULTS / "daily_results.csv", parse_dates=["Date"])
    pivot = daily.pivot(index="Date", columns="Method", values="Coverage_Daily")
    rolling = pivot.rolling(20, min_periods=10).mean()
    fig, ax = plt.subplots(figsize=(7.15, 2.65))
    for method in ("HSA-CP", "PControl", "SectorCP", "ACI"):
        ax.plot(rolling.index, rolling[method], label=method, color=COLORS[method], lw=1.55)
    ax.axhline(0.90, color="black", ls="--", lw=1.0, label="Nominal 90%")
    # Keep every trajectory inside the axes; the accepted-version plot clipped
    # peaks above 0.95 and was correctly flagged by a reviewer.
    finite = rolling[["HSA-CP", "PControl", "SectorCP", "ACI"]].to_numpy()
    lower = min(0.84, float(np.nanmin(finite)) - 0.005)
    upper = max(0.95, float(np.nanmax(finite)) + 0.005)
    ax.set_ylim(lower, upper)
    ax.set_ylabel("20-day coverage")
    ax.set_xlabel("2025 evaluation date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", color="#DDDDDD", lw=0.7)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, fontsize=8)
    save_figure(fig, "fig2_rolling_coverage")


def robustness_and_weights_figure() -> None:
    sector = pd.read_csv(RESULTS / "sector_results.csv")
    methods_sector = ["HSA-CP", "ACI", "PControl", "SectorCP"]
    hsa_sector = sector[sector["Method"] == "HSA-CP"].sort_values("Coverage")
    sector_order = hsa_sector["Sector"].tolist()
    paths = np.load(RESULTS / "online_paths.npz")
    dates = pd.to_datetime(paths["test_dates"])
    weights = paths["weights"].mean(axis=1)
    expert_names = paths["expert_names"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), gridspec_kw={"width_ratios": [1.08, 1.0]})
    ax = axes[0]
    y_pos = np.arange(len(sector_order))
    offsets = np.linspace(-0.24, 0.24, len(methods_sector))
    markers = {"HSA-CP": "o", "ACI": "s", "PControl": "^", "SectorCP": "D"}
    for off, method in zip(offsets, methods_sector):
        part = sector[sector["Method"] == method].set_index("Sector").reindex(sector_order)
        gap = np.abs(part["Coverage"].to_numpy() - 0.90)
        ax.scatter(gap, y_pos + off, s=13, marker=markers[method],
                   color=COLORS[method], label=method, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_order, fontsize=6.3)
    ax.set_xlim(left=0)
    ax.set_xlabel(r"$|$sector coverage $-0.90|$")
    ax.grid(axis="x", color="#E5E5E5", lw=0.6)
    ax.legend(frameon=False, ncol=2, fontsize=6.2, loc="lower right")
    ax.set_title("(a) Sector calibration gaps", fontsize=9)

    ax = axes[1]
    expert_colors = {"asset": "#0072B2", "sector": "#009E73", "global": "#D55E00"}
    for j, expert in enumerate(expert_names):
        smoothed = pd.Series(weights[:, j], index=dates).rolling(10, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed, label=expert.capitalize(), color=expert_colors[expert], lw=1.45)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean sector weight")
    ax.set_xlabel("2025")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper center")
    ax.set_title("(b) Online expert weights", fontsize=9)
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, "fig3_sector_and_weights")


def descriptive_tables() -> None:
    panel = load_panel(WORKBOOK)
    metadata = panel.metadata
    dates = panel.returns.index
    periods = [
        ("Point-model training", dates.min(), pd.Timestamp("2023-12-31")),
        ("Initial calibration", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-05-31")),
        ("Architecture validation", pd.Timestamp("2024-06-03"), pd.Timestamp("2024-12-31")),
        ("Final test", TEST_START, TEST_END),
    ]
    split_rows = []
    for name, start, end in periods:
        mask = (dates >= start) & (dates <= end)
        split_rows.append({
            "Partition": name,
            "Start": str(dates[mask].min().date()),
            "End": str(dates[mask].max().date()),
            "Sessions": int(mask.sum()),
        })
    pd.DataFrame(split_rows).to_csv(TABLES / "data_split.csv", index=False)

    eligible_meta = metadata[metadata["Original_Ticker"].isin(panel.eligible)]
    sector_counts = (
        eligible_meta.groupby("GICS_Sector")["Original_Ticker"].count().sort_values(ascending=False)
        .rename("N_Assets").reset_index().rename(columns={"GICS_Sector": "Sector"})
    )
    sector_counts.to_csv(TABLES / "sector_composition.csv", index=False)

    summary = pd.DataFrame([
        {"Quantity": "Snapshot securities", "Value": len(metadata)},
        {"Quantity": "Pre-lock eligible securities", "Value": len(panel.eligible)},
        {"Quantity": "GICS sectors", "Value": panel.sectors.nunique()},
        {"Quantity": "Full trading sessions", "Value": len(dates)},
        {"Quantity": "Valid 2025 intervals", "Value": int(
            pd.read_csv(RESULTS / "main_results.csv").set_index("Method").loc["HSA-CP", "N_Valid_Intervals"]
        )},
        {"Quantity": "Full-panel return missing rate", "Value": float(panel.returns.isna().to_numpy().mean())},
    ])
    summary.to_csv(TABLES / "dataset_summary.csv", index=False)

    cube, target, model_dates, _ = build_feature_cube(panel)
    _, _, pred, n_train = fit_point_model(panel, cube, target, model_dates)
    test = (model_dates >= TEST_START) & (model_dates <= TEST_END)
    y_test = target[test].astype(float)
    pred_test = pred[test]
    valid = np.isfinite(y_test) & np.isfinite(pred_test)
    yv, pv = y_test[valid], pred_test[valid]
    point = pd.DataFrame([{
        "N_Train_Rows": n_train,
        "N_Test_AssetDays": int(valid.sum()),
        "MAE": float(np.mean(np.abs(yv - pv))),
        "RMSE": float(np.sqrt(np.mean((yv - pv) ** 2))),
        "Sign_Accuracy": float(np.mean(np.sign(yv) == np.sign(pv))),
        "Out_Of_Sample_R2": float(1 - np.sum((yv - pv) ** 2) / np.sum((yv - np.mean(yv)) ** 2)),
    }])
    point.to_csv(TABLES / "point_forecast_metrics.csv", index=False)

    main = pd.read_csv(RESULTS / "main_results.csv")
    main[["Method", "Coverage", "Coverage_Gap", "Mean_Width", "Interval_Score", "ALCE20", "XLCE20", "Worst_Sector_Coverage", "N_Valid_Intervals"]].to_csv(
        TABLES / "main_comparison.csv", index=False
    )
    validation = pd.read_csv(RESULTS / "validation_ablation_results.csv")
    validation.to_csv(TABLES / "validation_ablation.csv", index=False)

    info = {
        "hsa_mean_expert_weights_2025": dict(zip(expert_names_for_json(), mean_weights_for_json())),
        "point_metrics": point.iloc[0].to_dict(),
    }
    (TABLES / "artifact_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def expert_names_for_json() -> list[str]:
    paths = np.load(RESULTS / "online_paths.npz")
    return [str(x) for x in paths["expert_names"]]


def mean_weights_for_json() -> list[float]:
    paths = np.load(RESULTS / "online_paths.npz")
    return [float(x) for x in paths["weights"].mean(axis=(0, 1))]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    architecture_figure()
    rolling_coverage_figure()
    robustness_and_weights_figure()
    descriptive_tables()
    print("Created:")
    for path in sorted([*FIGURES.glob("*"), *TABLES.glob("*")]):
        print(path)


if __name__ == "__main__":
    main()
