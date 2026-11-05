"""Sector calibration comparison with day-block uncertainty.

Resampling whole trading days preserves the contemporaneous cross-section;
moving blocks preserve short-range temporal dependence.  The statistic is the
unweighted mean across 11 sectors of absolute sector coverage deviation from
the 90% target.  Width is averaged over all valid sector-asset-days.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "sector_comparison"
TARGET = 0.90
BLOCK = 10
REPS = 1000


def block_indices(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]


def metrics(coverage, width, sector_ids):
    deviations = []
    sector_widths = []
    for s in np.unique(sector_ids):
        members = sector_ids == s
        deviations.append(abs(np.nanmean(coverage[:, members]) - TARGET))
        sector_widths.append(np.nanmean(width[:, members]))
    return float(np.mean(deviations)), float(np.mean(sector_widths))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "results" / "online_paths.npz", allow_pickle=True)
    methods = z["coverage_methods"].tolist()
    cov = z["coverage_cube"]
    width = z["width_cube"]
    sector_ids = z["sector_ids"]
    n_days = cov.shape[1]
    rng = np.random.default_rng(20261103)
    samples = np.empty((REPS, len(methods), 2))
    for b in range(REPS):
        idx = block_indices(n_days, BLOCK, rng)
        for m in range(len(methods)):
            samples[b, m] = metrics(cov[m, idx], width[m, idx], sector_ids)

    h = methods.index("HSA-CP")
    rows = []
    for m, method in enumerate(methods):
        dev, wid = metrics(cov[m], width[m], sector_ids)
        # Report normal intervals from the block-bootstrap standard errors for
        # the non-smooth absolute-deviation statistic; paired percentile
        # intervals below are used for the method contrast itself.
        dev_se = float(np.std(samples[:, m, 0], ddof=1))
        width_se = float(np.std(samples[:, m, 1], ddof=1))
        dev_ci = np.array([max(0.0, dev - 1.96 * dev_se), dev + 1.96 * dev_se])
        width_ci = np.array([wid - 1.96 * width_se, wid + 1.96 * width_se])
        diff = samples[:, m, 0] - samples[:, h, 0]
        diff_ci = np.quantile(diff, [0.025, 0.975])
        rows.append({
            "Method": method,
            "Mean_Absolute_Sector_Coverage_Deviation": dev,
            "Deviation_CI_Lower": dev_ci[0],
            "Deviation_CI_Upper": dev_ci[1],
            "Mean_Sector_Width": wid,
            "Width_CI_Lower": width_ci[0],
            "Width_CI_Upper": width_ci[1],
            "Deviation_Difference_vs_HSA": dev - metrics(cov[h], width[h], sector_ids)[0],
            "Difference_CI_Lower": diff_ci[0],
            "Difference_CI_Upper": diff_ci[1],
        })
    result = pd.DataFrame(rows).sort_values("Mean_Absolute_Sector_Coverage_Deviation")
    result.to_csv(OUT / "sector_method_comparison_mbb10.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
