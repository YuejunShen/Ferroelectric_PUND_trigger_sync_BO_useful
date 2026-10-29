"""Standalone: plots a hyperparameter-convergence curve from a run_parameter_sweep.py
sweep. X axis is a swept run_log.csv column (default: rise_points). Y axis is
the ratio of that run's best feasible Work to its own linear-rise (evaluation 0)
reference Work -- how much the BO-optimized protocol reduced switching work
relative to the un-optimized linear ramp, for that parameter value. When a
parameter value was run more than once (REPEATS > 1 in run_parameter_sweep.py),
its runs are averaged and drawn with an error bar (+/- 1 std).

Reads D:\\Ferroelectric_BTO_optimization\\run_log.csv (every run ever logged)
and, for each row, that run's own bo_history_*.csv. Only numpy, pandas, and
matplotlib are required. run_parameter_sweep.py calls plot_convergence()
directly at the end of a sweep; run this file standalone for any other run_log
selection.

    python plot_sweep_convergence.py
    python plot_sweep_convergence.py --sweep-folder "D:\\Ferroelectric_BTO_optimization\\rise_points_scan"
    python plot_sweep_convergence.py --sweep-label-prefix rise_points
    python plot_sweep_convergence.py --x-column bo_steps
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RUN_LOG = Path(r"D:\Ferroelectric_BTO_optimization\run_log.csv")


def work_ratio_for_run(output_directory: Path) -> float | None:
    """Best feasible Work reached during the run, divided by that run's own
    linear-rise (evaluation 0) reference Work. None if the run has no usable data."""
    history_files = list(Path(output_directory).glob("bo_history_*.csv"))
    if not history_files:
        print(f"  no bo_history_*.csv in {output_directory}, skipping")
        return None
    history = pd.read_csv(history_files[0])
    if history.empty or "Work" not in history:
        return None
    linear_rows = history[history["phase"] == "linear rise"] if "phase" in history else history.iloc[:0]
    linear_work = (linear_rows["Work"].iloc[0] if not linear_rows.empty
                   else history.sort_values("evaluation")["Work"].iloc[0])
    if linear_work == 0:
        return None
    feasible = history[history["feasible"]] if "feasible" in history else history
    if feasible.empty:
        feasible = history
    return feasible["Work"].min() / linear_work


def plot_convergence(run_log_path: Path, x_column: str = "rise_points",
                      sweep_folder: Path | None = None, sweep_label_prefix: str | None = None,
                      output_path: Path | None = None) -> Path:
    """Filter run_log_path's rows to one sweep (by sweep_folder -- every run whose
    output_directory lives directly under it -- or by sweep_label_prefix, or neither
    for "every logged run"), then plot the Work-ratio-vs-x_column convergence curve."""
    log = pd.read_csv(run_log_path)
    if sweep_folder is not None:
        sweep_folder = Path(sweep_folder).resolve()
        log = log[log["output_directory"].apply(lambda p: Path(p).resolve().parent == sweep_folder)]
    if sweep_label_prefix:
        log = log[log["sweep_label"].astype(str).str.startswith(sweep_label_prefix)]
    if log.empty:
        raise ValueError("No matching rows in run_log.csv")

    log = log.assign(work_ratio=[work_ratio_for_run(Path(p)) for p in log["output_directory"]])
    log = log.dropna(subset=["work_ratio"])
    if log.empty:
        raise ValueError("No runs had a computable Work ratio (missing/empty bo_history_*.csv?)")

    # Runs logged before rise_basis existed always used "points". One curve per basis.
    if "rise_basis" not in log:
        log["rise_basis"] = "points"
    log["rise_basis"] = log["rise_basis"].fillna("points")

    figure, axis = plt.subplots(figsize=(7, 5))
    for basis, basis_log in log.groupby("rise_basis"):
        grouped = basis_log.groupby(x_column)["work_ratio"]
        x = grouped.mean().index.to_numpy(float)
        means = grouped.mean().to_numpy(float)
        stds = grouped.std(ddof=0).fillna(0.0).to_numpy(float)  # 0 for single-repeat points
        counts = grouped.count().to_numpy(int)
        order = np.argsort(x)
        x, means, stds, counts = x[order], means[order], stds[order], counts[order]
        axis.errorbar(x, means, yerr=stds, marker="o", markersize=6, capsize=4,
                      linewidth=1.5, zorder=3, label=f"{basis} basis")
        for xi, yi, ni in zip(x, means, counts):
            if ni > 1:
                axis.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                              fontsize=8, ha="center", color="0.35")
    axis.axhline(1.0, color="0.6", linestyle="--", linewidth=1, label="linear-rise baseline", zorder=1)
    axis.set(xlabel=x_column, ylabel="Best feasible Work / linear-rise Work",
             title=f"BO convergence vs. {x_column}")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)

    if output_path is None:
        output_dir = sweep_folder if sweep_folder is not None else run_log_path.parent
        output_path = output_dir / "sweep_work_ratio_convergence.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-log", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--x-column", default="rise_points",
                        help="run_log.csv column to sweep over on the x axis (default: rise_points)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sweep-folder", type=Path,
                           help="Only include runs whose output folder lives directly under this sweep folder")
    selection.add_argument("--sweep-label-prefix", default=None,
                           help="Only include rows whose sweep_label starts with this")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to save the PNG (default: inside --sweep-folder, else next to --run-log)")
    args = parser.parse_args()
    output_path = plot_convergence(args.run_log, args.x_column, args.sweep_folder,
                                   args.sweep_label_prefix, args.output)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
