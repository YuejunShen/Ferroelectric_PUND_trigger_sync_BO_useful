"""Standalone: Work vs delta_P trade-off of a BO run, with its measured
Pareto front. Made for BOConfig.mode = "pareto", but works on any
bo_history_*.csv. Only numpy, pandas, and matplotlib are required.

Work is relative to the mean linear-rise Work, delta_P relative to the
target P_TARGET (mean linear-rise delta_P, dashed line). The reference point
(star) is "pareto" mode's hypervolume reference: the mean linear Work at
P_lower_bound = P_TARGET * (1 + p_lower_relative_tolerance), dotted line.
The highlighted point is the lowest-Work measured shape at or above the
reference delta_P. In drift-corrected runs (interleaved "linear reference"
rows) each point is instead relative to its own row's drifting reference. Runs from before linear_repeats existed may have a
pareto_reference_*.json (warm-up mean) instead, which is then used.

    python plot_pareto.py                      # newest bo_history_*.csv next to this script
    python plot_pareto.py --history <path>
    python plot_pareto.py --data-folder <path>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def latest_history(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"Data folder does not exist: {directory}")
    files = [path for path in directory.glob("bo_history_*.csv") if "eigenvectors" not in path.name]
    if not files:
        raise FileNotFoundError(f"No BO history CSV found in {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_reference(history_path: Path, history: pd.DataFrame) -> tuple[float, float, float, str]:
    """(reference Work, target delta_P, reference delta_P, label). The last
    row holds the final target (linear rows log a running mean)."""
    reference_path = history_path.with_name(
        history_path.stem.replace("bo_history_", "pareto_reference_", 1) + ".json")
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        return float(reference["Work"]), float(reference["delta_P"]), float(reference["delta_P"]), "warm-up mean"
    last = history.iloc[-1]
    linear = history[history["phase"] == "linear rise"]
    return (float(linear["Work"].mean()), float(last["P_target"]), float(last["P_lower_bound"]),
            f"mean of {len(linear)} linear rise(s)")


def pareto_mask(work: np.ndarray, delta_p: np.ndarray) -> np.ndarray:
    """True for points no other point beats on both (lower Work, higher delta_P)."""
    mask = np.ones(work.size, dtype=bool)
    for index in range(work.size):
        dominated_by = (work <= work[index]) & (delta_p >= delta_p[index]) & (
            (work < work[index]) | (delta_p > delta_p[index]))
        mask[index] = not dominated_by.any()
    return mask


def plot_pareto(history_path: Path, output_path: Path | None = None) -> Path:
    history_path = Path(history_path)
    history = pd.read_csv(history_path)
    reference_work, target_delta_p, reference_delta_p, reference_label = load_reference(history_path, history)
    references = int((history["phase"] == "linear reference").sum())
    if references:
        # Drift-corrected run (BOConfig.reference_interval): every row is
        # relative to the linear rises measured around it, not one global mean.
        work = history["Work"].to_numpy(dtype=float) / history["WORK_SCALE"].to_numpy(dtype=float)
        delta_p = (history["delta_P"].to_numpy(dtype=float) / history["P_target"].to_numpy(dtype=float)
                   - 1.0) * 100.0
        reference_label = f"drift-corrected, {references} interleaved linear reference(s)"
    else:
        work = history["Work"].to_numpy(dtype=float) / reference_work
        delta_p = (history["delta_P"].to_numpy(dtype=float) / target_delta_p - 1.0) * 100.0
    reference_x = (reference_delta_p / target_delta_p - 1.0) * 100.0
    front = pareto_mask(work, delta_p)

    figure, axis = plt.subplots(figsize=(8, 6))
    for phase, color in (("linear rise", "black"), ("linear reference", "tab:orange"),
                         ("random initial", "0.7"), ("BO optimization", "tab:blue")):
        selected = (history["phase"] == phase).to_numpy()
        axis.scatter(delta_p[selected], work[selected], s=14, color=color, label=phase, zorder=2)
    order = np.argsort(delta_p[front])
    axis.step(delta_p[front][order], work[front][order], where="post", color="tab:red",
              linewidth=1.5, label="Pareto front", zorder=3)
    axis.scatter(delta_p[front], work[front], s=30, facecolors="none", edgecolors="tab:red", zorder=4)
    axis.scatter([reference_x], [1.0], marker="*", s=200, color="black",
                 label=f"reference ({reference_label})", zorder=5)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1, label="target")
    if reference_x != 0.0:
        axis.axvline(reference_x, color="black", linestyle=":", linewidth=1, label="reference delta_P")

    eligible = np.flatnonzero(delta_p >= reference_x)
    if eligible.size:
        best = eligible[np.argmin(work[eligible])]
        axis.scatter([delta_p[best]], [work[best]], marker="D", s=80, color="tab:green", zorder=6,
                     label=f"lowest Work at delta_P >= reference (eval {int(history['evaluation'].iloc[best])})")
        axis.annotate(f"W = {work[best]:.3f}x, dP = {delta_p[best]:+.2f}%", (delta_p[best], work[best]),
                      textcoords="offset points", xytext=(8, -14))

    axis.set(xlabel="delta_P relative to target (%)", ylabel="Work / mean linear-rise Work",
             title=f"Work vs delta_P trade-off\n{history_path.name}")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path = output_path or history_path.with_name(history_path.stem + "_pareto.png")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--data-folder", type=Path, help="Folder containing bo_history_*.csv")
    selection.add_argument("--history", type=Path, help="Exact bo_history CSV")
    parser.add_argument("--output", type=Path, help="Output image path")
    args = parser.parse_args()
    history_path = args.history or latest_history(args.data_folder or Path(__file__).resolve().parent)
    print(f"Pareto front plot saved to {plot_pareto(history_path, args.output)}")


if __name__ == "__main__":
    main()
