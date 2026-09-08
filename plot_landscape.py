"""Standalone: copy this single file into a folder containing bo_history_*.csv
and run it there. No dependency on the rest of this experiment's package
(config.py, waveform.py, instruments.py, ...) -- only numpy, pandas, and
matplotlib are required.

The BO search space is rise_points-dimensional (one voltage knot V0..Vn per
point on the shared PUND rise shape -- see BOConfig.rise_points in config.py).
That can't be viewed directly in 2D, so this script principal-component-
decomposes the sampled V0..Vn vectors and projects every evaluation onto its
top two "eigenmodes" (principal components of the voltage-knot covariance
matrix). It then draws two 2D landscape plots on those same PC1/PC2 axes,
one colored by Work and one by delta_P, so the two objectives can be
compared on an identical map of the search space.

    python plot_landscape.py                      # uses the newest
                                                    # bo_history_*.csv next
                                                    # to this script
    python plot_landscape.py --history <path>
    python plot_landscape.py --data-folder <path>
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

V_COLUMN = re.compile(r"^V\d+$")


def latest_history(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"Data folder does not exist: {directory}")
    files = list(directory.glob("bo_history_*.csv"))
    if not files:
        raise FileNotFoundError(f"No BO history CSV found in {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def voltage_knot_columns(history: pd.DataFrame) -> list[str]:
    columns = [c for c in history.columns if V_COLUMN.match(c)]
    if len(columns) < 2:
        raise ValueError(f"Need at least 2 voltage-knot columns (V0, V1, ...), found: {columns}")
    return sorted(columns, key=lambda c: int(c[1:]))


def principal_eigenmodes(history: pd.DataFrame, columns: list[str]):
    """Project each row's voltage-knot vector onto the top-2 principal
    components ("eigenmodes") of the sampled V0..Vn covariance matrix."""
    knots = history[columns].to_numpy(float)
    mean = knots.mean(axis=0)
    centered = knots - mean
    # SVD of the centered design matrix gives the eigenvectors of the
    # covariance matrix directly, without forming it explicitly.
    _, singular_values, eigenvectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values ** 2
    explained = variance / variance.sum()
    scores = centered @ eigenvectors[:2].T
    return scores[:, 0], scores[:, 1], explained[:2]


def _scatter_landscape(axis, x, y, values, label, feasible=None):
    norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    color_map = plt.get_cmap("viridis")
    if feasible is None:
        feasible = np.ones_like(values, dtype=bool)
    axis.scatter(x[feasible], y[feasible], c=values[feasible], cmap=color_map, norm=norm,
                 marker="o", s=45, edgecolors="k", linewidths=0.4, zorder=2)
    if (~feasible).any():
        axis.scatter(x[~feasible], y[~feasible], c=values[~feasible], cmap=color_map, norm=norm,
                     marker="x", s=45, linewidths=1.2, zorder=2)
    axis.set(xlabel="Eigenmode 1 (PC1)", ylabel="Eigenmode 2 (PC2)")
    axis.grid(True, alpha=0.3)
    colorbar = plt.colorbar(ScalarMappable(norm=norm, cmap=color_map), ax=axis)
    colorbar.set_label(label)
    return axis


def plot_landscape(history_path: Path, output_dir: Path | None = None) -> Path:
    history = pd.read_csv(history_path).sort_values("evaluation").reset_index(drop=True)
    if history.empty:
        raise ValueError("History is empty")
    for required in ("Work", "delta_P"):
        if required not in history:
            raise ValueError(f"History is missing '{required}' column")

    columns = voltage_knot_columns(history)
    pc1, pc2, explained = principal_eigenmodes(history, columns)
    feasible = history["feasible"].to_numpy(bool) if "feasible" in history else None

    output_dir = output_dir or history_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, (ax_work, ax_dp) = plt.subplots(1, 2, figsize=(13, 5.5))
    _scatter_landscape(ax_work, pc1, pc2, history["Work"].to_numpy(float), "Work (J)", feasible)
    ax_work.set_title("Work landscape")
    _scatter_landscape(ax_dp, pc1, pc2, history["delta_P"].to_numpy(float), "delta_P (C/m^2)", feasible)
    ax_dp.set_title("delta_P landscape")

    figure.suptitle(
        f"BO landscape over top-2 eigenmodes of {columns[0]}..{columns[-1]} "
        f"(explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%})"
    )
    if feasible is not None and (~feasible).any():
        figure.text(0.5, 0.01, "o = feasible   x = infeasible", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.03, 1, 1))

    output_path = output_dir / f"{history_path.stem}_landscape.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--data-folder", type=Path, help="Folder containing the run CSV files")
    selection.add_argument("--history", type=Path, help="Exact timestamped bo_history CSV")
    parser.add_argument("--output-dir", type=Path, help="Where to save the PNG (default: next to the history CSV)")
    args = parser.parse_args()
    data_folder = args.data_folder or Path(__file__).resolve().parent
    history_path = (args.history or latest_history(data_folder)).resolve()
    output_path = plot_landscape(history_path, args.output_dir)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
