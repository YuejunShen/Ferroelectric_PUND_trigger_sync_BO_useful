"""Distributions of power-basis rise coefficients across independent sweep runs.

Convert each run's confirmed best Bernstein protocol exactly to
V(s) = a_0 + a_1 s + ... + a_n s^n, s = t / tau. No polynomial fitting is
performed. Coefficients refer to normalized time and therefore all have
units of volts; using physical time would instead multiply a_k by tau^-k.
Confirmation remeasurements of a protocol are not independent shapes and
are not counted as additional samples.

Writes a PDF, SVG and 600 dpi PNG, plus the coefficients and grouped summary
statistics as CSV files. Each panel shows one order, grouped by rise time:
dots are independent runs, boxes span the interquartile range, horizontal
lines mark medians, and whiskers span the full observed range.

    python plot_sweep_polynomial_coefficients.py --sweep-folder <folder> \
        --output-dir <folder>
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_process_figure import INK, MM, STYLE
from waveform import rise_voltage


def bernstein_to_power(controls: np.ndarray) -> np.ndarray:
    """Return ascending power coefficients a_k, with runs along the last axis.

    a_k = C(n, k) sum_{j=0}^k (-1)^(k-j) C(k, j) c_j.
    The last axis of the input contains c_0, ..., c_n.
    """
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 0 or controls.shape[-1] < 2:
        raise ValueError("Expected at least two Bernstein controls on the last axis")
    if not np.isfinite(controls).all():
        raise ValueError("Bernstein controls must be finite")
    degree = controls.shape[-1] - 1
    transform = np.zeros((degree + 1, degree + 1))
    for order in range(degree + 1):
        for index in range(order + 1):
            transform[order, index] = (math.comb(degree, order) * math.comb(order, index)
                                       * (-1) ** (order - index))
    return controls @ transform.T


def coefficient_table(sweep_folder: Path) -> pd.DataFrame:
    """One row per independent run and power order, retaining run provenance."""
    runs = pd.read_csv(sweep_folder / "sweep_all_runs.csv")
    runs = runs.loc[runs["result"] == "confirmed"].copy()
    if runs.empty:
        raise ValueError("No confirmed runs in sweep_all_runs.csv")
    if not runs["rise_basis"].eq("bernstein").all():
        raise ValueError("Exact power conversion requires Bernstein rise protocols")
    if runs["rise_points"].nunique() != 1 or runs["amplitude_v"].nunique() != 1:
        raise ValueError("Use a sweep with a single polynomial degree and amplitude")
    if runs["run_folder"].duplicated().any():
        raise ValueError("Repeated run_folder rows would double-count independent protocols")
    tau = runs["rise_time_s"].to_numpy(float)
    if not np.isfinite(tau).all() or (tau <= 0).any():
        raise ValueError("Rise times must be finite and positive")
    count = int(runs["rise_points"].iloc[0])
    if count != runs["rise_points"].iloc[0]:
        raise ValueError("rise_points must be an integer")
    controls = runs[[f"V{index}" for index in range(count)]].to_numpy(float)
    coefficients = bernstein_to_power(controls)
    # Check the conversion against the waveform implementation on every run.
    s = np.linspace(0.0, 1.0, 301)
    for control, coefficient in zip(controls, coefficients):
        np.testing.assert_allclose(np.polynomial.polynomial.polyval(s, coefficient),
                                   rise_voltage(control, s, "bernstein"),
                                   rtol=1e-9, atol=1e-10)
    metadata = runs[["run_folder", "sweep_tag", "rise_time_s", "amplitude_v", "rise_points",
                     "meets_target"]].copy()
    metadata["tau_us"] = tau * 1e6
    table = metadata.loc[metadata.index.repeat(count)].reset_index(drop=True)
    table["order"] = np.tile(np.arange(count), len(runs))
    table["coefficient_v"] = coefficients.reshape(-1)
    table["time_basis"] = "s = t / tau"
    return table.sort_values(["order", "rise_time_s", "run_folder"]).reset_index(drop=True)


def plot_polynomial_coefficients(sweep_folder: Path, output_dir: Path | None = None) -> list[Path]:
    table = coefficient_table(sweep_folder)
    summary = table.groupby(["order", "rise_time_s", "tau_us"])["coefficient_v"].agg(
        n="count", mean_v="mean", std_v="std", median_v="median", min_v="min", max_v="max",
        q25_v=lambda values: values.quantile(0.25), q75_v=lambda values: values.quantile(0.75),
    ).reset_index()
    taus = np.sort(table["tau_us"].unique())
    orders = np.sort(table["order"].unique())
    colours = plt.cm.Greens(np.linspace(0.4, 0.95, len(taus)))
    columns = min(4, len(orders))
    rows = math.ceil(len(orders) / columns)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(rows, columns, squeeze=False,
                                    figsize=(180 * MM, (25 + 43 * rows) * MM))
        for order, axis in zip(orders, axes.flat):
            subset = table.loc[table["order"] == order]
            groups = [subset.loc[subset["tau_us"] == tau, "coefficient_v"].to_numpy()
                      for tau in taus]
            boxes = axis.boxplot(groups, positions=np.arange(len(taus)), widths=0.55,
                                 patch_artist=True, whis=(0, 100), showfliers=False,
                                 medianprops={"color": INK, "linewidth": 0.8},
                                 boxprops={"linewidth": 0.6},
                                 whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
            for index, (box, colour, values) in enumerate(zip(boxes["boxes"], colours, groups)):
                box.set(facecolor=colour, alpha=0.5)
                # Deterministic offsets reveal coincident/nearby runs without
                # implying an extra dimension or duplicating confirmations.
                offsets = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else [0.0]
                axis.scatter(index + np.asarray(offsets), values, s=7, color=INK,
                             linewidths=0, alpha=0.8, zorder=3)
            axis.axhline(0, color="0.7", linewidth=0.5, linestyle=":", zorder=0)
            axis.set_xticks(np.arange(len(taus)), [f"{tau:g}" for tau in taus], rotation=45,
                            ha="right")
            axis.set(xlabel=r"Rise time $\tau$ (µs)", ylabel=fr"$a_{{{order}}}$ (V)",
                     title=fr"Order {order}")
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
            axis.margins(y=0.15)
        for axis in list(axes.flat)[len(orders):]:
            axis.set_visible(False)
        degree = len(orders) - 1
        figure.suptitle(fr"Power-basis rise coefficients: $V(s)=\sum_{{k=0}}^{{{degree}}} a_k s^k$"
                        r", $s=t/\tau$", fontsize=8, y=0.99)
        figure.text(0.5, 0.935, "Dots: confirmed best protocol per run; boxes: median and IQR; "
                    "whiskers: full range", ha="center", fontsize=6)
        counts = table.loc[table["order"] == 0].groupby("tau_us").size()
        count_text = "; ".join(f"{tau:g} µs: {n}" for tau, n in counts.items())
        figure.text(0.5, 0.015, f"Independent runs per rise time — {count_text}",
                    ha="center", fontsize=6)
        figure.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.16,
                               wspace=0.65, hspace=0.95)
        output_dir = output_dir or sweep_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / f"Fig_sweep_polynomial_coefficients.{suffix}"
                 for suffix in ("pdf", "svg", "png")]
        try:
            for path in paths:
                for attempt in range(5):
                    try:
                        figure.savefig(path, dpi=600)
                        break
                    except OSError:
                        if attempt == 4:
                            raise
                        time.sleep(1.0)
        finally:
            plt.close(figure)
    tables = [output_dir / "sweep_polynomial_coefficients.csv",
              output_dir / "sweep_polynomial_coefficient_summary.csv"]
    table.to_csv(tables[0], index=False)
    summary.to_csv(tables[1], index=False)
    return paths + tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-folder", type=Path, required=True, help="Folder with sweep_all_runs.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: the sweep folder")
    args = parser.parse_args()
    for path in plot_polynomial_coefficients(args.sweep_folder, args.output_dir):
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
