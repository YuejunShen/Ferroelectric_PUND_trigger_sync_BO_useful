"""Power-basis coefficient statistics and optimal-rise component decomposition.

Convert each independent run's confirmed best Bernstein protocol exactly to
V(s) = a_0 + a_1 s + ... + a_n s^n, s = t / tau. No polynomial fitting is
performed. Coefficients refer to normalized time and therefore all have
units of volts; using physical time would instead multiply a_k by tau^-k.
The coefficient figure shows mean +/- sample standard deviation across runs
at each rise time, one panel per order, on linear coefficient axes.
The component figure uses only the single optimal protocol per rise time
shown in panel a of the sweep figure (largest confirmed work saving). It pairs
the total rise with signed stacked polynomial terms, and summarizes absolute
term areas |a_k|/(k+1). Their percentage shares describe term magnitudes,
not work or a positive partition of the net voltage: signed terms cancel.
Writes PDF/SVG/PNG figures and CSV data with run provenance.

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
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from waveform import rise_voltage


def bernstein_to_power(controls: np.ndarray) -> np.ndarray:
    """Return ascending power coefficients a_k on the last axis.

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


def load_confirmed_runs(sweep_folder: Path) -> pd.DataFrame:
    """Load and order confirmed runs identically for all sweep figure panels."""
    runs = pd.read_csv(sweep_folder / "sweep_all_runs.csv")
    runs = runs.loc[runs["result"] == "confirmed"].copy()
    if runs.empty:
        raise ValueError("No confirmed runs in sweep_all_runs.csv")
    runs["tau_us"] = (runs["rise_time_s"] * 1e6).round(2)
    runs["repeat"] = runs["sweep_tag"].str.extract(r"rep(\d+)", expand=False).astype(int)
    return runs.sort_values(["tau_us", "repeat"]).reset_index(drop=True)


def select_optimal_protocols(runs: pd.DataFrame) -> pd.DataFrame:
    """One protocol per tau: prefer target-meeting runs, then largest saving.

    This matches best_per_setting in plot_sweep_results. If no run meets the
    target, retain the largest-saving run with its false meets_target flag.
    Across-run statistics still include all complete confirmed runs.
    """
    return pd.DataFrame([
        runs[runs["tau_us"] == tau].sort_values(["meets_target", "work_reduction_pct"]).iloc[-1]
        for tau in sorted(runs["tau_us"].unique())
    ]).reset_index(drop=True)


def coefficient_table(sweep_folder: Path, optimal_only: bool = False) -> pd.DataFrame:
    """One row per run and power order; optionally use just panel a's optima."""
    runs = load_confirmed_runs(sweep_folder)
    if not runs["rise_basis"].eq("bernstein").all():
        raise ValueError("Exact power conversion requires Bernstein rise protocols")
    if runs["rise_points"].nunique() != 1 or runs["amplitude_v"].nunique() != 1:
        raise ValueError("Use a sweep with a single polynomial degree and amplitude")
    if runs["run_folder"].duplicated().any():
        raise ValueError("Repeated run_folder rows would double-count independent protocols")
    selected_folders = set(select_optimal_protocols(runs)["run_folder"])
    if optimal_only:
        runs = select_optimal_protocols(runs)
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
    metadata = runs[["run_folder", "sweep_tag", "rise_time_s", "tau_us", "amplitude_v", "rise_points",
                     "meets_target", "work_reduction_pct", "candidate", "source_evaluation"]].copy()
    metadata["is_selected_optimal"] = metadata["run_folder"].isin(selected_folders)
    table = metadata.loc[metadata.index.repeat(count)].reset_index(drop=True)
    table["order"] = np.tile(np.arange(count), len(runs))
    table["coefficient_v"] = coefficients.reshape(-1)
    table["time_basis"] = "s = t / tau"
    return table.sort_values(["order", "rise_time_s", "run_folder"]).reset_index(drop=True)


def plot_polynomial_coefficients(sweep_folder: Path, output_dir: Path | None = None) -> list[Path]:
    table = coefficient_table(sweep_folder)
    summary = table.groupby(["order", "rise_time_s", "tau_us"])["coefficient_v"].agg(
        n="count", mean_v="mean", std_v="std").reset_index()
    taus = np.sort(table["tau_us"].unique())
    orders = np.sort(table["order"].unique())
    colours = plt.cm.Greens(np.linspace(0.4, 0.95, len(taus)))
    columns = min(4, len(orders))
    rows = math.ceil(len(orders) / columns)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(rows, columns, squeeze=False,
                                    figsize=(180 * MM, (25 + 43 * rows) * MM))
        for order, axis in zip(orders, axes.flat):
            subset = summary.loc[summary["order"] == order].sort_values("tau_us")
            axis.plot(taus, subset["mean_v"], color="0.65", linewidth=0.6, zorder=1)
            for (_, row), colour in zip(subset.iterrows(), colours):
                axis.errorbar(row["tau_us"], row["mean_v"],
                              yerr=row["std_v"] if row["n"] > 1 else None,
                              color=colour, marker="o", markersize=3.2, linestyle="none",
                              elinewidth=0.8, capsize=2, markeredgecolor=INK,
                              markeredgewidth=0.3, zorder=3)
            axis.axhline(0, color="0.7", linewidth=0.5, linestyle=":", zorder=0)
            axis.set_xscale("log")
            axis.set_xticks(taus, [f"{tau:g}" for tau in taus], rotation=45,
                            ha="right")
            axis.minorticks_off()
            axis.set(xlabel=r"Rise time $\tau$ (µs)", ylabel=fr"$a_{{{order}}}$ (V)",
                     title=fr"Order {order}")
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
            axis.margins(y=0.15)
        for axis in list(axes.flat)[len(orders):]:
            axis.set_visible(False)
        degree = len(orders) - 1
        figure.suptitle(fr"Commanded power-basis coefficients: $V(s)=a_0+a_1s+\cdots+a_{{{degree}}}s^{{{degree}}}$"
                        r", $s=t/\tau$", fontsize=8, y=0.99)
        figure.text(0.5, 0.935, "Mean ± sample standard deviation across runs' confirmed best protocols",
                    ha="center", fontsize=6)
        counts = table.loc[table["order"] == 0].groupby("tau_us").size()
        count_text = "; ".join(f"{tau:g} µs: {n}" for tau, n in counts.items())
        figure.text(0.5, 0.015, f"Independent runs per rise time — {count_text}", ha="center", fontsize=6)
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
    return paths + tables + plot_polynomial_components(sweep_folder, output_dir)


def plot_polynomial_components(sweep_folder: Path, output_dir: Path | None = None) -> list[Path]:
    """Signed monomial contributions to exactly the optimal curves in panel a.

    Band thickness is a term's signed voltage, not its electrical work.
    Absolute integrated term magnitude is |a_k|/(k+1) on s in [0,1].
    """
    table = coefficient_table(sweep_folder, optimal_only=True)
    taus = np.sort(table["tau_us"].unique())
    orders = np.sort(table["order"].unique())
    colours = plt.cm.tab10(np.linspace(0, 1, 10))[:len(orders)]
    if len(orders) > 10:
        colours = plt.cm.turbo(np.linspace(0, 1, len(orders)))
    greens = plt.cm.Greens(np.linspace(0.4, 0.95, len(taus)))
    s = np.linspace(0, 1, 401)
    protocols, trace_tables, areas = [], [], []
    for tau in taus:
        rows = table.loc[table["tau_us"] == tau].sort_values("order")
        coefficients = rows["coefficient_v"].to_numpy()
        terms = coefficients[:, None] * s[None, :] ** orders[:, None]
        total = terms.sum(axis=0)
        np.testing.assert_allclose(total, np.polynomial.polynomial.polyval(s, coefficients),
                                   atol=1e-10, rtol=1e-9)
        protocols.append((terms, total, rows.iloc[0]))
        trace = pd.DataFrame({"tau_us": tau, "normalized_time": s, "total_voltage_v": total,
                              "run_folder": rows.iloc[0]["run_folder"]})
        for order, values in zip(orders, terms):
            trace[f"term_{order}_v"] = values
        trace_tables.append(trace)
        magnitude = np.abs(coefficients) / (orders + 1)
        denominator = magnitude.sum()
        for order, coefficient, area in zip(orders, coefficients, magnitude):
            areas.append({"tau_us": tau, "order": order, "coefficient_v": coefficient,
                          "signed_term_area_v": coefficient / (order + 1),
                          "absolute_term_area_v": area,
                          "absolute_term_area_share_pct": 100 * area / denominator if denominator else 0,
                          "run_folder": rows.iloc[0]["run_folder"]})
    areas = pd.DataFrame(areas)
    peak = max(max(np.maximum(terms, 0).sum(axis=0).max(),
                   -np.minimum(terms, 0).sum(axis=0).min()) for terms, _, _ in protocols)
    rise_peak = max(float(table["amplitude_v"].max()), max(total.max() for _, total, _ in protocols))
    columns = 4
    rows_count = math.ceil((len(taus) + 1) / columns)
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(180 * MM, (25 + 60 * rows_count) * MM))
        grid = figure.add_gridspec(rows_count, columns, left=0.075, right=0.99, top=0.83,
                                   bottom=0.08, wspace=0.65, hspace=0.55)
        for index, ((terms, total, metadata), colour) in enumerate(zip(protocols, greens)):
            pair = grid[index // columns, index % columns].subgridspec(
                2, 1, height_ratios=(1, 1.3), hspace=0.12)
            shape = figure.add_subplot(pair[0])
            components = figure.add_subplot(pair[1], sharex=shape)
            shape.plot(s, total, color=colour, linewidth=1.2)
            shape.set(title=fr"$\tau={metadata['tau_us']:g}$ µs", ylabel="$V$ (V)",
                      ylim=(0, rise_peak * 1.08))
            shape.tick_params(axis="x", labelbottom=False)
            positive, negative = np.zeros_like(s), np.zeros_like(s)
            for order, values, term_colour in zip(orders, terms, colours):
                above, below = np.maximum(values, 0), np.minimum(values, 0)
                components.fill_between(s, positive, positive + above, facecolor=term_colour,
                                        alpha=0.85, linewidth=0)
                components.fill_between(s, negative, negative + below, facecolor=term_colour,
                                        alpha=0.85, linewidth=0)
                positive += above
                negative += below
            components.plot(s, total, color=INK, linewidth=0.7)
            components.axhline(0, color="0.5", linewidth=0.4, zorder=0)
            components.set(xlabel=r"$s=t/\tau$", ylabel="Term bands (V)", xlim=(0, 1),
                           ylim=(-peak * 1.05 if peak else -1, peak * 1.05 if peak else 1))
            components.set_xticks([0, 0.5, 1])
            components.yaxis.set_major_locator(MaxNLocator(3))
            for axis in (shape, components):
                axis.yaxis.label.set_size(6)
                axis.tick_params(labelsize=5)
                axis.title.set_fontsize(7)
        index = len(taus)
        shares = figure.add_subplot(grid[index // columns, index % columns])
        left = np.zeros(len(taus))
        for order, colour in zip(orders, colours):
            values = areas.loc[areas["order"] == order].sort_values("tau_us")["absolute_term_area_share_pct"]
            shares.barh(np.arange(len(taus)), values, left=left, color=colour, height=0.7)
            left += values.to_numpy()
        shares.set_yticks(np.arange(len(taus)), [f"{tau:g}" for tau in taus])
        shares.invert_yaxis()
        shares.set(xlim=(0, 100), xlabel="Absolute term-area share (%)",
                   ylabel=r"Rise time $\tau$ (µs)", title="Integrated term magnitudes")
        shares.set_xticks([0, 50, 100])
        shares.xaxis.label.set_size(6)
        shares.yaxis.label.set_size(6)
        shares.title.set_fontsize(6)
        figure.suptitle("Commanded polynomial components of the selected optimal rises", fontsize=8, y=0.99)
        figure.text(0.5, 0.945, r"Upper: total rise. Lower: signed term bands $a_ks^k$; black: their sum.",
                    ha="center", fontsize=6)
        handles = [Patch(facecolor=colour, label=fr"$k={order}$") for order, colour in zip(orders, colours)]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
                      ncol=len(orders), handlelength=1.2, columnspacing=1.1)
        figure.text(0.5, 0.015, r"Absolute area $=|a_k|/(k+1)$ over $0\leq s\leq1$; "
                    "positive and negative terms cancel. Shares describe term size, not electrical work.",
                    ha="center", fontsize=5.5)
        output_dir = output_dir or sweep_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / f"Fig_sweep_polynomial_components.{suffix}" for suffix in ("pdf", "svg", "png")]
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
    paths.extend([output_dir / "sweep_polynomial_component_traces.csv",
                  output_dir / "sweep_polynomial_component_areas.csv"])
    pd.concat(trace_tables, ignore_index=True).to_csv(paths[-2], index=False)
    areas.to_csv(paths[-1], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-folder", type=Path, required=True, help="Folder with sweep_all_runs.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: the sweep folder")
    args = parser.parse_args()
    for path in plot_polynomial_coefficients(args.sweep_folder, args.output_dir):
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
