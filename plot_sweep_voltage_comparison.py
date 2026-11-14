"""Compare measured device voltage with the proposed Bernstein waveforms.

The seven selected optimal protocols are exactly those in panel a of the
sweep figure. Each panel overlays the confirmation-averaged measured device
voltage with the proposed rise, above their signed difference in mV. A final
panel summarizes RMS error across independent runs (mean +/- sample SD),
with selected protocols marked separately. No alignment, amplitude fitting,
or rescaling is applied. The first/last 20 ns are excluded consistently with
the measured-rise plots. Metrics use the saved traces' native time grid.
Percent errors are relative to the linear-reference amplitude, not a
pointwise division by voltage near zero. Peak errors are exported as well.

    python plot_sweep_voltage_comparison.py --sweep-folder <folder> \
        --output-dir <folder>
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_process_figure import INK, MM, STYLE
from plot_sweep_polynomial_coefficients import load_confirmed_runs, select_optimal_protocols
from plot_sweep_results import TRIM_S
from plot_sweep_waveform_statistics import mean_measured_voltage
from waveform import rise_voltage


def comparison_data(sweep_folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = load_confirmed_runs(sweep_folder)
    if runs["run_folder"].duplicated().any() or not runs["rise_basis"].eq("bernstein").all():
        raise ValueError("Expected one confirmed Bernstein protocol per independent run")
    selected = set(select_optimal_protocols(runs)["run_folder"])
    tables, records = [], []
    for _, run in runs.iterrows():
        paths = [path for path in str(run["rise_files"]).split(";") if path and path != "nan"]
        if not paths:
            raise ValueError(f"No measured confirmation traces: {run['run_folder']}")
        tau_s, amplitude = float(run["rise_time_s"]), float(run["amplitude_v"])
        if not np.isfinite(tau_s) or tau_s <= 2 * TRIM_S or not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("Rise time and reference amplitude must be positive and finite")
        t = pd.read_csv(paths[0], usecols=["rise_time_s"])["rise_time_s"].to_numpy(float)
        t = t[(t >= TRIM_S) & (t <= tau_s - TRIM_S)]
        if len(t) < 2:
            raise ValueError("Insufficient measured data after edge trimming")
        s = t / tau_s
        controls = run[[f"V{k}" for k in range(int(run["rise_points"]))]].to_numpy(float)
        if not np.isfinite(controls).all():
            raise ValueError("Proposed Bernstein controls must be finite")
        proposed = rise_voltage(controls, s, "bernstein")
        measured = mean_measured_voltage(run["rise_files"], tau_s, s)
        difference = measured - proposed
        rms, peak = float(np.sqrt(np.mean(difference ** 2))), float(np.abs(difference).max())
        is_selected = run["run_folder"] in selected
        tables.append(pd.DataFrame({
            "run_folder": run["run_folder"], "tau_us": run["tau_us"],
            "time_in_rise_s": t, "normalized_time": s, "is_selected_optimal": is_selected,
            "proposed_voltage_v": proposed, "measured_device_voltage_v": measured,
            "difference_v": difference,
        }))
        records.append({
            "run_folder": run["run_folder"], "tau_us": run["tau_us"], "rise_time_s": tau_s,
            "is_selected_optimal": is_selected, "n_confirmation_traces": len(paths),
            "meets_target": bool(run["meets_target"]),
            "n_samples": len(t), "rms_error_v": rms, "max_abs_error_v": peak,
            "bias_v": float(difference.mean()), "reference_amplitude_v": amplitude,
            "rms_pct_reference": 100 * rms / amplitude, "max_abs_pct_reference": 100 * peak / amplitude,
            "edge_trim_s": TRIM_S, "voltage_column": "rise_voltage_v", "rise_files": run["rise_files"],
        })
    metrics = pd.DataFrame(records)
    summary = metrics.groupby("tau_us").agg(
        n_runs=("run_folder", "count"), rms_mean_v=("rms_error_v", "mean"),
        rms_std_v=("rms_error_v", "std"), max_abs_mean_v=("max_abs_error_v", "mean"),
        max_abs_std_v=("max_abs_error_v", "std"), rms_mean_pct_reference=("rms_pct_reference", "mean"),
        rms_std_pct_reference=("rms_pct_reference", "std"),
    ).reset_index()
    return pd.concat(tables, ignore_index=True), metrics, summary


def plot_voltage_comparison(sweep_folder: Path, output_dir: Path | None = None) -> list[Path]:
    traces, metrics, summary = comparison_data(sweep_folder)
    selected = metrics.loc[metrics["is_selected_optimal"]].sort_values("tau_us")
    colours = plt.cm.Greens(np.linspace(0.4, 0.95, len(selected)))
    selected_traces = traces.loc[traces["is_selected_optimal"]]
    voltage_min = min(0, selected_traces[["proposed_voltage_v", "measured_device_voltage_v"]].min().min())
    voltage_max = selected_traces[["proposed_voltage_v", "measured_device_voltage_v"]].max().max()
    residual_limit = max(1, 1.12 * selected["max_abs_error_v"].max() * 1000)
    columns = min(4, len(selected) + 1)
    rows = math.ceil((len(selected) + 1) / columns)
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(180 * MM, (25 + 59 * rows) * MM))
        grid = figure.add_gridspec(rows, columns, left=0.075, right=0.99, top=0.84,
                                   bottom=0.105, wspace=0.6, hspace=0.55)
        for index, ((_, protocol), colour) in enumerate(zip(selected.iterrows(), colours)):
            pair = grid[index // columns, index % columns].subgridspec(
                2, 1, height_ratios=(1.7, 1.0), hspace=0.12)
            overlay = figure.add_subplot(pair[0])
            residual = figure.add_subplot(pair[1], sharex=overlay)
            subset = traces.loc[traces["run_folder"] == protocol["run_folder"]]
            s = subset["normalized_time"]
            overlay.plot(s, subset["measured_device_voltage_v"], color=colour, linewidth=1.2)
            overlay.plot(s, subset["proposed_voltage_v"], color=INK, linestyle="--", linewidth=0.8)
            overlay.set(title=fr"$\tau={protocol['tau_us']:g}$ µs", ylabel="$V$ (V)",
                        ylim=(voltage_min - 0.04, voltage_max * 1.22))
            overlay.tick_params(axis="x", labelbottom=False)
            overlay.text(0.03, 0.97, f"RMS {protocol['rms_error_v'] * 1000:.1f} mV "
                         f"({protocol['rms_pct_reference']:.2f}%)\n"
                         f"Max {protocol['max_abs_error_v'] * 1000:.1f} mV",
                         transform=overlay.transAxes, va="top", fontsize=5)
            residual.plot(s, subset["difference_v"] * 1000, color=colour, linewidth=0.8)
            residual.axhline(0, color="0.6", linestyle=":", linewidth=0.5)
            residual.set(xlabel=r"$s=t/\tau$", ylabel="Difference (mV)", xlim=(0, 1),
                         ylim=(-residual_limit, residual_limit))
            residual.set_xticks([0, 0.5, 1])
            residual.yaxis.set_major_locator(MaxNLocator(3))
            for axis in (overlay, residual):
                axis.yaxis.label.set_size(6)
                axis.tick_params(labelsize=5)
                axis.title.set_fontsize(7)
        index = len(selected)
        errors = figure.add_subplot(grid[index // columns, index % columns])
        for (_, group), colour in zip(summary.iterrows(), colours):
            errors.errorbar(group["tau_us"], group["rms_mean_v"] * 1000,
                            yerr=group["rms_std_v"] * 1000 if group["n_runs"] > 1 else None,
                            color=colour, marker="o", markersize=3.5, linestyle="none",
                            elinewidth=0.7, capsize=2)
        errors.scatter(selected["tau_us"], selected["rms_error_v"] * 1000,
                       color=INK, marker="x", s=12, linewidths=0.7, zorder=3)
        errors.set_xscale("log")
        errors.set_xticks(summary["tau_us"], [f"{tau:g}" for tau in summary["tau_us"]],
                          rotation=45, ha="right")
        errors.minorticks_off()
        errors.set(xlabel=r"Rise time $\tau$ (µs)", ylabel="RMS difference (mV)",
                   title="Across-run agreement", ylim=(0, None))
        errors.xaxis.label.set_size(6)
        errors.yaxis.label.set_size(6)
        errors.title.set_fontsize(6)
        errors.tick_params(labelsize=5)
        errors.legend(handles=[Line2D([], [], color="#2f9e6e", marker="o", linestyle="none",
                                       markersize=3, label="Runs: mean ± SD"),
                               Line2D([], [], color=INK, marker="x", linestyle="none",
                                       markersize=3, label="Selected optimal")],
                      loc="upper right", fontsize=5)
        figure.suptitle("Measured versus proposed optimal-rise voltage", fontsize=8, y=0.99)
        figure.legend(handles=[Line2D([], [], color="#2f9e6e", linewidth=1.2,
                                       label="Measured device voltage"),
                               Line2D([], [], color=INK, linestyle="--", linewidth=0.8,
                                       label="Proposed Bernstein waveform")],
                      loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, columnspacing=2)
        figure.text(0.5, 0.895, "Selected protocols match figure a; residual = measured − proposed.",
                    ha="center", fontsize=6)
        figure.text(0.5, 0.038, "Measured voltage: confirmation-averaged device voltage with configured "
                    "sense-resistor correction; first/last 20 ns excluded.", ha="center", fontsize=5.5)
        figure.text(0.5, 0.015, "RMS percentages use the linear-reference amplitude. "
                    "No time alignment, voltage fitting, or amplitude rescaling applied.", ha="center", fontsize=5.5)
        output_dir = output_dir or sweep_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / f"Fig_sweep_measured_vs_proposed.{suffix}" for suffix in ("pdf", "svg", "png")]
        try:
            for path in paths:
                for attempt in range(5):
                    try:
                        figure.savefig(path, dpi=600)
                        break
                    except OSError:
                        if attempt == 4:
                            raise
                        time.sleep(1)
        finally:
            plt.close(figure)
    paths.extend([output_dir / "sweep_measured_vs_proposed_traces.csv",
                  output_dir / "sweep_measured_vs_proposed_metrics.csv",
                  output_dir / "sweep_measured_vs_proposed_summary.csv"])
    traces.to_csv(paths[-3], index=False)
    metrics.to_csv(paths[-2], index=False)
    summary.to_csv(paths[-1], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in plot_voltage_comparison(args.sweep_folder, args.output_dir):
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
