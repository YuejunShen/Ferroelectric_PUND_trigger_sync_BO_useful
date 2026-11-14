"""Mean commanded or measured rise +/- sample standard deviation across runs.

One confirmed best protocol per independent run, evaluated on a common
normalized-time grid s=t/tau using the saved Bernstein controls. Each rise
time gets a panel with a mean curve, a +/-1 sample-SD band (ddof=1), the
selected optimal protocol from panel a of the sweep figure, and the linear
reference. An overview compares mean shapes. Bands describe across-run
variation, not uncertainty of the mean. With --voltage-source measured,
average saved scope-derived device-voltage confirmation traces within each
run before across-run statistics. Exclude 20 ns at each rise edge. Use measured
confirmation references too; the CSV records the source and trimming.

    python plot_sweep_waveform_statistics.py --sweep-folder <folder> \
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
from matplotlib.patches import Patch

from plot_process_figure import INK, MM, STYLE
from plot_sweep_polynomial_coefficients import load_confirmed_runs, select_optimal_protocols
from plot_sweep_results import TRIM_S
from waveform import rise_voltage


def mean_measured_voltage(files: str, rise_time_s: float, normalized_time: np.ndarray) -> np.ndarray:
    """Average scope-derived device voltage within one run, on a common grid.

    rise_voltage_v is the voltage used for Work: measured probe voltage minus
    the sense-resistor drop when that run enabled correction. Never substitute
    a commanded waveform or extrapolate beyond a measured trace.
    """
    paths = [path for path in str(files).split(";") if path and path != "nan"]
    if not paths:
        raise ValueError("No measured rise traces available")
    query = np.asarray(normalized_time, dtype=float) * rise_time_s
    voltages = []
    for path in paths:
        trace = pd.read_csv(path, usecols=["rise_time_s", "rise_voltage_v"])
        t, voltage = trace["rise_time_s"].to_numpy(float), trace["rise_voltage_v"].to_numpy(float)
        if len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(voltage).all() or (np.diff(t) <= 0).any():
            raise ValueError(f"Invalid measured voltage trace: {path}")
        if query.min() < t[0] - 1e-15 or query.max() > t[-1] + 1e-15:
            raise ValueError(f"Requested grid extends beyond measured trace: {path}")
        voltages.append(np.interp(query, t, voltage))
    return np.mean(voltages, axis=0)


def waveform_statistics(sweep_folder: Path, voltage_source: str = "commanded") -> tuple[pd.DataFrame, pd.DataFrame]:
    if voltage_source not in ("commanded", "measured"):
        raise ValueError(f"Unknown voltage source: {voltage_source}")
    runs = load_confirmed_runs(sweep_folder)
    if runs["run_folder"].duplicated().any():
        raise ValueError("Each independent run must appear only once")
    if not runs["rise_basis"].eq("bernstein").all():
        raise ValueError("Expected Bernstein rise protocols")
    if runs["rise_points"].nunique() != 1 or runs["amplitude_v"].nunique() != 1:
        raise ValueError("Use a sweep with a single polynomial degree and amplitude")
    if not np.isfinite(runs["rise_time_s"]).all() or (runs["rise_time_s"] <= 0).any():
        raise ValueError("Rise times must be finite and positive")
    count = int(runs["rise_points"].iloc[0])
    inputs = [f"V{k}" for k in range(count)]
    if not np.isfinite(runs[inputs].to_numpy(float)).all():
        raise ValueError("Waveform controls must be finite")
    selected = select_optimal_protocols(runs)
    records = []
    for _, optimal in selected.iterrows():
        subset = runs.loc[runs["tau_us"] == optimal["tau_us"]]
        tau_s = float(optimal["rise_time_s"])
        trim = TRIM_S if voltage_source == "measured" else 0.0
        if tau_s <= 2 * trim:
            raise ValueError("Rise time is too short for the measured edge trim")
        s = np.linspace(trim / tau_s, 1 - trim / tau_s, 401)
        if voltage_source == "measured":
            shapes = np.array([mean_measured_voltage(files, tau_s, s) for files in subset["rise_files"]])
            linear_shapes = np.array([mean_measured_voltage(files, tau_s, s)
                                      for files in subset["linear_rise_files"]])
            optimal_shape = mean_measured_voltage(optimal["rise_files"], tau_s, s)
            linear_shape = linear_shapes.mean(axis=0)
        else:
            shapes = np.array([rise_voltage(controls, s, "bernstein")
                               for controls in subset[inputs].to_numpy(float)])
            optimal_shape = rise_voltage(optimal[inputs].to_numpy(float), s, "bernstein")
            linear_shape = s * float(optimal["amplitude_v"])
        records.append(pd.DataFrame({
            "tau_us": optimal["tau_us"], "rise_time_s": optimal["rise_time_s"],
            "normalized_time": s, "n_runs": len(shapes),
            "n_runs_meeting_target": int(subset["meets_target"].sum()),
            "mean_voltage_v": shapes.mean(axis=0),
            "std_voltage_v": shapes.std(axis=0, ddof=1) if len(shapes) > 1 else np.full_like(s, np.nan),
            "optimal_voltage_v": optimal_shape,
            "linear_voltage_v": linear_shape,
            "optimal_run_folder": optimal["run_folder"],
            "voltage_source": voltage_source,
            "voltage_column": "rise_voltage_v" if voltage_source == "measured" else "Bernstein V0..Vn",
            "edge_trim_s": trim,
        }))
    metadata = selected[["tau_us", "rise_time_s", "amplitude_v", "rise_points", "run_folder",
                         "sweep_tag", "work_reduction_pct", "meets_target", "candidate", "source_evaluation",
                         "rise_files", "linear_rise_files", *inputs]].copy()
    metadata["voltage_source"] = voltage_source
    metadata["edge_trim_s"] = TRIM_S if voltage_source == "measured" else 0.0
    return pd.concat(records, ignore_index=True), metadata


def plot_waveform_statistics(sweep_folder: Path, output_dir: Path | None = None,
                             voltage_source: str = "commanded") -> list[Path]:
    table, selected = waveform_statistics(sweep_folder, voltage_source)
    taus = np.sort(table["tau_us"].unique())
    colours = plt.cm.Greens(np.linspace(0.4, 0.95, len(taus)))
    columns = min(4, len(taus) + 1)
    rows = math.ceil((len(taus) + 1) / columns)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(rows, columns, squeeze=False, sharex=True, sharey=True,
                                    figsize=(180 * MM, (20 + 43 * rows) * MM))
        for axis, tau, colour in zip(axes.flat, taus, colours):
            subset = table.loc[table["tau_us"] == tau]
            s = subset["normalized_time"].to_numpy()
            mean = subset["mean_voltage_v"].to_numpy()
            sd = subset["std_voltage_v"].to_numpy()
            axis.fill_between(s, mean - sd, mean + sd, color=colour, alpha=0.25, linewidth=0)
            axis.plot(s, subset["linear_voltage_v"], color="0.65", linestyle=":", linewidth=0.8)
            axis.plot(s, mean, color=colour, linewidth=1.3)
            axis.plot(s, subset["optimal_voltage_v"], color=INK, linestyle="--", linewidth=0.8)
            axis.set_title(fr"$\tau={tau:g}$ µs, $n={int(subset['n_runs'].iloc[0])}$", fontsize=7)
        overview = axes.flat[len(taus)]
        for tau, colour in zip(taus, colours):
            subset = table.loc[table["tau_us"] == tau]
            overview.plot(subset["normalized_time"], subset["mean_voltage_v"],
                          color=colour, linewidth=1, label=f"{tau:g}")
        for tau in (taus if voltage_source == "measured" else taus[-1:]):
            reference = table.loc[table["tau_us"] == tau]
            overview.plot(reference["normalized_time"], reference["linear_voltage_v"],
                          color="0.65", linestyle=":", linewidth=0.6)
        overview.set_title("Mean-shape comparison", fontsize=7)
        overview.legend(loc="upper left", ncol=2, title=r"$\tau$ (µs)", title_fontsize=5,
                        fontsize=5, columnspacing=0.6, handlelength=1.2)
        lower = min(0, (table["mean_voltage_v"] - table["std_voltage_v"]).min(),
                    table["optimal_voltage_v"].min(), table["linear_voltage_v"].min())
        upper = max(selected["amplitude_v"].max(),
                    (table["mean_voltage_v"] + table["std_voltage_v"]).max(),
                    table["optimal_voltage_v"].max(), table["linear_voltage_v"].max())
        padding = 0.04 * (upper - lower)
        for index, axis in enumerate(axes.flat):
            if index > len(taus):
                axis.set_visible(False)
                continue
            axis.set(xlim=(0, 1), ylim=(lower - padding, upper + padding))
            axis.set_xticks([0, 0.5, 1])
            if index // columns == rows - 1:
                axis.set_xlabel(r"Normalized time $s=t/\tau$")
            if index % columns == 0:
                axis.set_ylabel(("Measured device voltage" if voltage_source == "measured"
                                 else "Commanded voltage") + " $V$ (V)")
        handles = [Line2D([], [], color="#2f9e6e", linewidth=1.3, label="Across-run mean"),
                   Patch(facecolor="#2f9e6e", alpha=0.25, label="±1 standard deviation"),
                   Line2D([], [], color=INK, linestyle="--", linewidth=0.8, label="Selected optimal"),
                   Line2D([], [], color="0.65", linestyle=":", linewidth=0.8, label="Linear reference")]
        figure.suptitle(("Measured" if voltage_source == "measured" else "Commanded")
                        + " optimal-rise shapes across independent runs", fontsize=8, y=0.99)
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
                      ncol=4, columnspacing=1.5, handlelength=1.8)
        caption = ("Confirmation traces averaged within each run; band: across-run sample SD. "
                   "Measured device voltage; first/last 20 ns excluded." if voltage_source == "measured"
                   else "One confirmed best protocol per run; bands show shape variation across runs, not measurement noise.")
        figure.text(0.5, 0.02, caption, ha="center", fontsize=6)
        figure.subplots_adjust(left=0.075, right=0.99, bottom=0.15, top=0.84,
                               wspace=0.25, hspace=0.4)
        output_dir = output_dir or sweep_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / f"Fig_sweep_waveform_statistics.{suffix}" for suffix in ("pdf", "svg", "png")]
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
    paths.extend([output_dir / "sweep_waveform_statistics.csv",
                  output_dir / "sweep_waveform_selected_protocols.csv"])
    table.to_csv(paths[-2], index=False)
    selected.to_csv(paths[-1], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--voltage-source", choices=("commanded", "measured"), default="commanded",
                        help="measured: scope-derived device voltage, confirmation-averaged and edge-trimmed")
    args = parser.parse_args()
    for path in plot_waveform_statistics(args.sweep_folder, args.output_dir, args.voltage_source):
        print(f"Output: {path}")


if __name__ == "__main__":
    main()
