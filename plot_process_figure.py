"""Publication figure of one BO run: how the optimization proceeds.

Journal style (Nature family): 180 mm wide, 7 pt Arial, panels a-f, no
titles (the caption explains), PDF with editable text + 600 dpi PNG.
a: the PUND waveform, linear vs optimized rise (the only part that changes);
b: rise shapes -- random seeds, each new best, the confirmed optimum;
c: the run measurement by measurement (Work vs the linear rise), phases shaded;
d: Work vs delta_P, the target constraint; e: switching current density;
f: V vs P, whose area is the Work per area. Uses the run's confirm_summary
(rank 1 = the optimum) and its confirmation rise traces.

    python plot_process_figure.py --data-folder <run folder> [--output-dir <folder>]
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from config import PUNDWaveformConfig
from plot_run_overview import load
from plot_sweep_results import _mean_rise
from waveform import build_pund_waveform, positive_rise_windows, rise_voltage

MM = 1 / 25.4
INK, MUTED = "#222222", "#9a9a9a"
RANDOM, BO, BEST = "#eb6834", "#2a78d6", "#2f9e6e"
SEARCH = ("random initial", "BO optimization")
LINEAR = ("linear rise", "linear reference", "confirmation reference")
STYLE = {"font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
         "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
         "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
         "lines.linewidth": 1.0, "legend.frameon": False, "legend.handlelength": 1.6,
         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"}


def _inputs(history):
    return [column for column in history.columns if column.startswith("V") and column[1:].isdigit()]


def _log_work_axis(axis, ticks=(0.5, 0.7, 1, 1.5, 2, 3)):
    axis.set_yscale("log")
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{tick:g}" for tick in ticks])
    axis.yaxis.set_minor_formatter(NullFormatter())


def _new_bests(history):
    """Searched rows (random + BO) with delta_P >= target - noise that lowered the best Work so far."""
    ok = history[history["phase"].isin(SEARCH) & history["ok"]]
    return ok[ok["Work_NORM"] < ok["Work_NORM"].cummin().shift(fill_value=np.inf)]


def panel_waveform(axis, config, history, best):
    waveform = PUNDWaveformConfig.from_saved(config["waveform"])
    inputs = _inputs(history)
    timing = positive_rise_windows(waveform)
    rise_us = waveform.rise_time_s * 1e6
    for start in (timing.first_rise_start_s, timing.second_rise_start_s):
        axis.axvspan(start * 1e6, start * 1e6 + rise_us, color=BEST, alpha=0.15, linewidth=0)
    for controls, color, style, label in ((best[inputs], BEST, "-", "Optimized rise"),
                                          (history.iloc[0][inputs], INK, "--", "Linear rise")):
        arb, _ = build_pund_waveform(np.asarray(controls, dtype=float), waveform)
        t_us = np.arange(arb.voltage_v.size) / arb.sample_rate_hz * 1e6
        axis.plot(t_us, arb.voltage_v, color=color, linestyle=style, linewidth=0.9, label=label)
    amplitude = waveform.amplitude_v
    reset_mid = (waveform.pre_baseline_s + waveform.other_ramp_s + waveform.reset_width_s / 2) * 1e6
    axis.text(reset_mid, -amplitude - 0.12, "Reset", ha="center", va="top")
    for start, name in ((timing.first_rise_start_s, "P"), (timing.second_rise_start_s, "U")):
        axis.text(start * 1e6 + rise_us + waveform.pulse_width_s * 0.5e6, amplitude + 0.08, name,
                  ha="center", va="bottom")
    # Annotations sit above the reset pulse, the only empty region.
    axis.annotate(f"Shaped rise (optimized), τ = {rise_us:g} µs",
                  xy=(timing.first_rise_start_s * 1e6 + 0.1 * rise_us, 0.45),
                  xytext=(reset_mid, 1.3), ha="center", va="center",
                  arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.5, shrinkB=1))
    axis.text(reset_mid - 0.4, 0.42, "Over each τ window:\n$W = \\int V\\,(I_P - I_U)\\,dt$\n"
              "$\\Delta P = \\int (I_P - I_U)\\,dt\\,/\\,A$", ha="center", va="center", fontsize=6)
    axis.set_ylim(-amplitude - 0.45, amplitude + 0.45)
    axis.set_xlim(0, t_us[-1])
    axis.set(xlabel="Time (µs)", ylabel="Voltage (V)")
    axis.legend(loc="lower right", ncol=2)


def panel_shapes(axis, config, history, best):
    waveform, inputs = config["waveform"], _inputs(history)
    s = np.linspace(0.0, 1.0, 300)
    t_ns = s * waveform["rise_time_s"] * 1e9

    def shape(values):
        return rise_voltage(np.asarray(values, dtype=float), s, waveform["rise_basis"])

    for _, row in history[history["phase"] == "random initial"].iterrows():
        axis.plot(t_ns, shape(row[inputs]), color=RANDOM, linewidth=0.4, alpha=0.35)
    improved = _new_bests(history)
    for (_, row), color in zip(improved.iterrows(), plt.cm.Blues(np.linspace(0.35, 0.9, len(improved)))):
        axis.plot(t_ns, shape(row[inputs]), color=color, linewidth=0.7)
    axis.plot(t_ns, shape(history.iloc[0][inputs]), color=INK, linestyle="--", linewidth=0.9)
    axis.plot(t_ns, shape(best[inputs]), color=BEST, linewidth=1.6)
    axis.set(xlabel="Time in rise (ns)", ylabel="Voltage (V)",
             ylim=(waveform["minimum_voltage_v"] - 0.05, waveform["maximum_voltage_v"] + 0.05))
    axis.legend(handles=[Line2D([], [], color=RANDOM, linewidth=0.6, label="Random seeds"),
                         Line2D([], [], color=plt.cm.Blues(0.65), linewidth=0.8, label="Successive bests"),
                         Line2D([], [], color=INK, linestyle="--", linewidth=0.9, label="Linear"),
                         Line2D([], [], color=BEST, linewidth=1.6, label="Optimized")],
                loc="upper left", ncol=2, columnspacing=1.0)


def panel_trajectory(axis, history, summary):
    best_candidate = int(summary.iloc[0]["candidate"])
    first_bo = history.index[history["phase"] == "BO optimization"].min()
    first_confirm = history.index[history["phase"].isin(("confirmation", "confirmation reference"))].min()
    last = history.index.max()
    for start, stop in ((-0.5, first_bo - 0.5), (first_confirm - 0.5, last + 0.5)):
        axis.axvspan(start, stop, color="#f0f0f0", linewidth=0, zorder=0)
    for start, stop, text in ((-0.5, first_bo - 0.5, "Initial"),
                              (first_bo - 0.5, first_confirm - 0.5, "Bayesian optimization"),
                              (first_confirm - 0.5, last + 0.5, "Confirmation")):
        axis.text((start + stop) / 2, 1.0, text, transform=axis.get_xaxis_transform(), ha="center", va="bottom")
    phase, feasible = history["phase"], history["ok"]
    confirmation = phase == "confirmation"
    is_best = history["candidate"] == best_candidate
    groups = [
        (phase.isin(LINEAR), dict(marker="s", s=5, color=INK), "Linear reference"),
        (phase == "random initial", dict(marker="o", s=6, color=RANDOM), "Random seed"),
        ((phase == "BO optimization") & ~feasible,
         dict(marker="o", s=6, facecolors="none", edgecolors=BO, linewidths=0.5), "BO, ΔP below"),
        ((phase == "BO optimization") & feasible, dict(marker="o", s=7, color=BO), "BO, ΔP ≥ target − noise"),
        (confirmation & ~is_best, dict(marker="D", s=5, color=MUTED), "Other candidates"),
        (confirmation & is_best, dict(marker="D", s=6, color=BEST), "Optimum (repeats)"),
    ]
    for mask, kwargs, label in groups:
        rows = history[mask]
        axis.scatter(rows.index, rows["Work_NORM"], linewidths=kwargs.pop("linewidths", 0), label=label,
                     zorder=3, **kwargs)
    searched = history[phase.isin(SEARCH)]
    running = searched["Work_NORM"].where(searched["ok"]).cummin()
    running = running.reindex(history.index[history.index < first_confirm]).ffill()
    axis.step(running.index, running, where="post", color=BEST, linewidth=1.0, label="Best so far (ΔP ≥ target − noise)",
              zorder=4)
    axis.axhline(1.0, color=INK, linewidth=0.5, linestyle=":")
    row = summary.iloc[0]
    axis.text(last + 1, row["Work_NORM_mean"] * 0.9,
              f"−{row['work_reduction_pct']:.1f} ± {100 * row['Work_NORM_std']:.1f}%\n(n = {int(row['n'])})",
              ha="right", va="top", color=BEST, fontsize=6)
    _log_work_axis(axis, (0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3))
    axis.set_xlim(-2, last + 2)
    axis.set(xlabel="Measurement number", ylabel="$W\\,/\\,W_{\\mathrm{linear}}$")
    axis.legend(loc="upper right", ncol=4, columnspacing=1.0, bbox_to_anchor=(1.0, 0.98))


def panel_tradeoff(axis, history, summary, p_floor, floor=0.95):
    best_candidate = int(summary.iloc[0]["candidate"])
    phase, feasible = history["phase"], history["ok"]
    groups = [
        (phase == "random initial", dict(marker="o", s=6, color=RANDOM)),
        ((phase == "BO optimization") & ~feasible,
         dict(marker="o", s=6, facecolors="none", edgecolors=BO, linewidths=0.5)),
        ((phase == "BO optimization") & feasible, dict(marker="o", s=7, color=BO)),
    ]
    for mask, kwargs in groups:
        rows = history[mask]
        x, y = rows["delta_P_NORM"].to_numpy(), rows["Work_NORM"].to_numpy()
        low = x < floor
        width = kwargs.pop("linewidths", 0)
        axis.scatter(x[~low], y[~low], linewidths=width, zorder=3, **kwargs)
        axis.scatter(np.full(low.sum(), floor), y[low], linewidths=width, zorder=3, **{**kwargs, "marker": "<"})
    axis.axvspan(p_floor, 1.02, color=BEST, alpha=0.12, linewidth=0, zorder=0)
    axis.axvline(1.0, color=BEST, linewidth=0.8)
    if p_floor < 1.0:
        axis.axvline(p_floor, color=BEST, linewidth=0.6, linestyle=":")
    row = summary.iloc[0]
    axis.errorbar([row["delta_P_NORM_mean"]], [row["Work_NORM_mean"]], xerr=[row["delta_P_NORM_std"]],
                  yerr=[row["Work_NORM_std"]], marker="*", markersize=9, color=BEST, markeredgecolor="white",
                  markeredgewidth=0.4, elinewidth=0.6, zorder=6)
    axis.scatter([1.0], [1.0], marker="s", s=16, color=INK, zorder=6)
    axis.text(1.0005, 0.97, "ΔP ≥ target\n− noise" if p_floor < 1.0 else "ΔP ≥ target",
              transform=axis.get_xaxis_transform(), color=BEST, va="top", fontsize=6)
    axis.text(0.9985, 1.0, "Linear", ha="right", va="center", fontsize=6)
    axis.annotate("Optimum", xy=(row["delta_P_NORM_mean"], row["Work_NORM_mean"] * 0.95),
                  xytext=(row["delta_P_NORM_mean"] + 0.001, row["Work_NORM_mean"] * 0.55), ha="center",
                  fontsize=6, color=BEST, arrowprops=dict(arrowstyle="-", color=BEST, linewidth=0.5))
    _log_work_axis(axis, (0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3))
    axis.set_xlim(floor - 0.002, max(1.008, history["delta_P_NORM"].max() + 0.002))
    axis.set(xlabel="$\\Delta P\\,/\\,\\Delta P_{\\mathrm{linear}}$", ylabel="$W\\,/\\,W_{\\mathrm{linear}}$")


def panels_measured(ax_current, ax_vp, history, run_dir, area_m2, best):
    def mean_rise(rows):
        return _mean_rise(";".join(str(run_dir / name) for name in rows["rise_file"]), area_m2)

    linear = mean_rise(history[history["phase"] == "confirmation reference"])
    optimum = mean_rise(history[(history["phase"] == "confirmation") & (history["candidate"] == best["candidate"])])
    area_cm2 = area_m2 * 1e4
    works = []
    for (t, voltage, current, polarization), color, style, label in (
            (linear, INK, "--", "Linear"), (optimum, BEST, "-", "Optimized")):
        p_uc_cm2 = polarization * 100.0  # C/m^2 -> uC/cm^2
        ax_current.plot(t * 1e9, current / area_cm2, color=color, linestyle=style, label=label)
        work_uj_cm2 = np.sum(0.5 * (voltage[1:] + voltage[:-1]) * np.diff(p_uc_cm2))  # V * uC/cm^2
        works.append(work_uj_cm2)
        ax_vp.plot(p_uc_cm2, voltage, color=color, linestyle=style,
                   label=f"{label}: {work_uj_cm2:.1f} µJ cm$^{{-2}}$")
        ax_vp.fill_between(p_uc_cm2, voltage, color=color, alpha=0.12, linewidth=0)
    ax_current.set(xlabel="Time in rise (ns)", ylabel="$(I_P - I_U)\\,/\\,A$ (A cm$^{-2}$)")
    ax_current.legend(loc="upper left")
    ax_vp.set(xlabel="Switched polarization (µC cm$^{-2}$)", ylabel="Voltage (V)")
    ax_vp.set_ylim(bottom=0)
    ax_vp.legend(loc="upper left", title="$W/A = \\int V\\,dP$", title_fontsize=6, alignment="left")


def plot_process_figure(run_dir: Path, output_dir: Path | None = None,
                        config_path: Path | None = None) -> list[Path]:
    history_path, history, config, summary, _ = load(run_dir, config_path)
    summary = summary.sort_values("rank").reset_index(drop=True)
    best = history[history["candidate"] == summary.iloc[0]["candidate"]].iloc[0]
    # Same rule as bo.py: delta_P may fall short of the target by confirm_noise_sd
    # s.d. of the repeated linear rises (0 for runs from before that setting).
    noise = history.loc[history["phase"].isin(LINEAR), "delta_P_NORM"].std()
    p_floor = 1.0 - config["bo"].get("confirm_noise_sd", 0.0) * noise
    history = history.assign(ok=history["delta_P_NORM"] >= p_floor)
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(180 * MM, 150 * MM))
        grid = figure.add_gridspec(3, 3, height_ratios=[1, 1.15, 1.05], hspace=0.55, wspace=0.38,
                                   left=0.065, right=0.985, top=0.965, bottom=0.07)
        axes = {"a": figure.add_subplot(grid[0, :2]), "b": figure.add_subplot(grid[0, 2]),
                "c": figure.add_subplot(grid[1, :]), "d": figure.add_subplot(grid[2, 0]),
                "e": figure.add_subplot(grid[2, 1]), "f": figure.add_subplot(grid[2, 2])}
        panel_waveform(axes["a"], config, history, best)
        panel_shapes(axes["b"], config, history, best)
        panel_trajectory(axes["c"], history, summary)
        panel_tradeoff(axes["d"], history, summary, p_floor)
        panels_measured(axes["e"], axes["f"], history, run_dir, config["device"]["electrode_area_m2"], best)
        for letter, axis in axes.items():
            box = axis.get_position()
            figure.text(box.x0 - 0.052 * (180 / 180), box.y1 + 0.012, letter, fontsize=8, fontweight="bold",
                        va="bottom", ha="left")
        output_dir = output_dir or run_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "Fig_BO_process"  # short: run folders already sit near Windows' 260-character path limit
        paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
        for path in paths:
            for attempt in range(5):  # Windows: a viewer may briefly hold the file
                try:
                    figure.savefig(path, dpi=600)
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(1.0)
        plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-folder", type=Path, required=True, help="one run folder (BO_pund_rise_...)")
    parser.add_argument("--output-dir", type=Path, help="where to write the PDF/PNG (default: the run folder)")
    parser.add_argument("--config", type=Path, help="run_config JSON to use instead of the run's own")
    args = parser.parse_args()
    for path in plot_process_figure(args.data_folder, args.output_dir, args.config):
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
