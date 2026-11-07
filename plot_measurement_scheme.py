"""Publication figure of the measurement scheme, from one BO run's data.

Same journal style as plot_process_figure.py (180 mm wide, 7 pt Arial).
a: setup and closed loop (AWG -> capacitor -> scope -> optimizer -> AWG);
b: one measured PUND period, voltage and current density, linear and optimized rise;
c: the two rise windows, I_P and I_U, whose difference is the switching current
   (with the Work and Delta_P definitions);
d: the Bernstein rise parametrization: 8 coefficients -> stacked contributions -> rise,
   for the linear and an arbitrary (the optimized) protocol;
e: the run measurement by measurement (W / W_linear), aligned with its blocks.
The rise-time sweep results are a separate figure: plot_sweep_figure.py.
Linear = mean of the run's confirmation-reference traces; optimized = mean of
the confirmed optimum's repeats (confirm_summary rank 1).

    python plot_measurement_scheme.py --data-folder <run folder> [--output-dir <folder>]
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from config import PUNDWaveformConfig
from plot_process_figure import BEST, BO, INK, LINEAR, MM, MUTED, RANDOM, STYLE, _inputs, _log_work_axis
from plot_run_overview import load
from waveform import positive_rise_windows, rise_voltage

BOX_FILL = "#f3f3f3"


def _mean_trace(run_dir: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Mean of full scope traces (time relative to the trigger edge), on the first trace's time grid."""
    traces = [pd.read_csv(run_dir / name) for name in rows["trace_file"]]
    time_s = traces[0]["time_s"].to_numpy()
    columns = ("current_a", "voltage_v")
    mean = {column: np.mean([np.interp(time_s, trace["time_s"], trace[column]) for trace in traces], axis=0)
            for column in columns}
    return pd.DataFrame({"time_s": time_s, **mean})


def _windows(trace: pd.DataFrame, timing, points: int = 1000):
    """Time in the rise, V and I_P during the P rise, I_U during the U rise (as pulse_subtraction.py)."""
    s = np.linspace(0.0, timing.rise_time_s, points)
    t, current, voltage = trace["time_s"], trace["current_a"], trace["voltage_v"]
    return (s, np.interp(timing.first_rise_start_s + s, t, voltage),
            np.interp(timing.first_rise_start_s + s, t, current),
            np.interp(timing.second_rise_start_s + s, t, current))


def _box(axis, x, y, width, height, text, fontsize=6.5, text_y=None, text_x=None):
    """Rounded box; text centred, or hanging from text_y (and centred on text_x)."""
    axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=1.2",
                                  facecolor=BOX_FILL, edgecolor=INK, linewidth=0.6))
    axis.text(x + width / 2 if text_x is None else text_x, y + height / 2 if text_y is None else text_y, text,
              ha="center",
              va="center" if text_y is None else "top", fontsize=fontsize, linespacing=1.3)


def _arrow(axis, start, end, color=INK, style="-|>", connection="arc3", linewidth=0.7):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=6, color=color,
                                   linewidth=linewidth, connectionstyle=connection, shrinkA=0, shrinkB=0))


def panel_setup(axis, config):
    """Schematic: AWG out -> top electrode; bottom electrode -> scope CH1 (50 ohm, the current);
    CH2 taps the top electrode; AWG sync -> CH3 (t = 0); the PC closes the loop."""
    axis.set_xlim(0, 88)
    axis.set_ylim(0, 50)
    axis.axis("off")
    bo, device = config["bo"], config["device"]
    side_um = math.sqrt(device["electrode_area_m2"]) * 1e6
    _box(axis, 19, 39, 50, 10, "Computer: Bayesian optimization\n(Xopt, constrained expected improvement)",
         fontsize=6)
    _box(axis, 1, 8, 21, 22, "AWG\n100 MSa s$^{-1}$", text_y=28.5)
    _box(axis, 63, 5, 24, 30, f"Oscilloscope\n{config['instruments']['average_count']} averages", text_y=33.5)
    # Wires: out (y = 24), sync (y = 12); scope ports CH1 (24), CH2 (18), CH3 (12).
    wire = dict(color=INK, linewidth=0.7, solid_capstyle="butt")
    axis.plot([22, 38.5], [24, 24], **wire)
    axis.plot([42.5, 63], [24, 24], **wire)
    axis.plot([31, 31, 63], [24, 18, 18], **wire)
    axis.plot([22, 63], [12, 12], **wire)
    axis.plot(31, 24, "o", color=INK, markersize=2.2)
    for y, text in ((24, "CH1, 50 Ω"), (18, "CH2"), (12, "CH3")):
        axis.text(63.8, y, text, ha="left", va="center", fontsize=5.5)
    axis.text(22.6, 24.5, "Out", ha="left", va="bottom", fontsize=5.5)
    axis.text(22.6, 12.5, "Sync", ha="left", va="bottom", fontsize=5.5)
    # Capacitor: top electrode | BTO | bottom electrode.
    axis.add_patch(Rectangle((38.5, 19), 0.9, 10, facecolor=INK, linewidth=0))
    axis.add_patch(Rectangle((39.4, 19), 2.2, 10, facecolor=BO, alpha=0.35, linewidth=0))
    axis.add_patch(Rectangle((41.6, 19), 0.9, 10, facecolor=INK, linewidth=0))
    axis.text(40.5, 30, f"BTO capacitor\n{side_um:g} × {side_um:g} µm$^2$", ha="center", va="bottom", fontsize=5.5)
    axis.text(52.5, 23.2, "$I = V_{\\mathrm{CH1}}\\,/\\,50\\,\\Omega$", ha="center", va="top", fontsize=5.5)
    axis.text(47, 17.2, "$V = V_{\\mathrm{CH2}} - 50\\,\\Omega \\cdot I$", ha="center", va="top", fontsize=5.5)
    axis.text(42.5, 11.2, "trigger, $t = 0$", ha="center", va="top", fontsize=5.5)
    # Closed loop.
    _arrow(axis, (19, 44), (11.5, 30), connection="arc3,rad=0.3")
    axis.text(8.5, 40.5, f"{bo['rise_points']} rise\ncoefficients $c_k$", ha="right", va="center", fontsize=5.5)
    _arrow(axis, (78, 35), (69, 44), connection="arc3,rad=0.3")
    axis.text(80, 41.5, "$V(t),\\,I(t)$\n→ $W$, $\\Delta P$", ha="left", va="center", fontsize=5.5)
    axis.text(44, 1.5, "PUND: reset → P → U, each P/U rise shaped over $\\tau$", ha="center", va="bottom",
              fontsize=5.5, color=MUTED)


def panel_period(ax_v, ax_i, trace, optimum_trace, timing, waveform, area_m2):
    """One measured period, the linear rise and the optimized (arbitrary) rise."""
    t_us = trace["time_s"].to_numpy() * 1e6
    tau_us = timing.rise_time_s * 1e6
    for axis in (ax_v, ax_i):
        for start in (timing.first_rise_start_s, timing.second_rise_start_s):
            axis.axvspan(start * 1e6, start * 1e6 + tau_us, color=BEST, alpha=0.15, linewidth=0)
    currents = []
    # Optimized first, so the dashed linear trace stays visible where the two coincide (outside the rises).
    for data, colour, style, width, label in ((optimum_trace, BEST, "-", 1.0, "Arbitrary rise"),
                                              (trace, INK, "--", 0.7, "Linear rise")):
        time_us = data["time_s"].to_numpy() * 1e6
        ax_v.plot(time_us, data["voltage_v"], color=colour, linestyle=style, linewidth=width, label=label)
        current = data["current_a"].to_numpy() / area_m2
        ax_i.plot(time_us, current, color=colour, linestyle=style, linewidth=width * 0.75)
        currents.append(current[time_us > timing.first_rise_start_s * 1e6 - 0.5])
    handles, labels = ax_v.get_legend_handles_labels()
    ax_v.legend(handles[::-1], labels[::-1], loc="upper left", borderaxespad=0.2)
    amplitude = waveform.amplitude_v
    reset_mid = (waveform.pre_baseline_s + waveform.other_ramp_s + waveform.pulse_width_s) * 1e6
    ax_v.text(reset_mid, -amplitude + 0.15, "Reset", ha="center", va="bottom")
    for start, name in ((timing.first_rise_start_s, "P"), (timing.second_rise_start_s, "U")):
        ax_v.text(start * 1e6 + tau_us + waveform.pulse_width_s * 0.5e6, amplitude - 0.15, name,
                  ha="center", va="top", fontweight="bold")
        ax_i.text(start * 1e6 + tau_us / 2, 1.0, "$\\tau$", transform=ax_i.get_xaxis_transform(),
                  ha="center", va="bottom", color=BEST)
    gap_us = waveform.gap1_s * 1e6
    ax_v.annotate("", xy=(timing.first_rise_start_s * 1e6, 0.25),
                  xytext=(timing.first_rise_start_s * 1e6 - gap_us, 0.25),
                  arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=0.5, shrinkA=0, shrinkB=0))
    ax_v.text(timing.first_rise_start_s * 1e6 - gap_us / 2, 0.35, f"{gap_us:g} µs", ha="center",
              va="bottom", fontsize=5.5, color=MUTED)
    ax_v.set_ylim(-amplitude - 0.3, amplitude + 0.3)
    ax_v.set_ylabel("$V$ (V)")
    ax_v.tick_params(labelbottom=False)
    # The reset edges (10 ns ramps) give large displacement spikes; keep the switching currents readable.
    top = 1.25 * max(np.max(np.abs(current)) for current in currents)
    ax_i.set_ylim(-top, top)
    ax_i.set(xlabel="Time (µs)", ylabel="$I/A$ (A m$^{-2}$)")
    ax_i.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)  # ×10^6 on the axis
    ax_i.set_xlim(t_us[0], t_us[-1])
    ax_v.set_xlim(t_us[0], t_us[-1])


def panel_windows(axis, windows, area_m2):
    s, _, current_p, current_u = windows
    t_ns = s * 1e9
    axis.fill_between(t_ns, current_u / area_m2, current_p / area_m2, color=BEST, alpha=0.25, linewidth=0,
                      label="$I_P - I_U$, switching")
    axis.plot(t_ns, current_p / area_m2, color=INK, linewidth=0.9, label="$I_P$, switching + non-switching")
    axis.plot(t_ns, current_u / area_m2, color=MUTED, linewidth=0.9, label="$I_U$, non-switching")
    axis.set(xlabel="Time in rise (ns)", ylabel="$I/A$ (A m$^{-2}$)", xlim=(0, t_ns[-1]))
    axis.set_ylim(min(0, axis.get_ylim()[0]), 1.45 * np.max(current_p / area_m2))
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axis.legend(loc="upper left")
    axis.text(0.98, 0.97, "Over the rise window $\\tau$:\n$W = \\int V\\,(I_P - I_U)\\,dt$\n"
              "$\\Delta P = \\int (I_P - I_U)\\,dt\\,/\\,A$", transform=axis.transAxes, ha="right", va="top",
              fontsize=6, linespacing=1.5)


def panel_bernstein(axes, protocols, rise_time_s, maximum_v):
    """Each protocol's rise as stacked Bernstein contributions c_k b_k(s): band k and
    coefficient dot c_k share a colour, and the bands add up to the rise V(s)."""
    degree = protocols[0][1].size - 1
    s = np.linspace(0.0, 1.0, 400)
    t_ns = s * rise_time_s * 1e9
    knots = np.linspace(0.0, rise_time_s * 1e9, degree + 1)
    colours = plt.cm.Blues(np.linspace(0.2, 0.9, degree + 1))
    for axis, (name, controls, colour, style) in zip(axes, protocols):
        terms = [c * math.comb(degree, k) * s ** k * (1 - s) ** (degree - k) for k, c in enumerate(controls)]
        axis.stackplot(t_ns, terms, colors=colours, edgecolor="white", linewidth=0.3)
        axis.plot(t_ns, rise_voltage(controls, s, "bernstein"), color=colour, linestyle=style, linewidth=1.3)
        axis.plot(knots, controls, color=INK, linestyle=":", linewidth=0.5, zorder=4)
        axis.scatter(knots, controls, s=14, c=colours, edgecolors=INK, linewidths=0.5, zorder=5, clip_on=False)
        for k, (x, c) in enumerate(zip(knots, controls)):
            axis.text(x, c + 0.07, f"$c_{k}$", ha="center", va="bottom", fontsize=5.5)
        axis.text(0.03, 0.97, name, transform=axis.transAxes, ha="left", va="top", fontweight="bold")
        axis.set(xlabel="Time in rise (ns)", xlim=(0, t_ns[-1]), ylim=(0, maximum_v + 0.25))
        axis.set_xticks([0, t_ns[-1] / 2, t_ns[-1]])
    axes[0].set_ylabel("$V$ (V)")
    axes[1].tick_params(labelleft=False)
    # The linear rise's upper left is empty: the formula goes there.
    axes[0].text(0.03, 0.85, f"$V(s) = \\Sigma_{{k=0}}^{{{degree}}}\\, c_k\\,b_{{k,{degree}}}(s)$\n"
                 f"$b_{{k,{degree}}}(s) = \\binom{{{degree}}}{{k}}\\,s^k (1-s)^{{{degree}-k}}$\n$s = t/\\tau$",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=5.5, linespacing=1.6)


def panel_sequence(ax_run, ax_bar, history, p_floor):
    """W / W_linear of every recorded measurement (BO steps filled if delta_P reached the
    target within noise, open if not), above one bar per measurement coloured by its role,
    with the run's blocks labelled on top. The warm-up (not recorded in the history) is left out."""
    phase = history["phase"].to_numpy()
    position = np.arange(len(history))
    first_bo = int(np.flatnonzero(phase == "BO optimization")[0])
    first_confirm = int(np.flatnonzero(np.isin(phase, ("confirmation", "confirmation reference")))[0])
    end = len(history)

    work = history["Work_NORM"].to_numpy()
    ok = (history["delta_P_NORM"] >= p_floor).to_numpy()
    linear, random = np.isin(phase, LINEAR), phase == "random initial"
    bo, confirmation = phase == "BO optimization", phase == "confirmation"
    roles = [(linear, INK, "s", "Linear"), (random, RANDOM, "o", "Random"), (bo, BO, "o", "BO"),
             (confirmation, BEST, "D", "Confirmation")]
    groups = [(linear, dict(marker="s", s=10, color=INK), "Linear"),
              (random, dict(marker="o", s=10, color=RANDOM), "Random"),
              # Open BO steps are drawn faint and underneath.
              (bo & ~ok, dict(marker="o", s=9, facecolors="white", edgecolors=BO, linewidths=0.6, alpha=0.55,
                              zorder=2), "BO, $\\Delta P <$ target"),
              (bo & ok, dict(marker="o", s=10, color=BO), "BO, $\\Delta P \\geq$ target"),
              (confirmation, dict(marker="D", s=10, color=BEST), "Confirmation")]
    for mask, kwargs, label in groups:
        kwargs = {"linewidths": 0, "zorder": 3, **kwargs}
        ax_run.scatter(position[mask], work[mask], label=label, **kwargs)
    ax_run.axhline(1.0, color=INK, linewidth=0.5, linestyle=":")
    _log_work_axis(ax_run, (0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5))
    ax_run.set_ylabel("$W\\,/\\,W_{\\mathrm{linear}}$")
    ax_run.tick_params(labelbottom=False)
    handles, labels = ax_run.get_legend_handles_labels()
    order = [0, 1, 3, 2, 4]  # BO >= target before BO < target
    ax_run.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", ncol=3,
                  columnspacing=1.2, handletextpad=0.2, borderaxespad=0.2)

    for mask, colour, _, _ in roles:
        ax_bar.bar(position[mask], 1, width=1.0, color=colour, linewidth=0)
    for boundary in (first_bo, first_confirm):
        for axis in (ax_run, ax_bar):
            axis.axvline(boundary - 0.5, color=MUTED, linewidth=0.5, zorder=0)
    n_random = int((phase == "random initial").sum())
    n_linear = int((phase == "linear rise").sum())
    n_bo = int((phase == "BO optimization").sum())
    n_ref = int((phase == "linear reference").sum())
    n_confirm = int((phase == "confirmation").sum())
    n_cref = int((phase == "confirmation reference").sum())
    top = ax_run.get_xaxis_transform()
    for start, stop, text in ((0, first_bo, f"Initial\n{n_linear} linear + {n_random} random"),
                              (first_bo, first_confirm,
                               f"Bayesian optimization\n{n_bo} steps + {n_ref} linear references (drift)"),
                              (first_confirm, end, f"Confirmation\n{n_confirm} + {n_cref} linear")):
        # The narrow end blocks: text aligned to the panel edge.
        x, ha = (start, "left") if start == 0 else (stop, "right") if stop == end else ((start + stop) / 2, "center")
        ax_run.text(x, 1.07, text, transform=top, ha=ha, va="bottom", fontsize=6, linespacing=1.2)
        ax_run.plot([start + 0.5, stop - 0.5], [1.03, 1.03], transform=top, color=INK, linewidth=0.5,
                    clip_on=False)
    ax_run.set_xlim(-0.5, end - 0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.spines["left"].set_visible(False)
    ax_bar.set_xlabel("Measurement number")


def plot_measurement_scheme(run_dir: Path, output_dir: Path | None = None,
                            config_path: Path | None = None) -> list[Path]:
    _, history, config, summary, _ = load(run_dir, config_path)
    summary = summary.sort_values("rank").reset_index(drop=True)
    waveform = PUNDWaveformConfig.from_saved(config["waveform"])
    timing = positive_rise_windows(waveform)
    area_m2 = config["device"]["electrode_area_m2"]
    # Same rule as bo.py: delta_P may fall short of the target by confirm_noise_sd s.d. of the linear rises.
    noise = history.loc[history["phase"].isin(LINEAR), "delta_P_NORM"].std()
    p_floor = 1.0 - config["bo"].get("confirm_noise_sd", 0.0) * noise
    best_candidate = summary.iloc[0]["candidate"]
    optimum_rows = history[(history["phase"] == "confirmation") & (history["candidate"] == best_candidate)]
    best_controls = history[history["candidate"] == best_candidate].iloc[0][_inputs(history)].to_numpy(float)
    linear_trace = _mean_trace(run_dir, history[history["phase"] == "confirmation reference"])
    optimum_trace = _mean_trace(run_dir, optimum_rows)
    linear_windows = _windows(linear_trace, timing)
    optimum_windows = _windows(optimum_trace, timing)
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(180 * MM, 170 * MM))
        grid = figure.add_gridspec(3, 6, height_ratios=[1.15, 1.0, 1.05], hspace=0.42, wspace=1.1,
                                   left=0.06, right=0.985, top=0.96, bottom=0.05)
        bernstein = grid[1, 3:].subgridspec(1, 2, wspace=0.16)
        top_right = grid[0, 3:].subgridspec(2, 1, hspace=0.22, height_ratios=[1, 1])
        # Row 0 of the bottom block is left empty: room for e's block labels.
        bottom = grid[2, :].subgridspec(3, 1, hspace=0.06, height_ratios=[0.2, 1, 0.16])
        axes = {"a": figure.add_subplot(grid[0, :3]), "b": figure.add_subplot(top_right[0]),
                "c": figure.add_subplot(grid[1, :3]), "d": figure.add_subplot(bernstein[0]),
                "e": figure.add_subplot(bottom[1])}
        ax_current = figure.add_subplot(top_right[1], sharex=axes["b"])
        ax_bar = figure.add_subplot(bottom[2], sharex=axes["e"])
        panel_setup(axes["a"], config)
        panel_period(axes["b"], ax_current, linear_trace, optimum_trace, timing, waveform, area_m2)
        panel_windows(axes["c"], linear_windows, area_m2)
        linear_controls = history[history["phase"] == "linear rise"].iloc[0][_inputs(history)].to_numpy(float)
        ax_arbitrary = figure.add_subplot(bernstein[1], sharey=axes["d"])
        panel_bernstein((axes["d"], ax_arbitrary), (("Linear rise", linear_controls, INK, "--"),
                                                    ("Arbitrary rise", best_controls, BEST, "-")),
                        waveform.rise_time_s, waveform.maximum_voltage_v)
        panel_sequence(axes["e"], ax_bar, history, p_floor)
        for letter, axis in axes.items():
            box = axis.get_position()
            x = 0.008 if letter in "ace" else box.x0 - 0.05
            # e: above its block labels.
            figure.text(x, box.y1 + (0.045 if letter == "e" else 0.01), letter, fontsize=8, fontweight="bold",
                        va="bottom", ha="left")
        output_dir = output_dir or run_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / f"Fig_measurement_scheme.{suffix}" for suffix in ("pdf", "svg", "png")]
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
    parser.add_argument("--data-folder", type=Path, required=True, help="BO run folder")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: the run folder")
    parser.add_argument("--config", type=Path, default=None, help="run_config json, if not in the run folder")
    args = parser.parse_args()
    for path in plot_measurement_scheme(args.data_folder, args.output_dir, args.config):
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
