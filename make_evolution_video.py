"""Create an MP4/GIF of a PUND rise-shape BO run from its saved CSV files.
Adapted from General_trigger_sync_BO_useful/make_evolution_video.py: no
analytical/simulated-circuit overlays exist for ferroelectric switching, so
every panel is measured-only; Q_tau/VI_integral become delta_P/Work, and the
"Charge on C" panel becomes a delta_P(t) trajectory (cumulative integral of
the subtracted rise-window current)."""

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from config import ExperimentConfig, PUNDWaveformConfig
from waveform import positive_rise_windows, rise_voltage

# User-editable folder containing bo_history_*.csv and rise_*.csv files.
DATA_FOLDER: Path | None = Path(r"D:\Ferroelectric_BTO_optimization\20260821\BO_pund_rise_BTO_F9_1533")

# Marker shape encodes which phase an evaluation belongs to; color encodes
# feasibility (independent of phase) -- see the work-vs-reference panel.
_PHASE_MARKERS = {"linear rise": "s", "linear reference": "s", "random initial": "x", "BO optimization": "o"}
_FEASIBLE_COLORS = {True: "tab:green", False: "tab:red"}
_WORK_PANEL_LEGEND_HANDLES = [
    Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="k", label="linear rise"),
    Line2D([0], [0], marker="x", linestyle="None", color="k", label="random initial"),
    Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="k", label="BO optimization"),
    Line2D([0], [0], marker="o", linestyle="None", color=_FEASIBLE_COLORS[True], label="feasible"),
    Line2D([0], [0], marker="o", linestyle="None", color=_FEASIBLE_COLORS[False], label="infeasible"),
]


def _cumulative_trapz(y, x):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(increments)))


def latest_history(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"Data folder does not exist: {directory}")
    files = list(directory.glob("bo_history_*.csv"))
    if not files:
        raise FileNotFoundError(f"No BO history CSV found in {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def latest_run_folder(mother_path: Path) -> Path:
    """The folder holding the most recently written bo_history_*.csv anywhere
    under mother_path -- lets --data-folder/--history be omitted entirely to
    regenerate a video from the last real run, without needing to know its
    timestamped subfolder name (cfg.output_directory can't be used for this:
    it's derived from datetime.now(), so it names a fresh folder every call,
    never a past run's)."""
    if not mother_path.is_dir():
        raise NotADirectoryError(f"data_mother_path does not exist: {mother_path}")
    candidates = list(mother_path.rglob("bo_history_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No bo_history_*.csv found anywhere under {mother_path}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def load_run_config(history_path: Path) -> dict | None:
    """Load the run_config_<timestamp>.json written alongside bo_history_<timestamp>.csv,
    if one exists (older runs predate that log and simply fall back to config.py)."""
    if not history_path.stem.startswith("bo_history_"):
        return None
    run_timestamp = history_path.stem.removeprefix("bo_history_")
    config_path = history_path.parent / f"run_config_{run_timestamp}.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


def load_run(history_path: Path):
    history = pd.read_csv(history_path).sort_values("evaluation").reset_index(drop=True)
    if history.empty:
        raise ValueError("History is empty")
    voltage_columns = sorted(
        (name for name in history if name.startswith("V") and name[1:].isdigit()),
        key=lambda name: int(name[1:]),
    )
    if not voltage_columns:
        raise ValueError("History contains no voltage control-point columns")
    rises = []
    full_traces = []
    for _, row in history.iterrows():
        rise_path = history_path.parent / str(row["rise_file"])
        if not rise_path.exists():
            raise FileNotFoundError(f"Missing rise trace referenced by history: {rise_path}")
        rises.append(pd.read_csv(rise_path))
        trace_path = history_path.parent / str(row["trace_file"])
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing full waveform/current trace referenced by history: {trace_path}")
        full_traces.append(pd.read_csv(trace_path))
    return history, voltage_columns, rises, full_traces


def _padded_bounds(values, pad_frac=0.05):
    values = np.asarray(values, dtype=float)
    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high):
        span = max(abs(low), 1.0)
        return low - span * pad_frac, high + span * pad_frac
    pad = pad_frac * (high - low)
    return low - pad, high + pad


def make_video(history_path: Path, output_path: Path, area_m2: float,
               waveform_cfg: PUNDWaveformConfig, fps: int = 6) -> Path:
    history, voltage_columns, rises, full_traces = load_run(history_path)
    feasible = history["feasible"].astype(str).str.lower().eq("true")
    history = history.assign(feasible=feasible)
    timing = positive_rise_windows(waveform_cfg)
    # Runs whose run_config predates rise_basis load with its "points" default.
    basis = waveform_cfg.rise_basis
    control_label = "Bernstein coefficients" if basis == "bernstein" else "BO control knots"

    reference_rise = rises[0]
    reference_time = reference_rise["rise_time_s"].to_numpy(float)
    reference_v = reference_rise["rise_voltage_v"].to_numpy(float)
    reference_idiff = reference_rise["rise_current_diff_a"].to_numpy(float)
    reference_delta_p_t = _cumulative_trapz(reference_idiff, reference_time) / area_m2

    if (history["phase"] == "linear reference").any():
        # Drift-corrected run (BOConfig.reference_interval): each row has its
        # own reference, interpolated between the linear rises around it.
        excess_work = history["Work"] / history["WORK_SCALE"] - 1.0
        p_target_rows = history["P_target"].to_numpy(float)
        p_lower_bound_rows = history["P_lower_bound"].to_numpy(float)
    else:
        # Mean of the linear-rise rows (BOConfig.linear_repeats of them; one in
        # older runs); the last row holds the final target, not a running mean.
        reference_work = float(history.loc[history["phase"] == "linear rise", "Work"].mean())
        excess_work = (history["Work"] - reference_work) / abs(reference_work)
        p_target_rows = np.full(len(history), float(history.iloc[-1]["P_target"]))
        p_lower_bound_rows = np.full(len(history), float(history.iloc[-1]["P_lower_bound"]))
    p_target, p_lower_bound = p_target_rows[-1], p_lower_bound_rows[-1]

    delta_p_all = history["delta_P"].to_numpy(float)
    p_pad = max(0.15 * abs(p_target), 5.0 * abs(p_lower_bound - p_target), 1e-15)
    p_axis_low = min(delta_p_all.min(), p_lower_bound_rows.min()) - p_pad
    p_axis_high = max(delta_p_all.max(), p_target_rows.max()) + p_pad

    excess_bounds = [excess_work.min(), excess_work.max()]
    excess_axis_low, excess_axis_high = min(excess_bounds), max(excess_bounds)
    excess_pad = 0.05 * max(excess_axis_high - excess_axis_low, 1e-12)

    # Precompute every frame's delta_P(t) trajectory once (reused below both
    # for plotting and for fixing this panel's y-axis across the whole video).
    delta_p_trajectories = [
        _cumulative_trapz(rise["rise_current_diff_a"].to_numpy(float),
                          rise["rise_time_s"].to_numpy(float)) / area_m2
        for rise in rises
    ]

    # Fix every panel's axis limits across the whole animation (computed once,
    # over every frame's data) so the view doesn't rescale frame to frame --
    # per-frame autoscaling made the video hard to read while playing.
    rise_v_low, rise_v_high = _padded_bounds(
        np.concatenate([reference_v] + [r["rise_voltage_v"].to_numpy(float) for r in rises]))
    rise_i_low, rise_i_high = _padded_bounds(
        np.concatenate([reference_idiff] + [r["rise_current_diff_a"].to_numpy(float) for r in rises]))
    delta_p_t_low, delta_p_t_high = _padded_bounds(
        np.concatenate([reference_delta_p_t, p_target_rows, p_lower_bound_rows] + delta_p_trajectories))
    full_v_low, full_v_high = _padded_bounds(
        np.concatenate([t["voltage_v"].to_numpy(float) for t in full_traces]))
    full_i_low, full_i_high = _padded_bounds(
        np.concatenate([t["current_a"].to_numpy(float) for t in full_traces]))
    full_all_time = np.concatenate([t["time_s"].to_numpy(float) for t in full_traces])
    full_t_low, full_t_high = float(full_all_time.min()), float(full_all_time.max())

    # If this runs inside run_experiment.py, control_panel.py has already put
    # pyplot into interactive mode (plt.ion()) for its live dashboard -- any
    # figure created while interactive pops up its own window. This is purely
    # rendered-to-file, so suppress that and restore whatever mode was active
    # before we're done.
    was_interactive = plt.isinteractive()
    plt.ioff()
    figure, axes = plt.subplots(2, 3, figsize=(17, 8))
    ax_v, ax_current, ax_p = axes[0]
    ax_delta_p_history, ax_work, ax_full = axes[1]
    ax_full_i = ax_full.twinx()

    def update(frame):
        row = history.iloc[frame]
        rise = rises[frame]
        time_s = rise["rise_time_s"].to_numpy(float)
        voltage_v = rise["rise_voltage_v"].to_numpy(float)
        current_diff_a = rise["rise_current_diff_a"].to_numpy(float)
        controls = row[voltage_columns].to_numpy(float)
        control_t = np.linspace(time_s[0], time_s[-1], controls.size)
        is_feasible = bool(row["feasible"])

        full_trace = full_traces[frame]
        full_time_s = full_trace["time_s"].to_numpy(float)
        full_voltage_v = full_trace["voltage_v"].to_numpy(float)
        full_current_a = full_trace["current_a"].to_numpy(float)

        for axis in (ax_v, ax_current, ax_p, ax_delta_p_history, ax_work, ax_full, ax_full_i):
            axis.clear()

        ax_v.plot(reference_time, reference_v, "k--", linewidth=1.2, label="linear rise V(t)")
        ax_v.plot(time_s, voltage_v, color="tab:blue", linewidth=2, label="measured V(t)")
        commanded_v = rise_voltage(controls, (time_s - time_s[0]) / (time_s[-1] - time_s[0]), basis)
        ax_v.plot(time_s, commanded_v, color="tab:cyan", linestyle=":", linewidth=1.5,
                  label=f"commanded V(t) ({basis})")
        ax_v.scatter(control_t, controls, color="tab:blue", s=24, zorder=4, label=control_label)
        ax_v.set(title="Rise voltage", xlabel="Rise time (s)", ylabel="Voltage (V)")
        ax_v.set_xlim(reference_time[0], reference_time[-1])
        ax_v.set_ylim(rise_v_low, rise_v_high)
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc="upper left", fontsize=8)

        ax_current.plot(reference_time, reference_idiff, "k--", linewidth=1.2, label="linear rise I_diff(t)")
        ax_current.plot(time_s, current_diff_a, color="tab:green", linewidth=1.8, label="measured I_diff(t)")
        ax_current.set(title="Subtracted current (I1-I2)", xlabel="Rise time (s)", ylabel="Current (A)")
        ax_current.set_xlim(reference_time[0], reference_time[-1])
        ax_current.set_ylim(rise_i_low, rise_i_high)
        ax_current.grid(True, alpha=0.3)
        ax_current.legend(loc="upper right", fontsize=8)

        measured_delta_p_t = delta_p_trajectories[frame]
        ax_p.plot(reference_time, reference_delta_p_t, "k--", linewidth=1.2, label="linear rise delta_P(t)")
        ax_p.plot(time_s, measured_delta_p_t, color="tab:blue", linewidth=2, label="measured delta_P(t)")
        ax_p.axhline(p_target_rows[frame], color="k", linestyle=":", linewidth=1, label="target delta_P")
        ax_p.axhline(p_lower_bound_rows[frame], color="tab:red", linestyle="--", linewidth=1.2,
                     label="BO feasibility boundary")
        ax_p.set(title="Change of polarization", xlabel="Rise time (s)", ylabel="delta_P (C/m^2)")
        ax_p.set_xlim(reference_time[0], reference_time[-1])
        ax_p.set_ylim(delta_p_t_low, delta_p_t_high)
        ax_p.grid(True, alpha=0.3)
        ax_p.legend(loc="upper left", fontsize=8)

        past = history.iloc[: frame + 1]
        for phase_name, marker in _PHASE_MARKERS.items():
            subset = past[past["phase"] == phase_name]
            if subset.empty:
                continue
            marker_size = 32 if marker == "x" else 22
            colors = subset["feasible"].map(_FEASIBLE_COLORS)
            ax_work.scatter(subset["evaluation"], excess_work.loc[subset.index],
                            s=marker_size, alpha=0.85, c=colors, marker=marker)
            ax_delta_p_history.scatter(subset["evaluation"], subset["delta_P"],
                                       s=marker_size, alpha=0.85, c=colors, marker=marker)

        ax_work.axhline(0.0, color="k", linestyle="--", linewidth=1)
        work_legend_handles = list(_WORK_PANEL_LEGEND_HANDLES)
        work_legend_handles.append(
            Line2D([0], [0], color="k", linestyle="--", linewidth=1, label="linear rise work")
        )
        ax_work.set_xlim(-1, len(history))
        ax_work.set_ylim(excess_axis_low - excess_pad, excess_axis_high + excess_pad)
        ax_work.set(xlabel="Evaluation", ylabel="(Work - linear rise Work) / |linear rise Work|",
                    title="Work vs. linear rise")
        ax_work.grid(True, alpha=0.3)
        ax_work.legend(handles=work_legend_handles, loc="upper right", fontsize=10, ncol=2)

        evaluations = history["evaluation"].to_numpy(float)
        ax_delta_p_history.plot(evaluations, p_target_rows, color="k", linestyle=":", linewidth=1.4,
                                label="target delta_P")
        ax_delta_p_history.plot(evaluations, p_lower_bound_rows, color="tab:red", linestyle="--", linewidth=1.6,
                                label="BO feasibility boundary")
        delta_p_legend_handles = list(_WORK_PANEL_LEGEND_HANDLES) + [
            Line2D([0], [0], color="k", linestyle=":", linewidth=1.4, label="target delta_P"),
            Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.6, label="BO feasibility boundary"),
        ]
        ax_delta_p_history.set_xlim(-1, len(history))
        ax_delta_p_history.set_ylim(p_axis_low, p_axis_high)
        ax_delta_p_history.set(xlabel="Evaluation", ylabel="delta_P (C/m^2)", title="delta_P vs. evaluation")
        ax_delta_p_history.grid(True, alpha=0.3)
        ax_delta_p_history.legend(handles=delta_p_legend_handles, loc="best", fontsize=8, ncol=2)

        ax_full.plot(full_time_s, full_voltage_v, color="tab:blue", linewidth=1.2, label="V(t)")
        ax_full_i.plot(full_time_s, full_current_a, color="tab:green", linewidth=1.0, alpha=0.85, label="I(t)")
        ax_full.axvspan(timing.first_rise_start_s, timing.first_rise_start_s + timing.rise_time_s,
                        color="tab:orange", alpha=0.15, label="pulse 1 rise")
        ax_full.axvspan(timing.second_rise_start_s, timing.second_rise_start_s + timing.rise_time_s,
                        color="tab:purple", alpha=0.15, label="pulse 2 rise")
        ax_full.set(title="Full captured waveform", xlabel="Time from trigger (s)", ylabel="Voltage (V)")
        ax_full.set_xlim(full_t_low, full_t_high)
        ax_full.set_ylim(full_v_low, full_v_high)
        ax_full_i.set_ylim(full_i_low, full_i_high)
        ax_full.yaxis.set_major_locator(MaxNLocator(5))
        ax_full_i.yaxis.set_major_locator(MaxNLocator(5))
        # clear() resets a twinx axis' label/ticks to the left side; put them back.
        ax_full_i.yaxis.tick_right()
        ax_full_i.yaxis.set_label_position("right")
        ax_full_i.set_ylabel("Current (A)", color="tab:green")
        ax_full_i.tick_params(axis="y", colors="tab:green")
        ax_full.grid(True, alpha=0.3)
        full_handles, full_labels = ax_full.get_legend_handles_labels()
        i_handles, i_labels = ax_full_i.get_legend_handles_labels()
        ax_full.legend(full_handles + i_handles, full_labels + i_labels, loc="upper right", fontsize=8)

        figure.suptitle(
            f"Evaluation {int(row['evaluation'])} | {row['phase']} | feasible={is_feasible}",
            fontsize=14,
        )
        figure.tight_layout(pad=2.0)
        return []

    movie = animation.FuncAnimation(figure, update, frames=len(history),
                                    interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        writer = animation.FFMpegWriter(fps=fps)
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(f"Unsupported video extension {suffix!r}; use .mp4 or .gif.")
    movie.save(output_path, writer=writer)
    plt.close(figure)
    if was_interactive:
        plt.ion()
    return output_path


def default_output_path(history_path: Path) -> Path:
    if history_path.stem.startswith("bo_history_"):
        default_name = history_path.stem.replace("bo_history_", "bo_evolution_", 1) + ".mp4"
    else:
        default_name = "bo_evolution.mp4"
    return history_path.with_name(default_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--data-folder", type=Path, help="Folder containing all run CSV files")
    selection.add_argument("--history", type=Path, help="Exact timestamped bo_history CSV")
    parser.add_argument("--output", type=Path, help="Output video path (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    cfg = ExperimentConfig()
    if args.history is not None:
        history_path = args.history
    else:
        data_folder = args.data_folder or DATA_FOLDER or latest_run_folder(cfg.data_mother_path)
        history_path = latest_history(data_folder)
    run_config = load_run_config(history_path)
    if run_config is not None:
        area_m2 = float(run_config["device"]["electrode_area_m2"])
        waveform_cfg = PUNDWaveformConfig(**run_config["waveform"])
    else:
        area_m2 = cfg.device.electrode_area_m2
        waveform_cfg = cfg.waveform
        print(f"No run_config_*.json found next to {history_path.name}; "
              "using the current config.py electrode_area_m2/waveform settings.")
    output_path = args.output or default_output_path(history_path)
    result = make_video(history_path.resolve(), output_path.resolve(), area_m2, waveform_cfg, args.fps)
    print(f"Evolution video saved to {result}")


if __name__ == "__main__":
    main()
