"""Drift-test reporting: a static Work/delta_P-vs-time plot (rest periods
shaded and labeled with their duration) and an MP4/GIF showing how the
PUND I1-I2 (subtracted current) trace evolves measurement-by-measurement,
with that same Work/delta_P-vs-time plot embedded as live side panels (a
moving marker tracks "now" through the run, rests included). Both are
called automatically at the end of run_drift_test.py, or run this file
standalone against an existing drift_log_*.csv to regenerate either or
both.

pulse_subtraction.subtract_pund_pulses normally only integrates over
[0, rise_time_s] (the BO objective's window); the video widens that same
subtraction to [0, rise_time_s + pulse_width_s] so the flat-top plateau is
visible too, and shades the two regions differently in the background. Each
inter-batch rest gets its own short paused stretch of video (a bold
"RESTING" banner, plus the side panels' "now" marker sweeping across the
shaded rest interval), so rests are as visible as the measurements
themselves.
"""

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import matplotlib.animation as animation
import numpy as np
import pandas as pd

from config import PUNDWaveformConfig
from pulse_subtraction import subtract_pund_pulses
from waveform import positive_rise_windows

# User-editable folder for standalone use; the newest drift_log_*.csv in it
# is used when no path is given on the command line.
DATA_FOLDER: Path | None = None

# Fixed number of video frames spent "inside" each rest, independent of the
# rest's real duration (which can range from seconds to many minutes) --
# long enough to read the banner, short enough not to dominate the video.
REST_PAUSE_FRAMES = 24


def _read_log(log_path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    batches, elapsed_s, works, deltas_p = [], [], [], []
    with log_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            batches.append(int(row["batch"]))
            elapsed_s.append(float(row["elapsed_s"]))
            works.append(float(row["Work"]))
            deltas_p.append(float(row["delta_P"]))
    return batches, elapsed_s, works, deltas_p


def _read_rests(rests_path: Path) -> list[tuple[float, float]]:
    intervals = []
    with rests_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            intervals.append((float(row["rest_start_s"]), float(row["rest_end_s"])))
    return intervals


def _infer_rests(batches: list[int], elapsed_s: list[float]) -> list[tuple[float, float]]:
    """Fallback for older drift-test runs recorded before drift_rests_*.csv
    existed: treat the elapsed_s gap wherever the batch number changes as
    that batch's rest. Includes one measurement's acquisition time on top of
    the actual sleep, so it slightly overstates the true rest duration."""
    intervals = []
    for i in range(1, len(batches)):
        if batches[i] != batches[i - 1]:
            intervals.append((elapsed_s[i - 1], elapsed_s[i]))
    return intervals


def _load_rest_intervals(log_path: Path, rests_path: Path | None,
                         batches: list[int], elapsed_s: list[float]) -> tuple[list[tuple[float, float]], bool]:
    if rests_path is None:
        rests_path = log_path.with_name(log_path.name.replace("drift_log_", "drift_rests_"))
    rests_path = Path(rests_path)
    if rests_path.exists():
        return _read_rests(rests_path), True
    return _infer_rests(batches, elapsed_s), False


def _delta_p_recovery_per_rest(
    elapsed_s: list[float], deltas_p: list[float], rest_intervals: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """For each rest, (duration, delta_P recovered) using the last point before
    the rest starts and the first point after it ends -- positive means
    delta_P rose (recovered) during the rest, negative means it kept falling.
    Skips a rest if either side has no data point (e.g. a rest right at the
    start or end of the log)."""
    recoveries = []
    for rest_start, rest_end in rest_intervals:
        before = [(t, p) for t, p in zip(elapsed_s, deltas_p) if t <= rest_start]
        after = [(t, p) for t, p in zip(elapsed_s, deltas_p) if t >= rest_end]
        if not before or not after:
            continue
        _, p_before = before[-1]
        _, p_after = after[0]
        recoveries.append((rest_end - rest_start, p_after - p_before))
    return recoveries


def _shade_rests(axis, rest_intervals: list[tuple[float, float]], rest_word: str) -> None:
    y_low, y_high = axis.get_ylim()
    label_y = y_low + 0.97 * (y_high - y_low)
    for index, (rest_start, rest_end) in enumerate(rest_intervals):
        axis.axvspan(rest_start, rest_end, color="0.6", alpha=0.3,
                    label=rest_word if index == 0 else None)
        duration = rest_end - rest_start
        axis.text((rest_start + rest_end) / 2.0, label_y, f"{duration:.0f}s",
                  ha="center", va="top", fontsize=8, color="0.3")
    axis.set_ylim(y_low, y_high)
    if rest_intervals:
        axis.legend(loc="upper left", fontsize=8)


def plot_drift(log_path: Path, rests_path: Path | None = None, output_path: Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    log_path = Path(log_path)
    if output_path is None:
        output_path = log_path.with_name(log_path.name.replace("drift_log_", "drift_plot_")).with_suffix(".png")

    batches, elapsed_s, works, deltas_p = _read_log(log_path)
    rest_intervals, rests_are_measured = _load_rest_intervals(log_path, rests_path, batches, elapsed_s)

    figure, (ax_work, ax_delta_p, ax_loss) = plt.subplots(3, 1, figsize=(9, 10))
    ax_work.sharex(ax_delta_p)
    ax_work.plot(elapsed_s, works, marker="o", markersize=3, color="tab:blue")
    ax_work.set(title="Work drift", ylabel="Work (J)")
    ax_work.grid(True, alpha=0.3)
    ax_delta_p.plot(elapsed_s, deltas_p, marker="o", markersize=3, color="tab:purple")
    ax_delta_p.set(title="delta_P drift", xlabel="Elapsed time (s)", ylabel="delta_P (C/m^2)")
    ax_delta_p.grid(True, alpha=0.3)

    rest_word = "Rest" if rests_are_measured else "Rest (inferred)"
    for axis in (ax_work, ax_delta_p):
        _shade_rests(axis, rest_intervals, rest_word)

    recoveries = _delta_p_recovery_per_rest(elapsed_s, deltas_p, rest_intervals)
    if recoveries:
        durations, recovery_values = zip(*recoveries)
        ax_loss.scatter(durations, recovery_values, color="tab:red", zorder=3)
        ax_loss.set_xscale("log")
    ax_loss.set(title="Recovered delta_P vs rest duration", xlabel="Rest duration (s)",
                ylabel="Recovered delta_P (C/m^2)")
    ax_loss.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_loss.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


# --------------------------------------------------------------------------
# I1-I2 evolution video
# --------------------------------------------------------------------------

def latest_log(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"Data folder does not exist: {directory}")
    files = list(directory.glob("drift_log_*.csv"))
    if not files:
        raise FileNotFoundError(f"No drift_log_*.csv found in {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_run_config(log_path: Path) -> dict | None:
    """Load the run_config_<timestamp>.json saved alongside drift_log_<timestamp>.csv
    by storage.save_run_config(), if one exists (older runs predate that log)."""
    if not log_path.stem.startswith("drift_log_"):
        return None
    run_timestamp = log_path.stem.removeprefix("drift_log_")
    config_path = log_path.parent / f"run_config_{run_timestamp}.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


def _extended_timing(waveform_cfg: PUNDWaveformConfig):
    """Same positive-pulse rise onsets waveform.positive_rise_windows locates,
    but with the subtraction window stretched from just the rise out to rise
    + flat top -- i.e. including the part of the pulse the BO objective
    (rise-window-only) never looks at."""
    window_s = waveform_cfg.rise_time_s + waveform_cfg.pulse_width_s
    timing = dataclasses.replace(positive_rise_windows(waveform_cfg), rise_time_s=window_s)
    return timing, window_s


def _padded_bounds(values, pad_frac=0.08):
    values = np.asarray(values, dtype=float)
    low, high = float(np.min(values)), float(np.max(values))
    if np.isclose(low, high):
        span = max(abs(low), 1.0)
        return low - span * pad_frac, high + span * pad_frac
    pad = pad_frac * (high - low)
    return low - pad, high + pad


def _compute_idiff_traces(log_dir: Path, trace_files, timing, periods: int):
    """I1-I2(t) over the extended window, recomputed from each measurement's
    raw trace_*.csv (current_a is pre-subtraction; the saved rise_*.csv only
    covers the rise, not the plateau, so it can't be reused here)."""
    relative_time = None
    traces = []
    for name in trace_files:
        trace = pd.read_csv(log_dir / name, usecols=["time_s", "current_a"])
        relative_time, idiff = subtract_pund_pulses(
            trace["time_s"].to_numpy(float), trace["current_a"].to_numpy(float), timing, periods
        )
        traces.append(idiff)
    return relative_time, traces


def make_drift_video(log_path: Path, rests_path: Path | None, output_path: Path,
                     waveform_cfg: PUNDWaveformConfig | None = None,
                     periods: int = 1, fps: int = 10) -> Path:
    import matplotlib.pyplot as plt

    log_path = Path(log_path)
    run_config = load_run_config(log_path)
    if run_config is not None:
        if waveform_cfg is None:
            waveform_cfg = PUNDWaveformConfig(**run_config["waveform"])
        periods = int(run_config["scope_capture_periods"])
    elif waveform_cfg is None:
        waveform_cfg = PUNDWaveformConfig()
        print(f"No run_config_*.json found next to {log_path.name}; "
              "using config.py's current PUNDWaveformConfig defaults.")

    log = pd.read_csv(log_path)
    if log.empty:
        raise ValueError(f"{log_path} has no rows")
    timing, window_s = _extended_timing(waveform_cfg)
    relative_time, idiff_traces = _compute_idiff_traces(log_path.parent, log["trace_file"], timing, periods)

    batches = log["batch"].tolist()
    elapsed_s_list = log["elapsed_s"].tolist()
    rest_intervals, rests_are_measured = _load_rest_intervals(log_path, rests_path, batches, elapsed_s_list)
    rest_word = "rest" if rests_are_measured else "rest (inferred)"

    elapsed_s = log["elapsed_s"].to_numpy(float)
    works = log["Work"].to_numpy(float)
    deltas_p = log["delta_P"].to_numpy(float)
    i_low, i_high = _padded_bounds(np.concatenate(idiff_traces))

    # Map each rest to the row right before it (the last measurement of the
    # batch it follows), so the pause frames can be inserted right after that
    # measurement's frame.
    rest_after_row = {}
    for rest_start, rest_end in rest_intervals:
        preceding = np.flatnonzero(elapsed_s <= rest_start)
        if preceding.size:
            rest_after_row[int(preceding[-1])] = (rest_start, rest_end)

    frames = []  # ("measurement", row_index) or ("rest", row_index, frac_through_rest)
    for row_index in range(len(log)):
        frames.append(("measurement", row_index, None))
        if row_index in rest_after_row:
            for step in range(REST_PAUSE_FRAMES):
                frames.append(("rest", row_index, step / max(REST_PAUSE_FRAMES - 1, 1)))

    # This mirrors make_evolution_video.make_video's handling of control_panel.py's
    # plt.ion() -- purely rendered-to-file, so force non-interactive and restore after.
    was_interactive = plt.isinteractive()
    plt.ioff()
    figure = plt.figure(figsize=(14, 7))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.4, 1), hspace=0.5, wspace=0.3)
    ax_current = figure.add_subplot(grid[:, 0])
    ax_work = figure.add_subplot(grid[0, 1])
    ax_delta_p = figure.add_subplot(grid[1, 1], sharex=ax_work)
    reference_idiff = idiff_traces[0]

    def draw_drift_panel(axis, values, ylabel, title, current_elapsed, current_value):
        axis.clear()
        axis.plot(elapsed_s, values, marker="o", markersize=3, color="tab:blue" if ylabel.startswith("Work")
                  else "tab:purple")
        _shade_rests(axis, rest_intervals, rest_word)
        axis.axvline(current_elapsed, color="tab:red", linewidth=2, zorder=5, label="now")
        if current_value is not None:
            axis.scatter([current_elapsed], [current_value], color="tab:red", zorder=6, s=40)
        axis.set(title=title, xlabel="Elapsed time (s)", ylabel=ylabel)
        axis.grid(True, alpha=0.3)

    def update(frame_index):
        kind, row_index, frac = frames[frame_index]
        row = log.iloc[row_index]
        ax_current.clear()
        ax_current.axvspan(0.0, waveform_cfg.rise_time_s, color="tab:orange", alpha=0.15,
                           label="rise window")
        ax_current.axvspan(waveform_cfg.rise_time_s, window_s, color="tab:green", alpha=0.15,
                           label="flat-top plateau")
        ax_current.plot(relative_time, reference_idiff, "k--", linewidth=1.1,
                        label="first measurement (ref.)")

        if kind == "measurement":
            ax_current.plot(relative_time, idiff_traces[row_index], color="tab:blue", linewidth=1.8,
                            label="I1-I2 (this measurement)")
            current_elapsed = float(row["elapsed_s"])
            ax_current.set_title(
                f"Batch {int(row['batch'])}, meas {int(row['batch_iteration'])} | t={current_elapsed:.1f}s",
                fontsize=12)
            work_value, delta_p_value = float(row["Work"]), float(row["delta_P"])
        else:
            rest_start, rest_end = rest_after_row[row_index]
            ax_current.plot(relative_time, idiff_traces[row_index], color="tab:blue", linewidth=1.8,
                            alpha=0.3, label="I1-I2 (last measurement)")
            current_elapsed = rest_start + frac * (rest_end - rest_start)
            ax_current.set_title(f"Batch {int(row['batch'])} done | resting before batch {int(row['batch']) + 1}",
                                 fontsize=12)
            ax_current.text(0.5, 0.92, f"RESTING -- {rest_end - rest_start:.0f}s",
                            transform=ax_current.transAxes, ha="center", fontsize=14,
                            color="tab:red", weight="bold")
            work_value = delta_p_value = None

        ax_current.set_xlim(0.0, window_s)
        ax_current.set_ylim(i_low, i_high)
        ax_current.set(xlabel="Time since pulse rise onset (s)", ylabel="I1 - I2 (A)")
        ax_current.grid(True, alpha=0.3)
        ax_current.legend(loc="upper right", fontsize=8)

        draw_drift_panel(ax_work, works, "Work (J)", "Work drift", current_elapsed, work_value)
        draw_drift_panel(ax_delta_p, deltas_p, "delta_P (C/m^2)", "delta_P drift", current_elapsed, delta_p_value)
        figure.tight_layout()
        return []

    movie = animation.FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    output_path = Path(output_path)
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


def default_video_output_path(log_path: Path) -> Path:
    if log_path.stem.startswith("drift_log_"):
        name = log_path.stem.replace("drift_log_", "drift_idiff_video_", 1) + ".mp4"
    else:
        name = "drift_idiff_video.mp4"
    return log_path.with_name(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, nargs="?", help="Path to drift_log_*.csv")
    parser.add_argument("--data-folder", type=Path, default=None,
                        help="Folder containing drift_log_*.csv; the newest one is used")
    parser.add_argument("--rests-path", type=Path, default=None,
                        help="Path to the matching drift_rests_*.csv (default: inferred from log_path)")
    parser.add_argument("--output-plot", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--output-video", type=Path, default=None, help="Output video path (.mp4 or .gif)")
    parser.add_argument("--skip-plot", action="store_true", help="Don't generate the static drift plot")
    parser.add_argument("--skip-video", action="store_true", help="Don't generate the I1-I2 evolution video")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.log_path is not None:
        log_path = args.log_path
    else:
        data_folder = args.data_folder or DATA_FOLDER
        if data_folder is None:
            raise SystemExit("Provide log_path, or --data-folder, or set DATA_FOLDER in this file.")
        log_path = latest_log(data_folder)
    log_path = log_path.resolve()

    if not args.skip_plot:
        plot_path = plot_drift(log_path, args.rests_path, args.output_plot)
        print(f"Drift plot saved to {plot_path}")
    if not args.skip_video:
        output_video = args.output_video or default_video_output_path(log_path)
        video_path = make_drift_video(log_path, args.rests_path, output_video.resolve(), fps=args.fps)
        print(f"Drift I1-I2 evolution video saved to {video_path}")


if __name__ == "__main__":
    main()
