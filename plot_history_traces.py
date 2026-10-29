"""Standalone: copy this single file into a folder containing bo_history_*.csv
(plus its trace_*.csv/rise_*.csv/run_config_*.json siblings) and run it there.
No dependency on the rest of this experiment's package (config.py, waveform.py,
instruments.py, ...) -- only numpy, pandas, and matplotlib are required.

Overlays every iteration's I/V traces from a bo_history_*.csv onto static
plots: one figure for the full captured waveform, one for the rise window only.

    python plot_history_traces.py                      # uses the newest
                                                         # bo_history_*.csv next
                                                         # to this script
    python plot_history_traces.py --history <path>
    python plot_history_traces.py --data-folder <path>
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pandas as pd


def latest_history(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"Data folder does not exist: {directory}")
    files = list(directory.glob("bo_history_*.csv"))
    if not files:
        raise FileNotFoundError(f"No BO history CSV found in {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_run_config(history_path: Path) -> dict | None:
    """Load the run_config_<timestamp>.json saved alongside bo_history_<timestamp>.csv,
    if one exists (older runs predate that log and simply won't get rise-window shading)."""
    if not history_path.stem.startswith("bo_history_"):
        return None
    run_timestamp = history_path.stem.removeprefix("bo_history_")
    config_path = history_path.parent / f"run_config_{run_timestamp}.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


def positive_rise_windows(waveform_cfg: dict):
    """Offsets of each positive pulse's rise onset, relative to the AWG
    buffer restart (t=0 = start of pre_baseline_s). Mirrors waveform.py's
    positive_rise_windows, inlined here to keep this script dependency-free."""
    pulse_width_s = waveform_cfg["pulse_width_s"]
    rise_time_s = waveform_cfg["rise_time_s"]
    # Absent (older runs) or None means every other ramp reused rise_time_s.
    other_ramp_s = waveform_cfg.get("other_ramp_time_s") or rise_time_s
    reset_pulse_duration = 2.0 * pulse_width_s + 2.0 * other_ramp_s
    positive_pulse_duration = rise_time_s + pulse_width_s + other_ramp_s
    first_rise_start = waveform_cfg["pre_baseline_s"] + reset_pulse_duration + waveform_cfg["gap1_s"]
    second_rise_start = first_rise_start + positive_pulse_duration + waveform_cfg["gap2_s"]
    return first_rise_start, second_rise_start, rise_time_s


def load_run(history_path: Path):
    history = pd.read_csv(history_path).sort_values("evaluation").reset_index(drop=True)
    if history.empty:
        raise ValueError("History is empty")
    if "rise_file" not in history or "trace_file" not in history:
        raise ValueError("History is missing rise_file/trace_file columns")
    rises, full_traces = [], []
    for _, row in history.iterrows():
        rise_path = history_path.parent / str(row["rise_file"])
        if not rise_path.exists():
            raise FileNotFoundError(f"Missing rise trace referenced by history: {rise_path}")
        rises.append(pd.read_csv(rise_path))
        trace_path = history_path.parent / str(row["trace_file"])
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing full waveform/current trace referenced by history: {trace_path}")
        full_traces.append(pd.read_csv(trace_path))
    return history, rises, full_traces


def _evaluation_colors(history):
    evaluations = history["evaluation"].to_numpy(float)
    norm = Normalize(vmin=evaluations.min(), vmax=evaluations.max())
    return norm, plt.get_cmap("viridis")


def plot_full_waveform_traces(history, full_traces, rise_windows, output_path: Path) -> Path:
    norm, color_map = _evaluation_colors(history)
    figure, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for (_, row), trace in zip(history.iterrows(), full_traces):
        color = color_map(norm(row["evaluation"]))
        ax_v.plot(trace["time_s"], trace["voltage_v"], color=color, linewidth=0.8, alpha=0.7)
        ax_i.plot(trace["time_s"], trace["current_a"], color=color, linewidth=0.8, alpha=0.7)
    for axis in (ax_v, ax_i):
        if rise_windows is not None:
            first_start, second_start, rise_time_s = rise_windows
            axis.axvspan(first_start, first_start + rise_time_s, color="tab:orange", alpha=0.12, zorder=0)
            axis.axvspan(second_start, second_start + rise_time_s, color="tab:purple", alpha=0.12, zorder=0)
        axis.grid(True, alpha=0.3)
    ax_v.set(title="Full captured voltage, every iteration", ylabel="Voltage (V)")
    ax_i.set(title="Full captured current, every iteration", xlabel="Time from trigger (s)", ylabel="Current (A)")
    figure.colorbar(ScalarMappable(norm=norm, cmap=color_map), ax=[ax_v, ax_i]).set_label("Evaluation")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_rise_traces(history, rises, output_path: Path) -> Path:
    norm, color_map = _evaluation_colors(history)
    figure, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for (_, row), rise in zip(history.iterrows(), rises):
        color = color_map(norm(row["evaluation"]))
        ax_v.plot(rise["rise_time_s"], rise["rise_voltage_v"], color=color, linewidth=0.9, alpha=0.8)
        ax_i.plot(rise["rise_time_s"], rise["rise_current_diff_a"], color=color, linewidth=0.9, alpha=0.8)
    reference = rises[0]
    ax_v.plot(reference["rise_time_s"], reference["rise_voltage_v"], "k--", linewidth=1.8, label="linear rise (eval 0)")
    ax_i.plot(reference["rise_time_s"], reference["rise_current_diff_a"], "k--", linewidth=1.8, label="linear rise (eval 0)")
    for axis in (ax_v, ax_i):
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
    ax_v.set(title="Rise-window voltage, every iteration", ylabel="Voltage (V)")
    ax_i.set(title="Rise-window subtracted current (I1-I2), every iteration",
            xlabel="Rise time (s)", ylabel="Current (A)")
    figure.colorbar(ScalarMappable(norm=norm, cmap=color_map), ax=[ax_v, ax_i]).set_label("Evaluation")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_history(history_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    history, rises, full_traces = load_run(history_path)
    output_dir = output_dir or history_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(history_path)
    if run_config is not None:
        rise_windows = positive_rise_windows(run_config["waveform"])
    else:
        rise_windows = None
        print(f"No run_config_*.json found next to {history_path.name}; "
              "skipping the rise-window shading on the full-waveform plot.")

    full_path = plot_full_waveform_traces(history, full_traces, rise_windows,
                                          output_dir / f"{history_path.stem}_full_traces.png")
    rise_path = plot_rise_traces(history, rises, output_dir / f"{history_path.stem}_rise_traces.png")
    return full_path, rise_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--data-folder", type=Path, help="Folder containing the run CSV files")
    selection.add_argument("--history", type=Path, help="Exact timestamped bo_history CSV")
    parser.add_argument("--output-dir", type=Path, help="Where to save the PNGs (default: next to the history CSV)")
    args = parser.parse_args()
    data_folder = args.data_folder or Path(__file__).resolve().parent
    history_path = (args.history or latest_history(data_folder)).resolve()
    full_path, rise_path = plot_history(history_path, args.output_dir)
    print(f"Saved {full_path}")
    print(f"Saved {rise_path}")


if __name__ == "__main__":
    main()
