"""Standalone: pick and plot the best protocol of every setting in a
run_parameter_sweep.py sweep.

Each run's result is its confirmed best protocol: rank 1 of
confirm_summary_<timestamp>.csv, i.e. the mean Work reduction over its
confirmation repeats (see BOConfig.confirm_top). Runs from before the
confirmation phase existed fall back to their single best BO measurement
(lowest feasible Work_NORM), marked "single BO measurement" in the table.
Among the repeats of one setting, the run with the highest reduction is that
setting's best -- preferring runs whose confirmed delta_P reached the target.

Outputs, written into the sweep folder:
1. sweep_all_runs.csv: one row per complete run.
2. sweep_best_per_setting.csv: the best run of every setting, with its
   protocol (V0..Vn) and the run folder/config that reproduce it.
3. sweep_work_reduction.png: reduction vs the x parameter, one color per
   series value. Hollow dots: every run; filled line: best run per setting.
4. sweep_best_shapes_overlay.png: each setting's best protocol, measured
   rise voltage and I1-I2 averaged over its confirmation repeats, one column
   per series value, colored by the x parameter; dashed = linear rise.

x and series default to the swept settings (whichever of rise_basis,
rise_points, rise_time_s, amplitude_v vary across the runs). Only numpy,
pandas, and matplotlib are required.

    python plot_sweep_results.py --sweep-folder "D:\\Ferroelectric_BTO_optimization\\20261031\\rise_time_amplitude_scan"
    python plot_sweep_results.py --sweep-folder ... --x amplitude_v --series rise_time_s
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DEFAULT_SWEEP_FOLDER = Path(r"D:\Ferroelectric_BTO_optimization\20261030\rise_points_basis_scan")
SETTINGS = ["rise_basis", "rise_points", "rise_time_s", "amplitude_v"]
LABELS = {"rise_basis": "Rise basis", "rise_points": "Number of rise points",
          "rise_time_s": "Rise time (us)", "amplitude_v": "Amplitude (V)"}
# Categorical series colors, in fixed order (blue, orange, then more hues).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc9", "#c9a227"]
# One blue ramp, light -> dark with increasing x (ordinal, starts at step 250).
X_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def _name(setting: str) -> str:
    """LABELS without the unit, for "<name> <value with unit>" labels."""
    return LABELS[setting].split(" (")[0]


def _format(setting: str, value) -> str:
    if setting == "rise_time_s":
        return f"{value * 1e6:g} us"
    if setting == "amplitude_v":
        return f"{value:g} V"
    return str(value)


def _confirmed(run_dir: Path, timestamp: str, history: pd.DataFrame):
    """Rank-1 row of the run's confirmation summary and its repeats' rise files
    (plus the confirmation linear references'), or None if there is none."""
    path = run_dir / f"confirm_summary_{timestamp}.csv"
    if not path.exists():
        return None
    best = pd.read_csv(path).sort_values("rank").iloc[0]
    repeats = history[(history["phase"] == "confirmation") & (history["candidate"] == best["candidate"])]
    linear = history[history["phase"] == "confirmation reference"]
    return best, list(repeats["rise_file"]), list(linear["rise_file"])


def load_runs(sweep_folder: Path) -> pd.DataFrame:
    """One row per complete run: its settings, confirmed (or single BO) best
    protocol, and the rise files to average for its plots."""
    records = []
    for run_dir in sorted(p for p in sweep_folder.iterdir() if p.is_dir()):
        config_files = list(run_dir.glob("run_config_*.json"))
        if not config_files:
            print(f"  skipping {run_dir.name}: no run_config")
            continue
        cfg = json.loads(config_files[0].read_text())
        timestamp = config_files[0].stem.removeprefix("run_config_")
        history_path = run_dir / f"bo_history_{timestamp}.csv"
        if not history_path.exists():
            print(f"  skipping {run_dir.name}: no history")
            continue
        history = pd.read_csv(history_path)
        bo_rows = int((history["phase"] == "BO optimization").sum())
        if bo_rows < cfg["bo"]["bo_steps"]:
            print(f"  skipping {run_dir.name}: incomplete ({bo_rows}/{cfg['bo']['bo_steps']} BO steps)")
            continue
        waveform = cfg["waveform"]
        record = {
            "rise_basis": waveform["rise_basis"],
            "rise_points": int(cfg["bo"]["rise_points"]),
            "rise_time_s": float(waveform["rise_time_s"]),
            "amplitude_v": float(waveform.get("amplitude_v", waveform.get("positive_amplitude_v"))),
            "sweep_tag": cfg.get("sweep_tag", ""),
            "run_folder": run_dir.name,
            "run_config": config_files[0].name,
        }
        confirmed = _confirmed(run_dir, timestamp, history)
        if confirmed is None and cfg["bo"].get("confirm_top", 0) > 0:
            print(f"  skipping {run_dir.name}: no confirmation summary (stopped before it finished?)")
            continue
        names = [f"V{i}" for i in range(record["rise_points"])]
        if confirmed is not None:
            best, rise_files, linear_files = confirmed
            record.update({
                "result": "confirmed",
                "work_reduction_pct": float(best["work_reduction_pct"]),
                "work_reduction_sem_pct": 100.0 * float(best["Work_NORM_sem"]),
                "delta_P_NORM": float(best["delta_P_NORM_mean"]),
                "meets_target": bool(best["meets_target"]),
                "n_repeats": int(best["n"]),
                "candidate": int(best["candidate"]),
                "source_evaluation": int(best["source_evaluation"]),
                **{name: float(best[name]) for name in names},
            })
        else:
            bo = history[~history["phase"].astype(str).str.startswith("confirmation")]
            feasible = bo[bo["feasible"].astype(bool)]
            if feasible.empty:
                print(f"  skipping {run_dir.name}: no feasible row")
                continue
            if (bo["phase"] == "linear reference").any():
                # Drift-corrected run: each row's Work against its own linear reference.
                ratio = feasible["Work"] / feasible["WORK_SCALE"]
            else:
                ratio = feasible["Work"] / bo.loc[bo["phase"] == "linear rise", "Work"].mean()
            row = bo.loc[ratio.idxmin()]
            rise_files = [row["rise_file"]]
            linear_files = list(bo.loc[bo["phase"] == "linear rise", "rise_file"][:1])
            record.update({
                "result": "single BO measurement",
                "work_reduction_pct": 100.0 * (1.0 - float(ratio.min())),
                "work_reduction_sem_pct": np.nan,
                "delta_P_NORM": float(row["delta_P_NORM"]),
                "meets_target": True,
                "n_repeats": 1,
                "candidate": np.nan,
                "source_evaluation": int(row["evaluation"]),
                **{name: float(row[name]) for name in names},
            })
        record["rise_files"] = ";".join(str(run_dir / name) for name in rise_files)
        record["linear_rise_files"] = ";".join(str(run_dir / name) for name in linear_files)
        records.append(record)
    if not records:
        raise ValueError(f"No complete runs in {sweep_folder}")
    return pd.DataFrame(records)


def best_per_setting(runs: pd.DataFrame, settings: list[str]) -> pd.DataFrame:
    """The highest-reduction run of each setting, runs whose confirmed delta_P
    reached the target first."""
    ordered = runs.sort_values(["meets_target", "work_reduction_pct"], ascending=[False, False])
    return ordered.groupby(settings, sort=True).head(1).sort_values(settings).reset_index(drop=True)


def _mean_trace(files: str, column: str):
    traces = [pd.read_csv(path) for path in files.split(";") if path]
    return traces[0]["rise_time_s"].to_numpy() * 1e6, np.mean([t[column].to_numpy() for t in traces], axis=0)


def _x_values(runs: pd.DataFrame, x: str) -> pd.Series:
    return runs[x] * 1e6 if x == "rise_time_s" else runs[x]


def plot_reduction(runs: pd.DataFrame, best: pd.DataFrame, x: str, series: str | None,
                   output_path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(7.5, 5))
    groups = sorted(runs[series].unique()) if series else [None]
    spacing = np.diff(np.sort(_x_values(runs, x).unique().astype(float))) if runs[x].dtype != object else []
    step = 0.08 * (spacing.min() if len(spacing) else 1.0)
    for index, value in enumerate(groups):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        offset = (index - (len(groups) - 1) / 2) * step  # side by side at each x
        subset = runs if series is None else runs[runs[series] == value]
        chosen = best if series is None else best[best[series] == value]
        axis.scatter(_x_values(subset, x) + offset, subset["work_reduction_pct"], s=40,
                     facecolors="none", edgecolors=color, linewidths=1.2, zorder=2)
        label = "Best run per setting" if series is None else f"{_name(series)} {_format(series, value)}: best run"
        axis.errorbar(_x_values(chosen, x) + offset, chosen["work_reduction_pct"],
                      yerr=chosen["work_reduction_sem_pct"].fillna(0.0), color=color, marker="o",
                      markersize=8, linewidth=2, capsize=4, zorder=3, label=label)
    axis.scatter([], [], s=40, facecolors="none", edgecolors="0.4", label="Each run")
    axis.axhline(0.0, color="0.55", linestyle="--", linewidth=1, label="Linear rise (0 %)", zorder=1)
    axis.set_xticks(sorted(_x_values(runs, x).unique()))
    confirmed = (runs["result"] == "confirmed").all()
    axis.set(xlabel=LABELS[x], ylabel="Work reduction vs linear rise (%)",
             title=("Confirmed best protocol per run (mean ± s.e.m. of repeats)" if confirmed
                    else "Best Work reduction per run"))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def plot_best_shapes(best: pd.DataFrame, x: str, series: str | None, output_path: Path) -> Path:
    x_values = sorted(best[x].unique())
    colors = {v: X_RAMP[round(i * (len(X_RAMP) - 1) / max(len(x_values) - 1, 1))]
              for i, v in enumerate(x_values)}
    groups = sorted(best[series].unique()) if series else [None]
    figure, axes = plt.subplots(2, len(groups), figsize=(6.5 * len(groups), 8),
                                sharex=True, sharey="row", squeeze=False)
    for column, value in enumerate(groups):
        ax_v, ax_i = axes[0, column], axes[1, column]
        subset = best if series is None else best[best[series] == value]
        for _, run in subset.iterrows():
            color = colors[run[x]]
            for axis, key, scale in ((ax_v, "rise_voltage_v", 1.0), (ax_i, "rise_current_diff_a", 1e6)):
                if run["linear_rise_files"]:
                    t, linear = _mean_trace(run["linear_rise_files"], key)
                    axis.plot(t, linear * scale, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
                t, mean = _mean_trace(run["rise_files"], key)
                axis.plot(t, mean * scale, color=color, linewidth=1.8)
        title = "Best protocol per setting" if series is None else f"{_name(series)} {_format(series, value)}"
        ax_v.set_title(title)
        ax_i.set_xlabel("Rise time (us)")
        for axis in (ax_v, ax_i):
            axis.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Measured rise voltage (V)")
    axes[1, 0].set_ylabel("Subtracted current I1-I2 (uA)")
    handles = [Line2D([], [], color=colors[v], linewidth=2, label=f"{_name(x)}: {_format(x, v)}")
               for v in x_values]
    handles.append(Line2D([], [], color="0.4", linestyle="--", linewidth=1.0, label="Linear rise"))
    axes[0, -1].legend(handles=handles, loc="best", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def summarize_sweep(sweep_folder: Path, x: str | None = None, series: str | None = None) -> pd.DataFrame:
    """Write the tables and plots described in the module docstring; return
    the best-per-setting table."""
    runs = load_runs(sweep_folder)
    varying = [s for s in SETTINGS if runs[s].nunique() > 1]
    settings = varying or ["rise_basis"]
    if x is None:
        numeric = [s for s in varying if s != "rise_basis"]
        x = numeric[0] if numeric else "rise_points"
    if series is None:
        others = [s for s in varying if s != x]
        if len(others) > 1:
            raise ValueError(f"More than two settings vary ({varying}); pass --x and --series")
        series = others[0] if others else None
    best = best_per_setting(runs, settings)
    runs.sort_values(settings).to_csv(sweep_folder / "sweep_all_runs.csv", index=False)
    best.to_csv(sweep_folder / "sweep_best_per_setting.csv", index=False)
    print(best[[*settings, "work_reduction_pct", "work_reduction_sem_pct", "delta_P_NORM",
                "result", "run_folder"]].to_string(index=False))
    print(f"\nTables: {sweep_folder / 'sweep_all_runs.csv'}\n        {sweep_folder / 'sweep_best_per_setting.csv'}")
    print(f"Plot:   {plot_reduction(runs, best, x, series, sweep_folder / 'sweep_work_reduction.png')}")
    print(f"Plot:   {plot_best_shapes(best, x, series, sweep_folder / 'sweep_best_shapes_overlay.png')}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-folder", type=Path, default=DEFAULT_SWEEP_FOLDER)
    parser.add_argument("--x", choices=SETTINGS, help="x axis (default: a swept numeric setting)")
    parser.add_argument("--series", choices=SETTINGS, help="one color/column per value (default: the other swept setting)")
    args = parser.parse_args()
    summarize_sweep(args.sweep_folder, args.x, args.series)


if __name__ == "__main__":
    main()
