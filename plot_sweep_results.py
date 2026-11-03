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
4. sweep_work_delta_p_absolute.png: the same runs' Work (top, J) and
   delta_P (bottom, C/m^2), not normalized, vs the x parameter: hollow dots
   (Work only) every run's best protocol; filled line, Work: the best run per setting
   (mean +/- s.e.m. of its confirmation repeats), delta_P: every run's best
   protocol averaged (mean +/- std across runs); dashed line with open
   squares the linear rise averaged over all runs (mean +/- std across runs).
5. sweep_best_shapes_overlay.png: each setting's best protocol: measured
   rise voltage, I1-I2 and P(t) = integral of I1-I2 / electrode area, averaged
   over its confirmation repeats, vs time in the rise; one column per series
   value, colored by the x parameter; dashed = linear rise.
6. sweep_best_shapes_normalized.png: the same vs normalized time t/tau.
Both shape plots cut TRIM_S off each end of the rise window: the AWG's step
onto the flat top (and from 0 V to V0) already starts inside the window.

x and series default to the swept settings (whichever of rise_basis,
rise_points, rise_time_s, amplitude_v vary across the runs). Only numpy,
pandas, and matplotlib are required.

    python plot_sweep_results.py --sweep-folder "D:\\Ferroelectric_BTO_optimization\\20261031\\rise_time_amplitude_scan"
    python plot_sweep_results.py --sweep-folder ... --x amplitude_v --series rise_time_s
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

DEFAULT_SWEEP_FOLDER = Path(r"D:\Ferroelectric_BTO_optimization\20261030\rise_points_basis_scan")
SETTINGS = ["rise_basis", "rise_points", "rise_time_s", "amplitude_v"]
LABELS = {"rise_basis": "Rise basis", "rise_points": "Number of rise points",
          "rise_time_s": "Rise time (us)", "amplitude_v": "Amplitude (V)"}
# Categorical series colors, in fixed order (blue, orange, then more hues).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#2f9e6e", "#8a5cc9", "#c9a227"]
# One blue ramp, light -> dark with increasing x (ordinal, starts at step 250).
X_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
# Cut from each end of the rise window in the shape plots (see docstring).
TRIM_S = 20e-9


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
    """Rank-1 row of the run's confirmation summary and the history rows of its
    repeats and of the confirmation linear references, or None if there is none."""
    path = run_dir / f"confirm_summary_{timestamp}.csv"
    if not path.exists():
        return None
    best = pd.read_csv(path).sort_values("rank").iloc[0]
    repeats = history[(history["phase"] == "confirmation") & (history["candidate"] == best["candidate"])]
    linear = history[history["phase"] == "confirmation reference"]
    return best, repeats, linear


def _mean_sem(values: pd.Series) -> tuple[float, float]:
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
    return float(values.mean()), float(sem)


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
            "electrode_area_m2": float(cfg["device"]["electrode_area_m2"]),
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
            best, repeats, linear = confirmed
            rise_files, linear_files = list(repeats["rise_file"]), list(linear["rise_file"])
            work_j, work_sem_j = _mean_sem(repeats["Work"])
            linear_work_j, linear_work_sem_j = _mean_sem(linear["Work"])
            delta_p, delta_p_sem = _mean_sem(repeats["delta_P"])
            linear_delta_p, linear_delta_p_sem = _mean_sem(linear["delta_P"])
            record.update({
                "result": "confirmed",
                "work_reduction_pct": float(best["work_reduction_pct"]),
                "work_reduction_sem_pct": 100.0 * float(best["Work_NORM_sem"]),
                "delta_P_NORM": float(best["delta_P_NORM_mean"]),
                "meets_target": bool(best["meets_target"]),
                "n_repeats": int(best["n"]),
                "candidate": int(best["candidate"]),
                "source_evaluation": int(best["source_evaluation"]),
                "work_j": work_j, "work_sem_j": work_sem_j,
                "linear_work_j": linear_work_j, "linear_work_sem_j": linear_work_sem_j,
                "delta_P": delta_p, "delta_P_sem": delta_p_sem,
                "linear_delta_P": linear_delta_p, "linear_delta_P_sem": linear_delta_p_sem,
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
                "work_j": float(row["Work"]), "work_sem_j": np.nan,
                "linear_work_j": float(row["Work"]) / float(ratio.min()), "linear_work_sem_j": np.nan,
                "delta_P": float(row["delta_P"]), "delta_P_sem": np.nan,
                "linear_delta_P": float(row["delta_P"]) / float(row["delta_P_NORM"]), "linear_delta_P_sem": np.nan,
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


def _mean_rise(files: str, area_m2: float):
    """Time (s), voltage, I1-I2 and P = cumulative integral of I1-I2 / area,
    averaged over the rise files, with TRIM_S cut off each end (P starts at 0
    after the cut)."""
    traces = [pd.read_csv(path) for path in files.split(";") if path]
    t = traces[0]["rise_time_s"].to_numpy()
    keep = (t >= t[0] + TRIM_S) & (t <= t[-1] - TRIM_S)
    voltage, current = (np.mean([trace[column].to_numpy() for trace in traces], axis=0)[keep]
                        for column in ("rise_voltage_v", "rise_current_diff_a"))
    t = t[keep]
    charge = np.concatenate([[0.0], np.cumsum(0.5 * (current[1:] + current[:-1]) * np.diff(t))])
    return t, voltage, current, charge / area_m2


def _save(figure, output_path: Path, attempts: int = 5) -> None:
    """savefig, retrying while another program (image preview, virus scan)
    briefly holds the file -- Windows then raises OSError "Invalid argument"."""
    for attempt in range(attempts):
        try:
            figure.savefig(output_path, dpi=200)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1.0)


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
    _save(figure, output_path)
    plt.close(figure)
    return output_path


def _energy_unit(values: pd.Series) -> tuple[float, str]:
    """Scale and SI unit that put the typical Work between 1 and 1000."""
    exponent = 3 * int(np.floor(np.log10(np.nanmedian(np.abs(values))) / 3))
    exponent = min(max(exponent, -15), 0)
    return 10.0 ** -exponent, {0: "J", -3: "mJ", -6: "uJ", -9: "nJ", -12: "pJ", -15: "fJ"}[exponent]


def _power_law(x_values, y_values) -> dict:
    """Least-squares fit of log y = log A + alpha log x, i.e. y = A * x^alpha,
    with standard errors (NaN with fewer than 4 points) and R^2 in log space."""
    log_x, log_y = np.log(np.asarray(x_values, float)), np.log(np.asarray(y_values, float))
    if len(log_x) > 3:
        (alpha, log_a), cov = np.polyfit(log_x, log_y, 1, cov=True)
        alpha_err, log_a_err = np.sqrt(np.diag(cov))
    else:
        alpha, log_a = np.polyfit(log_x, log_y, 1)
        alpha_err = log_a_err = np.nan
    residual = log_y - (log_a + alpha * log_x)
    r2 = 1.0 - np.sum(residual ** 2) / np.sum((log_y - log_y.mean()) ** 2)
    a = np.exp(log_a)
    return {"alpha": alpha, "alpha_err": alpha_err, "A": a, "A_err": a * log_a_err, "r2": r2,
            "n_points": len(log_x)}


# Symbol of the x parameter in the power-law fit labels.
SYMBOLS = {"rise_time_s": "τ", "amplitude_v": "V", "rise_points": "n"}


def plot_absolute(runs: pd.DataFrame, best: pd.DataFrame, x: str, series: str | None,
                  output_path: Path) -> Path:
    """Work (top) and delta_P (bottom), not normalized, vs x. For a numeric x,
    both Work lines get a power-law fit Work = A * x^alpha (dotted), written
    to sweep_power_law_fit.csv next to output_path."""
    fits = []
    scale, unit = _energy_unit(pd.concat([runs["work_j"], runs["linear_work_j"]]))
    # (best protocol, its s.e.m., linear rise, scale, y label, solid line averages all runs?)
    # delta_P drifts between runs more than the best protocol gains over the
    # linear rise, so one chosen run's delta_P says little: average every run's.
    panels = (("work_j", "work_sem_j", "linear_work_j", scale, f"Work ({unit})", False),
              ("delta_P", "delta_P_sem", "linear_delta_P", 1.0, "ΔP (C/m²)", True))
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 8.5), sharex=True)
    groups = sorted(runs[series].unique()) if series else [None]
    spacing = np.diff(np.sort(_x_values(runs, x).unique().astype(float))) if runs[x].dtype != object else []
    step = 0.08 * (spacing.min() if len(spacing) else 1.0)
    for axis, (column, sem, linear, factor, ylabel, averaged) in zip(axes, panels):
        for index, value in enumerate(groups):
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            offset = (index - (len(groups) - 1) / 2) * step  # side by side at each x
            subset = runs if series is None else runs[runs[series] == value]
            chosen = best if series is None else best[best[series] == value]
            prefix = "" if series is None else f"{_name(series)} {_format(series, value)}: "
            if not averaged:
                axis.scatter(_x_values(subset, x) + offset, subset[column] * factor, s=40,
                             facecolors="none", edgecolors=color, linewidths=1.2, zorder=2)
            if averaged:
                best_runs = subset.groupby(x)[column].agg(["mean", "std"]).reset_index()
                axis.errorbar(_x_values(best_runs, x) + offset, best_runs["mean"] * factor,
                              yerr=best_runs["std"].fillna(0.0) * factor, color=color, marker="o",
                              markersize=8, linewidth=2, capsize=4, zorder=3,
                              label=f"{prefix}best protocol (mean ± std of runs)")
            else:
                axis.errorbar(_x_values(chosen, x) + offset, chosen[column] * factor,
                              yerr=chosen[sem].fillna(0.0) * factor, color=color, marker="o",
                              markersize=8, linewidth=2, capsize=4, zorder=3,
                              label=f"{prefix}best protocol (best run)")
            # Linear rise: every run of the setting, not just the best one (runs drift apart in time).
            linear_runs = subset.groupby(x)[linear].agg(["mean", "std"]).reset_index()
            # Nudged right so its error bar doesn't hide the best protocol's.
            axis.errorbar(_x_values(linear_runs, x) + offset + step, linear_runs["mean"] * factor,
                          yerr=linear_runs["std"].fillna(0.0) * factor, color=color, marker="s",
                          markersize=7, markerfacecolor="white", linewidth=1.2, linestyle="--", capsize=3,
                          zorder=3, label=f"{prefix}linear rise (mean ± std of runs)")
            if not averaged and runs[x].dtype != object:
                for name, x_fit, y_fit in (("best protocol", _x_values(chosen, x), chosen[column] * factor),
                                           ("linear rise", _x_values(linear_runs, x), linear_runs["mean"] * factor)):
                    fit = _power_law(x_fit, y_fit)
                    grid = np.geomspace(x_fit.min(), x_fit.max(), 50)
                    axis.plot(grid, fit["A"] * grid ** fit["alpha"], color=color, linestyle=":", linewidth=1.5,
                              zorder=4, label=f"{prefix}{name} fit: ∝ {SYMBOLS.get(x, 'x')}"
                                              f"^({fit['alpha']:.3f} ± {fit['alpha_err']:.3f})")
                    fits.append({**({series: value} if series else {}), "line": name, **fit,
                                 "A_units": f"{ylabel.split('(')[-1].rstrip(')')} at {LABELS[x]} = 1"})
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if not averaged:
            axis.scatter([], [], s=40, facecolors="none", edgecolors="0.4", label="Each run (best protocol)")
        axis.legend(loc="best", fontsize=9)
    # Work on log-log axes (x is shared, so delta_P gets the log x too): a
    # power law Work ~ x^a is a straight line.
    axes[0].set_yscale("log")
    if runs[x].dtype != object:
        axes[1].set_xscale("log")
    plain = FormatStrFormatter("%.3g")
    for axis in (axes[0].yaxis, axes[1].xaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(NullFormatter())
    axes[0].yaxis.set_major_locator(LogLocator(subs=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)))
    axes[1].set_xticks(sorted(_x_values(runs, x).unique()))
    axes[1].set_xlabel(LABELS[x])
    confirmed = (runs["result"] == "confirmed").all()
    axes[0].set_title("Confirmed best protocol per run (mean ± s.e.m. of repeats)" if confirmed
                      else "Best protocol per run")
    figure.tight_layout()
    _save(figure, output_path)
    plt.close(figure)
    if fits:
        table = pd.DataFrame(fits)
        table.to_csv(output_path.with_name("sweep_power_law_fit.csv"), index=False)
        print(table.to_string(index=False))
    return output_path


def plot_best_shapes(best: pd.DataFrame, x: str, series: str | None, output_path: Path,
                     normalized: bool = False) -> Path:
    """Measured rise voltage, I1-I2 and P(t) of each setting's best protocol
    vs time in the rise (us), or vs t/tau if normalized."""
    x_values = sorted(best[x].unique())
    if len(x_values) <= len(X_RAMP):
        colors = {v: X_RAMP[round(i * (len(X_RAMP) - 1) / max(len(x_values) - 1, 1))]
                  for i, v in enumerate(x_values)}
    else:  # more values than ramp steps: sample a continuous sequential map instead of repeating steps
        colors = dict(zip(x_values, plt.cm.viridis(np.linspace(0.0, 0.9, len(x_values)))))
    groups = sorted(best[series].unique()) if series else [None]
    figure, axes = plt.subplots(3, len(groups), figsize=(6.5 * len(groups), 11),
                                sharex=True, sharey="row", squeeze=False)
    for column, value in enumerate(groups):
        subset = best if series is None else best[best[series] == value]
        for _, run in subset.iterrows():
            color = colors[run[x]]
            for files, style in ((run["linear_rise_files"], {"linestyle": "--", "linewidth": 1.0, "alpha": 0.7}),
                                 (run["rise_files"], {"linewidth": 1.8})):
                if not files:
                    continue
                t, voltage, current, polarization = _mean_rise(files, run["electrode_area_m2"])
                t_plot = t / run["rise_time_s"] if normalized else t * 1e6
                for axis, values in zip(axes[:, column], (voltage, current * 1e6, polarization)):
                    axis.plot(t_plot, values, color=color, **style)
        title = "Best protocol per setting" if series is None else f"{_name(series)} {_format(series, value)}"
        axes[0, column].set_title(title)
        axes[-1, column].set_xlabel("Normalized time t/τ" if normalized else "Time in rise (us)")
        for axis in axes[:, column]:
            axis.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Measured rise voltage (V)")
    axes[1, 0].set_ylabel("Subtracted current I1-I2 (uA)")
    axes[2, 0].set_ylabel("P(t) = ∫(I1-I2)dt / area (C/m²)")
    figure.suptitle(f"First and last {TRIM_S * 1e9:g} ns of the rise window cut", fontsize=10, color="0.35")
    handles = [Line2D([], [], color=colors[v], linewidth=2, label=f"{_name(x)}: {_format(x, v)}")
               for v in x_values]
    handles.append(Line2D([], [], color="0.4", linestyle="--", linewidth=1.0, label="Linear rise"))
    axes[0, -1].legend(handles=handles, loc="best", fontsize=9)
    figure.tight_layout()
    _save(figure, output_path)
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
    print(best[[*settings, "work_reduction_pct", "work_reduction_sem_pct", "work_j", "linear_work_j", "delta_P", "linear_delta_P",
                "result", "run_folder"]].to_string(index=False))
    print(f"\nTables: {sweep_folder / 'sweep_all_runs.csv'}\n        {sweep_folder / 'sweep_best_per_setting.csv'}")
    print(f"Plot:   {plot_reduction(runs, best, x, series, sweep_folder / 'sweep_work_reduction.png')}")
    print(f"Plot:   {plot_absolute(runs, best, x, series, sweep_folder / 'sweep_work_delta_p_absolute.png')}")
    print(f"Plot:   {plot_best_shapes(best, x, series, sweep_folder / 'sweep_best_shapes_overlay.png')}")
    print(f"Plot:   {plot_best_shapes(best, x, series, sweep_folder / 'sweep_best_shapes_normalized.png', normalized=True)}")
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
