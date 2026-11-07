"""One BO run in one figure: what each phase does, with the run's own data.

Panels: the whole run measurement by measurement (Work and delta_P,
normalized to the drift-corrected linear reference, with the phases shaded);
1 warm-up settling; 4 the drift reference built from the linear rises; the
Work vs delta_P trade-off with the BO constraint; the rise shapes BO tried and
kept; the measured I1-I2 and V vs P (Work/area = area under V(P)) of the
linear rise and the confirmed best; 5 the confirmation repeats. Writes
<history stem>_overview.png in the run folder.

    python plot_run_overview.py --data-folder D:\\...\\BO_pund_rise_<device>_<HHMM>_<tag>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from bo import LINEAR_PHASES, _centered_mean
from plot_sweep_results import SERIES_COLORS, _mean_rise, _save
from waveform import rise_voltage

INK, MUTED, BAND = "#222222", "#8a8a8a", "#f1f1f1"
RANDOM, BO = SERIES_COLORS[1], SERIES_COLORS[0]
CANDIDATE_COLORS = SERIES_COLORS[2:]  # confirmation candidates 1, 2, 3
P_FLOOR = 0.96  # delta_P_NORM axis floor (timeline); lower points are drawn on it as triangles
P_FLOOR_TRADEOFF = 0.95  # the same for the trade-off panel's x axis


def load(run_dir: Path, config_path: Path | None = None):
    """config_path: a run_config JSON to use instead of the run's own (e.g. a
    reconstruction when the run's file is missing)."""
    history_path = max(path for path in run_dir.glob("bo_history_*.csv")
                       if not path.stem.endswith("eigenvectors"))
    timestamp = history_path.stem.removeprefix("bo_history_")
    history = pd.read_csv(history_path).set_index("evaluation")
    config_path = config_path or run_dir / f"run_config_{timestamp}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / f"confirm_summary_{timestamp}.csv")
    warmup_path = run_dir / "warmup" / f"warmup_{timestamp}.csv"
    warmup = pd.read_csv(warmup_path) if warmup_path.exists() else None
    return history_path, history, config, summary, warmup


def _groups(history: pd.DataFrame, threshold: float):
    """(rows, label, plot kwargs) per measurement kind, in legend order."""
    bo = history["phase"] == "BO optimization"
    feasible = history["P_lower_NORM"] >= threshold
    groups = [
        (history[history["phase"].isin(LINEAR_PHASES)], "linear rise (target / drift reference)",
         dict(marker="s", s=16, color=INK)),
        (history[history["phase"] == "random initial"], "random shape (initial phase)",
         dict(marker="o", s=18, color=RANDOM)),
        (history[bo & ~feasible], "BO step, below the ΔP bound",
         dict(marker="o", s=18, facecolors="none", edgecolors=BO, linewidths=0.9)),
        (history[bo & feasible], "BO step, meets the ΔP bound", dict(marker="o", s=22, color=BO)),
    ]
    confirmation = history[history["phase"] == "confirmation"]
    for candidate, rows in confirmation.groupby("candidate"):
        groups.append((rows, f"confirmation, candidate {int(candidate)}",
                       dict(marker="D", s=18, color=CANDIDATE_COLORS[int(candidate) - 1])))
    return groups


def _scatter(axis, x, y, floor=None, **kwargs):
    """Scatter; with floor, points below it are drawn on it as down-triangles."""
    if floor is None:
        return axis.scatter(x, y, zorder=3, **kwargs)
    low = y < floor
    axis.scatter(x[~low], y[~low], zorder=3, **kwargs)
    axis.scatter(x[low], np.full(low.sum(), floor), zorder=3, **{**kwargs, "marker": "v"})


def _work_axis(axis) -> None:
    """Log Work_NORM y axis with plain-number ticks."""
    ticks = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]
    axis.set_yscale("log")
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{tick:g}" for tick in ticks])
    axis.yaxis.set_minor_formatter(NullFormatter())


def plot_timeline(ax_work, ax_p, history, threshold, best_color):
    first_bo = history.index[history["phase"] == "BO optimization"].min()
    first_confirm = history.index[history["phase"].isin(("confirmation", "confirmation reference"))].min()
    last = history.index.max()
    bands = [(-0.5, first_bo - 0.5, "② Initial: linear rises + random shapes"),
             (first_bo - 0.5, first_confirm - 0.5, "③ BO: constrained Expected Improvement\n"
                                                   "(+ ④ a linear reference every 10 steps)"),
             (first_confirm - 0.5, last + 0.5, "⑤ Confirmation:\ntop 3 × 10 rounds")]
    for axis in (ax_work, ax_p):
        for index, (start, stop, _) in enumerate(bands):
            if index != 1:
                axis.axvspan(start, stop, color=BAND, zorder=0)
    for start, stop, text in bands:
        ax_work.text((start + stop) / 2, 1.02, text, transform=ax_work.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=9, color=INK)
    for rows, label, kwargs in _groups(history, threshold):
        _scatter(ax_work, rows.index.to_numpy(), rows["Work_NORM"].to_numpy(), label=label, **kwargs)
        _scatter(ax_p, rows.index.to_numpy(), rows["delta_P_NORM"].to_numpy(), floor=P_FLOOR, **kwargs)
    # Best Work so far among shapes meeting the BO bound (what BO is improving).
    searched = history[history["phase"].isin(("random initial", "BO optimization"))]
    best = searched["Work_NORM"].where(searched["P_lower_NORM"] >= threshold).cummin()
    best = best.reindex(history.index[history.index < first_confirm]).ffill()
    ax_work.step(best.index, best, where="post", color=best_color, linewidth=2,
                 label="best Work so far that meets the ΔP bound", zorder=4)
    ax_work.axhline(1.0, color=INK, linewidth=0.8, linestyle=":")
    _work_axis(ax_work)
    ax_work.set_ylabel("Work / linear-rise Work")
    ax_work.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.95)
    ax_p.axhspan(threshold, 1.02, color=SERIES_COLORS[2], alpha=0.08, zorder=0)
    ax_p.axhline(threshold, color=SERIES_COLORS[2], linewidth=1.2,
                 label=f"BO constraint: ΔP ≥ target × {threshold:.3f}")
    ax_p.axhline(1.0, color=INK, linewidth=0.8, linestyle=":", label="target = linear-rise ΔP")
    ax_p.set_ylim(P_FLOOR - 0.002, max(1.008, history["delta_P_NORM"].max() + 0.001))
    ax_p.set(xlabel="measurement number (after warm-up)", ylabel="ΔP / linear-rise ΔP")
    handles, _ = ax_p.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="v", linestyle="none", color=MUTED, label=f"below {P_FLOOR} (off scale)"))
    ax_p.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)
    for axis in (ax_work, ax_p):
        axis.set_xlim(-2, last + 2)
        axis.grid(True, which="major", alpha=0.25)


def plot_warmup(axis, warmup):
    axis.set_title("① Warm-up: repeated linear rise until the device settles", fontsize=10, loc="left")
    if warmup is None:
        axis.text(0.5, 0.5, "no warm-up table saved", ha="center", transform=axis.transAxes)
        return
    settled = warmup.tail(20)
    for column, color, label in (("Work", INK, "Work"), ("delta_P", MUTED, "ΔP")):
        axis.plot(warmup["warmup"], 100 * (warmup[column] / settled[column].mean() - 1),
                  color=color, linewidth=1.5, marker="o", markersize=2.5, label=label)
    axis.axhline(0, color=INK, linewidth=0.8, linestyle=":")
    axis.set(xlabel="warm-up repeat", ylabel="change vs settled value (%)\n(mean of last 20 repeats)")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.25)


def plot_drift(axis, history, window):
    axis.set_title("④ Drift reference: linear rises measured through the run", fontsize=10, loc="left")
    linear = history[history["phase"].isin(LINEAR_PHASES)]
    for column, color, label in (("Work", INK, "Work"), ("delta_P", MUTED, "ΔP")):
        values = linear[column].to_numpy(dtype=float)
        mean = values.mean()
        axis.scatter(linear.index, 100 * (values / mean - 1), s=12, color=color, zorder=3)
        axis.plot(linear.index, 100 * (_centered_mean(values, window) / mean - 1), color=color,
                  linewidth=1.8, label=f"{label}: points = single linear rise, line = mean of {window} (reference)")
    axis.set(xlabel="measurement number\n(each measurement is divided by this reference at the time it was measured)",
             ylabel="change vs run mean (%)")
    axis.legend(fontsize=7.5, loc="upper left")
    axis.grid(True, alpha=0.25)


def plot_tradeoff(axis, history, threshold, best, best_color):
    axis.set_title("Trade-off: less Work ⇄ less switching", fontsize=10, loc="left")
    for rows, label, kwargs in _groups(history, threshold):
        if rows["phase"].isin(LINEAR_PHASES).all():
            continue
        x, y = rows["delta_P_NORM"].to_numpy(), rows["Work_NORM"].to_numpy()
        low = x < P_FLOOR_TRADEOFF
        axis.scatter(x[~low], y[~low], label=label, zorder=3, **kwargs)
        axis.scatter(np.full(low.sum(), P_FLOOR_TRADEOFF), y[low], zorder=3, **{**kwargs, "marker": "<"})
    axis.axvspan(threshold, 1.02, color=SERIES_COLORS[2], alpha=0.08, zorder=0)
    axis.axvline(threshold, color=SERIES_COLORS[2], linewidth=1.2)
    axis.axvline(1.0, color=INK, linewidth=0.8, linestyle=":")
    axis.scatter([1.0], [1.0], marker="s", s=60, color=INK, zorder=5, label="linear rise")
    axis.scatter([best["delta_P_NORM_mean"]], [best["Work_NORM_mean"]], marker="*", s=260,
                 color=best_color, edgecolors="white", linewidths=1, zorder=6, label="confirmed best")
    _work_axis(axis)
    axis.set_xlim(P_FLOOR_TRADEOFF - 0.002, max(1.008, history["delta_P_NORM"].max() + 0.002))
    axis.set(xlabel=f"ΔP / linear-rise ΔP   (◀ = below {P_FLOOR_TRADEOFF}, off scale)",
             ylabel="Work / linear-rise Work")
    axis.text(threshold + 0.0005, 0.97, "allowed", transform=axis.get_xaxis_transform(),
              fontsize=8, color=SERIES_COLORS[2], va="top")
    axis.legend(fontsize=7, loc="upper left")
    axis.grid(True, which="major", alpha=0.25)


def plot_shapes(axis, history, config, threshold, best, best_color):
    waveform = config["waveform"]
    inputs = [column for column in history.columns if column.startswith("V") and column[1:].isdigit()]
    s = np.linspace(0.0, 1.0, 300)
    t_ns = s * waveform["rise_time_s"] * 1e9

    def shape(values):
        return rise_voltage(np.asarray(values, dtype=float), s, waveform["rise_basis"])

    axis.set_title("Rise shapes: what was tried and what was kept", fontsize=10, loc="left")
    for _, row in history[history["phase"] == "random initial"].iterrows():
        axis.plot(t_ns, shape(row[inputs]), color=RANDOM, linewidth=0.6, alpha=0.35)
    searched = history[history["phase"].isin(("random initial", "BO optimization"))]
    ok = searched[searched["P_lower_NORM"] >= threshold]
    improved = ok[ok["Work_NORM"] < ok["Work_NORM"].cummin().shift(fill_value=np.inf)]
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(improved)))
    for (evaluation, row), color in zip(improved.iterrows(), shades):
        axis.plot(t_ns, shape(row[inputs]), color=color, linewidth=1.2)
    axis.plot(t_ns, shape(history.iloc[0][inputs]), color=INK, linewidth=2, linestyle="--")
    axis.plot(t_ns, shape(best[inputs]), color=best_color, linewidth=2.8)
    for bound in (waveform["minimum_voltage_v"], waveform["maximum_voltage_v"]):
        axis.axhline(bound, color=MUTED, linewidth=0.8, linestyle=":")
    axis.axhline(waveform["amplitude_v"], color=INK, linewidth=0.6, alpha=0.4)
    axis.text(t_ns[-1], waveform["maximum_voltage_v"], "BO upper bound ", ha="right", va="bottom",
              fontsize=7.5, color=MUTED)
    axis.text(t_ns[-1], waveform["amplitude_v"], "flat top ", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    axis.legend(handles=[Line2D([], [], color=INK, linestyle="--", linewidth=2, label="linear rise"),
                         Line2D([], [], color=RANDOM, linewidth=1, label="random initial shapes"),
                         Line2D([], [], color=plt.cm.Blues(0.6), linewidth=1.2,
                                label=f"new best during the run ({len(improved)}×, light → dark = later)"),
                         Line2D([], [], color=best_color, linewidth=2.8, label="confirmed best")],
                fontsize=7.5, loc="upper left")
    axis.set(xlabel="time in the rise (ns)", ylabel="commanded voltage (V)")
    axis.grid(True, alpha=0.25)


def plot_measured(ax_current, ax_vp, history, run_dir, area, best, best_color):
    def mean_rise(rows):
        return _mean_rise(";".join(str(run_dir / name) for name in rows["rise_file"]), area)

    linear = mean_rise(history[history["phase"] == "confirmation reference"])
    chosen = mean_rise(history[(history["phase"] == "confirmation") & (history["candidate"] == best["candidate"])])
    ax_current.set_title("Measured switching current (confirmation means)", fontsize=10, loc="left")
    ax_vp.set_title("Work / area = ∫ V dP  (area under each curve)", fontsize=10, loc="left")
    for (t, voltage, current, polarization), color, style, label in (
            (linear, INK, "--", "linear rise"), (chosen, best_color, "-", "confirmed best")):
        ax_current.plot(t * 1e9, current * 1e6, color=color, linestyle=style, linewidth=2, label=label)
        work_per_area = np.sum(0.5 * (voltage[1:] + voltage[:-1]) * np.diff(polarization))
        ax_vp.plot(polarization, voltage, color=color, linestyle=style, linewidth=2,
                   label=f"{label}: {work_per_area:.4f} J/m², ΔP = {polarization[-1]:.4f} C/m²")
        ax_vp.fill_between(polarization, voltage, color=color, alpha=0.12)
    ax_current.set(xlabel="time in the rise (ns)", ylabel="I_P − I_U (µA)")
    ax_current.legend(fontsize=8, loc="upper left")
    ax_vp.set(xlabel="switched polarization P(t) = ∫(I_P − I_U)dt / area (C/m²)", ylabel="voltage (V)")
    ax_vp.legend(fontsize=7.5, loc="upper left")
    ax_vp.set_ylim(bottom=0)
    for axis in (ax_current, ax_vp):
        axis.grid(True, alpha=0.25)
        axis.text(0.99, 0.02, "20 ns cut at each window end", transform=axis.transAxes, ha="right",
                  fontsize=7, color=MUTED)


def plot_confirmation(axis, history, summary):
    axis.set_title("⑤ Confirmation: each candidate re-measured in 10 shuffled rounds, "
                   "one linear reference per round", fontsize=10, loc="left")
    confirm = history[history["phase"].isin(("confirmation", "confirmation reference"))]
    rounds = (confirm["phase"] == "confirmation reference").cumsum()
    references = confirm[confirm["phase"] == "confirmation reference"]
    axis.plot(rounds[references.index], references["Work_NORM"], color=INK, marker="s", markersize=5,
              linestyle="--", linewidth=1, label="linear reference (last one closes round 10)")
    for _, row in summary.sort_values("candidate").iterrows():
        candidate = int(row["candidate"])
        color = CANDIDATE_COLORS[candidate - 1]
        repeats = confirm[confirm["candidate"] == candidate]
        axis.plot(rounds[repeats.index], repeats["Work_NORM"], color=color, marker="D", markersize=5,
                  linewidth=1.5, label=f"candidate {candidate} (rank {int(row['rank'])}, from measurement "
                                       f"{int(row['source_evaluation'])}): {row['Work_NORM_mean']:.3f} ± "
                                       f"{row['Work_NORM_std']:.3f}, ΔP {row['delta_P_NORM_mean']:.4f}")
        axis.scatter([0], [row["bo_Work_NORM"]], marker="*", s=180, color=color, edgecolors="white", zorder=4)
    axis.set_xticks(range(0, int(rounds.max()) + 1))
    axis.set_xticklabels(["BO\nsingle"] + [str(r) for r in range(1, int(rounds.max()) + 1)])
    axis.set(xlabel="confirmation round (★ = the single BO measurement that nominated it)",
             ylabel="Work / linear-rise Work")
    axis.legend(fontsize=8, loc="center right", framealpha=0.95)
    axis.grid(True, alpha=0.25)


def plot_text(axis, history, config, summary, threshold, warmup):
    axis.axis("off")
    waveform, bo = config["waveform"], config["bo"]
    best = summary.iloc[0]
    n_bo = int((history["phase"] == "BO optimization").sum())
    n_ok = int(((history["phase"] == "BO optimization") & (history["P_lower_NORM"] >= threshold)).sum())
    lines = [
        ("Settings", True),
        (f"rise τ = {waveform['rise_time_s'] * 1e6:.3g} µs, {bo['rise_points']} {waveform['rise_basis']} "
         f"controls in [{waveform['minimum_voltage_v']:g}, {waveform['maximum_voltage_v']:g}] V", False),
        (f"amplitude ±{waveform['amplitude_v']:g} V, gaps {waveform['gap1_s'] * 1e6:g} / "
         f"{waveform['gap2_s'] * 1e6:g} µs", False),
        ("Objective: minimize Work = ∫ V (I_P − I_U) dt over the rise", False),
        (f"Constraint: ΔP ≥ target × (1 {bo['p_lower_relative_tolerance']:+.3f})", False),
        ("", False),
        ("What each phase does", True),
        (f"① {len(warmup) if warmup is not None else 0} linear rises, not used: settle the device", False),
        (f"② {bo['linear_repeats']} linear rises set the ΔP target; {bo['initial_random_measurements']} "
         f"random shapes seed the GP", False),
        (f"③ {n_bo} BO steps: GP models Work and ΔP, proposes the shape", False),
        ("    with the highest expected improvement that likely meets ΔP", False),
        (f"④ linear reference every {bo['reference_interval']} steps; each point is normalized", False),
        (f"    to the mean of {bo['reference_window']} nearby linear rises (removes drift)", False),
        (f"⑤ top {bo['confirm_top']} distinct shapes × {bo['confirm_repeats']} rounds, ranked by mean Work", False),
        ("", False),
        ("Result", True),
        (f"{n_ok} of {n_bo} BO steps met the ΔP bound", False),
        (f"confirmed best: candidate {int(best['candidate'])}, Work reduction "
         f"{best['work_reduction_pct']:.1f} ± {100 * best['Work_NORM_sem']:.2f} %", False),
        (f"(mean ± s.e.m. of {int(best['n'])}), at ΔP = {best['delta_P_NORM_mean']:.4f} × target", False),
    ]
    y = 1.0
    for text, heading in lines:
        axis.text(0.0, y, text, transform=axis.transAxes, va="top", fontsize=9.5 if heading else 8.6,
                  fontweight="bold" if heading else "normal", color=INK)
        y -= 0.062 if heading else 0.054


def plot_overview(run_dir: Path) -> Path:
    history_path, history, config, summary, warmup = load(run_dir)
    threshold = 1.0 + config["bo"]["p_lower_relative_tolerance"]
    best = summary.iloc[0]
    best_color = CANDIDATE_COLORS[int(best["candidate"]) - 1]
    area = config["device"]["electrode_area_m2"]

    figure = plt.figure(figsize=(19, 21))
    grid = figure.add_gridspec(5, 3, height_ratios=[1.15, 0.8, 1, 1, 1], hspace=0.45, wspace=0.24,
                               top=0.925, bottom=0.04, left=0.06, right=0.98)
    ax_work = figure.add_subplot(grid[0, :])
    ax_p = figure.add_subplot(grid[1, :], sharex=ax_work)
    plot_timeline(ax_work, ax_p, history, threshold, best_color)
    plot_warmup(figure.add_subplot(grid[2, 0]), warmup)
    plot_drift(figure.add_subplot(grid[2, 1]), history, config["bo"]["reference_window"])
    plot_tradeoff(figure.add_subplot(grid[2, 2]), history, threshold, best, best_color)
    plot_shapes(figure.add_subplot(grid[3, 0]), history, config, threshold, best, best_color)
    plot_measured(figure.add_subplot(grid[3, 1]), figure.add_subplot(grid[3, 2]), history, run_dir,
                  area, best, best_color)
    plot_confirmation(figure.add_subplot(grid[4, :2]), history, summary)
    plot_text(figure.add_subplot(grid[4, 2]), history, config, summary, threshold, warmup)
    figure.suptitle(f"One BO run, step by step — {run_dir.name}\n"
                    "Goal: the rise shape that switches at least as much polarization as the linear rise "
                    "with the least Work", fontsize=14, y=0.985)
    output_path = history_path.with_name(f"{history_path.stem}_overview.png")
    _save(figure, output_path)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-folder", type=Path, required=True, help="one run folder (BO_pund_rise_...)")
    args = parser.parse_args()
    print(f"Overview: {plot_overview(args.data_folder)}")


if __name__ == "__main__":
    main()
