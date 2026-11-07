"""Publication figure of a rise-time sweep's results (one confirmed best protocol per run).

Same journal style as plot_process_figure.py (180 mm wide, 7 pt Arial, SI units).
Reads <sweep folder>/sweep_all_runs.csv (written by plot_sweep_results.py).
a: the best optimized rise per tau (largest Work saving of its repeats) vs t/tau;
b, c: its measured switching current density (log axis) and P(t) / Delta_P vs t/tau,
   with the linear rise's;
d: W vs tau, mean +/- s.d. over the repeats, linear and optimized, power-law fits
   (alpha +/- its standard error);
e: Delta_P vs tau, mean +/- s.d. over the repeats, linear and optimized;
f: Work saved vs the linear rise, mean +/- s.d. over the repeats.

Also writes Fig_sweep_polynomial_coefficients: distributions of the exact
power-basis coefficients (orders 0..n) across confirmed best protocols,
grouped by rise time, with per-run and summary CSV tables.

    python plot_sweep_figure.py --sweep-folder <sweep folder> [--output-dir <folder>]
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from plot_process_figure import BEST, INK, MM, STYLE, _inputs
from plot_sweep_polynomial_coefficients import plot_polynomial_coefficients
from plot_sweep_results import _mean_rise
from waveform import rise_voltage

TAU_TICKS = (0.7, 1, 2, 5, 10)


def load_runs(sweep_folder: Path) -> pd.DataFrame:
    runs = pd.read_csv(sweep_folder / "sweep_all_runs.csv")
    runs = runs[runs["result"] == "confirmed"].copy()
    runs["tau_us"] = (runs["rise_time_s"] * 1e6).round(2)
    runs["repeat"] = runs["sweep_tag"].str.extract(r"rep(\d+)", expand=False).astype(int)
    return runs.sort_values(["tau_us", "repeat"]).reset_index(drop=True)


def _plain_log_ticks(axis, which, ticks, labels=None):
    """Log axis labelled with plain numbers (0.7, 1, 2, ...) instead of powers of ten."""
    target = axis.xaxis if which == "x" else axis.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FixedFormatter(labels or [f"{tick:g}" for tick in ticks]))
    target.set_minor_formatter(NullFormatter())


def _tau_colours(taus):
    """One green per rise time, light (short) to dark (long)."""
    return dict(zip(taus, plt.cm.Greens(np.linspace(0.4, 0.95, len(taus)))))


def panel_shapes(axis, runs, linear_end_v):
    s = np.linspace(0.0, 1.0, 300)
    inputs = _inputs(runs)
    taus = sorted(runs["tau_us"].unique())
    colours = _tau_colours(taus)
    for tau in taus:
        best = runs[runs["tau_us"] == tau].sort_values("work_reduction_pct").iloc[-1]
        axis.plot(s, rise_voltage(best[inputs].to_numpy(float), s, "bernstein"), color=colours[tau],
                  linewidth=1.0, label=f"{tau:g}")
    axis.plot(s, s * linear_end_v, color=INK, linestyle="--", linewidth=0.8, label="Linear")
    axis.set(xlabel="Normalized time in rise, $t/\\tau$", ylabel="$V$ (V)", xlim=(0, 1), ylim=(0, None))
    axis.legend(loc="upper left", ncol=2, title="$\\tau$ (µs)", title_fontsize=6, columnspacing=0.8,
                handlelength=1.2, alignment="left")


def panel_work(axis, runs):
    taus = np.array(sorted(runs["tau_us"].unique()))
    fit_tau = np.geomspace(taus.min(), taus.max(), 50)
    for column, colour, marker, name in (("linear_work_j", INK, "s", "Linear"),
                                         ("work_j", BEST, "o", "Optimized")):
        value = runs[column] * 1e12  # J -> pJ
        mean = value.groupby(runs["tau_us"]).mean().loc[taus]
        sd = value.groupby(runs["tau_us"]).std().loc[taus]
        # Least squares on log W vs log tau (the means); the alpha error is the slope's standard error.
        (alpha, log_a), covariance = np.polyfit(np.log(taus), np.log(mean), 1, cov=True)
        alpha_err = np.sqrt(covariance[0, 0])
        digits = max(2, -int(np.floor(np.log10(alpha_err))))  # the error to one significant figure
        axis.plot(fit_tau, np.exp(log_a) * fit_tau ** alpha, color=colour, linewidth=0.6, linestyle=":")
        axis.errorbar(taus, mean, yerr=sd, color=colour, marker=marker, markersize=3.5, linestyle="none",
                      elinewidth=0.6, capsize=1.5, zorder=3,
                      label=f"{name}, $\\alpha$ = {alpha:.{digits}f} ± {alpha_err:.{digits}f}".replace("-", "−"))
    axis.set_xscale("log")
    axis.set_yscale("log")
    _plain_log_ticks(axis, "x", TAU_TICKS)
    _plain_log_ticks(axis, "y", (10, 20, 30, 50))
    axis.set(xlabel="Rise time $\\tau$ (µs)", ylabel="$W$ (pJ)")
    axis.legend(loc="lower left", title="$W \\propto \\tau^{\\alpha}$", title_fontsize=6, alignment="left")


def panel_delta_p(axis, runs):
    """Delta_P/A vs tau, mean +/- s.d. over the repeats."""
    taus = np.array(sorted(runs["tau_us"].unique()))
    for column, colour, marker, name in (("linear_delta_P", INK, "s", "Linear"),
                                         ("delta_P", BEST, "o", "Optimized")):
        stats = runs.groupby("tau_us")[column].agg(["mean", "std"]).loc[taus]
        axis.errorbar(taus, stats["mean"], yerr=stats["std"], color=colour, marker=marker, markersize=3.5,
                      linestyle="none", elinewidth=0.6, capsize=1.5, label=name,
                      zorder=3 if name == "Optimized" else 2)
    axis.set_xscale("log")
    _plain_log_ticks(axis, "x", TAU_TICKS)
    axis.set(xlabel="Rise time $\\tau$ (µs)", ylabel="$\\Delta P$ (C m$^{-2}$)")
    axis.legend(loc="lower right")


def panel_reduction(axis, runs):
    taus = np.array(sorted(runs["tau_us"].unique()))
    stats = runs.groupby("tau_us")["work_reduction_pct"].agg(["mean", "std"]).loc[taus]
    axis.errorbar(taus, stats["mean"], yerr=stats["std"], color=BEST, marker="o", markersize=3.5, linestyle="none",
                  elinewidth=0.6, capsize=1.5, zorder=3)
    axis.set_xscale("log")
    _plain_log_ticks(axis, "x", TAU_TICKS)
    axis.set(xlabel="Rise time $\\tau$ (µs)", ylabel="Work saved vs linear (%)",
             ylim=(0, max(50, stats["mean"].max() + stats["std"].max() + 8)))


def panels_dynamics(ax_current, ax_p, runs):
    """Measured switching dynamics of each tau's best protocol (as in a) and its linear references,
    vs t / tau: the switching current density (log axis: it scales about as 1 / tau) and P(t) / Delta_P. TRIM_S
    is cut off each end of the rise window (the AWG edge, see plot_sweep_results.py)."""
    area_m2 = float(runs["electrode_area_m2"].iloc[0])
    taus = sorted(runs["tau_us"].unique())
    greens = _tau_colours(taus)
    greys = dict(zip(taus, plt.cm.Greys(np.linspace(0.35, 0.85, len(taus)))))
    for tau in taus:
        best = runs[runs["tau_us"] == tau].sort_values("work_reduction_pct").iloc[-1]
        for files, colours, style, width in ((best["linear_rise_files"], greys, "--", 0.6),
                                             (best["rise_files"], greens, "-", 0.9)):
            t, _, current, polarization = _mean_rise(files, area_m2)
            s = t / (tau * 1e-6)
            ax_current.plot(s, current / area_m2, color=colours[tau], linestyle=style, linewidth=width)
            ax_p.plot(s, polarization / polarization[-1], color=colours[tau], linestyle=style, linewidth=width)
    handles = [Line2D([], [], color=greens[taus[len(taus) // 2]], linewidth=0.9, label="Optimized"),
               Line2D([], [], color=greys[taus[len(taus) // 2]], linestyle="--", linewidth=0.6, label="Linear")]
    ax_current.set_yscale("log")
    ax_current.set(xlabel="Normalized time in rise, $t/\\tau$", ylabel="$(I_P - I_U)/A$ (A m$^{-2}$)",
                   xlim=(0, 1), ylim=(5e3, 3e6))
    ax_current.legend(handles=handles, loc="upper left")
    ax_p.set(xlabel="Normalized time in rise, $t/\\tau$", ylabel="$P(t)\\,/\\,\\Delta P$", xlim=(0, 1),
             ylim=(0, 1.05))
    ax_p.legend(handles=handles, loc="lower right")


def plot_sweep_figure(sweep_folder: Path, output_dir: Path | None = None) -> list[Path]:
    runs = load_runs(sweep_folder)
    linear_end_v = float(runs["amplitude_v"].iloc[0])
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(180 * MM, 115 * MM))
        grid = figure.add_gridspec(2, 3, hspace=0.42, wspace=0.42, left=0.065, right=0.985, top=0.96,
                                   bottom=0.085)
        axes = {"a": figure.add_subplot(grid[0, 0]), "b": figure.add_subplot(grid[0, 1]),
                "c": figure.add_subplot(grid[0, 2]), "d": figure.add_subplot(grid[1, 0]),
                "e": figure.add_subplot(grid[1, 1]), "f": figure.add_subplot(grid[1, 2])}
        panel_shapes(axes["a"], runs, linear_end_v)
        panels_dynamics(axes["b"], axes["c"], runs)
        panel_work(axes["d"], runs)
        panel_delta_p(axes["e"], runs)
        panel_reduction(axes["f"], runs)
        for letter, axis in axes.items():
            box = axis.get_position()
            figure.text(box.x0 - 0.055, box.y1 + 0.012, letter, fontsize=8, fontweight="bold", va="bottom",
                        ha="left")
        output_dir = output_dir or sweep_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / f"Fig_sweep_results.{suffix}" for suffix in ("pdf", "svg", "png")]
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
    paths.extend(plot_polynomial_coefficients(sweep_folder, output_dir))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-folder", type=Path, required=True, help="Folder with sweep_all_runs.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: the sweep folder")
    args = parser.parse_args()
    for path in plot_sweep_figure(args.sweep_folder, args.output_dir):
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
