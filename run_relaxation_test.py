"""Relaxation test: delta_P vs the gaps of the PUND waveform, at one or
more amplitudes (no BO). Edit the settings below, then run this file.

gap1 (reset -> P) and gap2 (P -> U) are always set to the same value. Each
of ROUNDS rounds steps through every amplitude in AMPLITUDES_V in turn
(reset, flat top and linear-rise end together, as waveform.amplitude_v) --
e.g. 1.0, 1.2, 1.8 V, then 1.0, 1.2, 1.8 V again -- so slow drift is spread
over all amplitudes instead of mapping onto the amplitude axis. Each
amplitude block: the usual linear warm-up at that amplitude
(ExperimentConfig.warmup_repeats, plus initial_warmup_repeats in the very
first block; at config.py's gaps, saved to warmup/ if save_warmup), then every gap in GAPS_S once, in a new random order each time
(so drift doesn't map onto the gap axis either). Each gap is rounded to whole
AWG samples -- the rise windows use the nominal gap, so a fractional sample
would misalign the U window against the U pulse.

Writes relaxation_log_<timestamp>.csv (one row per measurement), the
trace/rise CSVs, and relaxation_delta_p_vs_gap_{logx,logy,loglog}.png
(delta_P = integral of I1-I2 / area vs gap, mean +/- std over rounds, one
curve per amplitude; see PLOT_SCALES). Replot an existing log:

    python run_relaxation_test.py --replot D:\\...\\relaxation_log_<timestamp>.csv
"""

import argparse
import csv
import dataclasses
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from config import ExperimentConfig
from instruments import ArbGenerator
from measurement import apply_rise_and_measure
from plot_sweep_results import SERIES_COLORS, _save
from run_drift_test import linear_ramp
from storage import _write_traces, save_run_config, save_warmup_measurement
from waveform import build_pund_waveform

# gap1 = gap2 values to test. Upper limit: the whole waveform must fit in
# maximum_arb_points at sample_rate_hz (1e6 points at 100 MSa/s = 10 ms), and
# the scope captures it at scope_time_step_s (1 ns -> 1e5 points per 100 us).
GAPS_S = np.logspace(np.log10(10e-9), np.log10(1e-3), num=50)
ROUNDS = 3
# Amplitudes to scan, in this order within every round (each block: warm-up + all gaps once).
AMPLITUDES_V: list[float] = [1.0,1.2,1.8]
# Rise applied in every measurement: None = the linear rise (to each
# amplitude), or a list of rise_points control voltages (e.g. a confirmed
# best protocol's V0..Vn), applied unchanged at every amplitude.
PROTOCOL_V: list[float] | None = None
TEST_NAME = "relaxation_test"


def _with(cfg: ExperimentConfig, amplitude_v: float, gap_s: float | None = None) -> ExperimentConfig:
    """cfg at this amplitude and (unless None) with gap1 = gap2 = gap_s."""
    gaps = {} if gap_s is None else {"gap1_s": gap_s, "gap2_s": gap_s}
    return dataclasses.replace(cfg, waveform=dataclasses.replace(cfg.waveform, amplitude_v=amplitude_v, **gaps))


def _protocol(cfg: ExperimentConfig) -> np.ndarray:
    return linear_ramp(cfg) if PROTOCOL_V is None else np.asarray(PROTOCOL_V, dtype=float)


def run_relaxation(cfg: ExperimentConfig) -> Path:
    sample_rate = cfg.waveform.sample_rate_hz
    gaps = sorted({round(gap * sample_rate) / sample_rate for gap in GAPS_S})
    for amplitude in AMPLITUDES_V:  # fail now, not hours in, on a bad amplitude or too long a waveform
        at_amplitude = _with(cfg, amplitude)
        build_pund_waveform(linear_ramp(at_amplitude), at_amplitude.waveform)
        for gap in gaps:
            build_pund_waveform(_protocol(at_amplitude), _with(cfg, amplitude, gap).waveform)
    save_run_config(cfg)
    log_path = cfg.output_directory / f"relaxation_log_{cfg.run_timestamp}.csv"
    print(f"Relaxation test: amplitudes {AMPLITUDES_V} V, gaps {[f'{gap * 1e9:g} ns' for gap in gaps]}, "
          f"{ROUNDS} round(s). Log: {log_path}")
    rng = np.random.default_rng(cfg.bo.random_seed)
    warmup = 0  # numbers the warm-up files across all amplitudes
    try:
        with log_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["amplitude_v", "round", "gap_s", "logged_at", "Work", "delta_P",
                             "trace_file", "rise_file"])
            for round_index in range(1, ROUNDS + 1):
                for amplitude in AMPLITUDES_V:
                    at_amplitude = _with(cfg, amplitude)
                    # The very first block also gets the start-of-program warm-up.
                    repeats = cfg.warmup_repeats + (cfg.initial_warmup_repeats if warmup == 0 else 0)
                    for repeat in range(1, repeats + 1):
                        measurement = apply_rise_and_measure(linear_ramp(at_amplitude), at_amplitude)
                        warmup += 1
                        if cfg.save_warmup:
                            save_warmup_measurement(measurement, warmup, at_amplitude)
                        print(f"Round {round_index}/{ROUNDS}, {amplitude:g} V warm-up "
                              f"{repeat}/{repeats}: delta_P={measurement.delta_p_c_per_m2:.6g} C/m^2")
                    for gap in rng.permutation(gaps):
                        measurement = apply_rise_and_measure(_protocol(at_amplitude), _with(cfg, amplitude, gap))
                        trace_path, rise_path = _write_traces(
                            measurement, cfg.output_directory,
                            f"{cfg.run_timestamp}_amp{amplitude:g}V_gap{gap * 1e9:.0f}ns_round{round_index:02d}")
                        writer.writerow([amplitude, round_index, gap, dt.datetime.now().isoformat(timespec="seconds"),
                                         measurement.work_j, measurement.delta_p_c_per_m2,
                                         trace_path.name, rise_path.name])
                        handle.flush()
                        print(f"{amplitude:g} V, round {round_index}/{ROUNDS}, gap {gap * 1e9:g} ns: "
                              f"delta_P={measurement.delta_p_c_per_m2:.6g} C/m^2, Work={measurement.work_j:.6g} J")
    except KeyboardInterrupt:
        print("\nRelaxation test stopped by user (Ctrl+C).")
    finally:
        ArbGenerator(cfg).output(False)
        print("AWG output confirmed off.")
    return log_path


# Axis versions of the relaxation plot: (log x?, log y?, title note). Written
# as relaxation_delta_p_vs_gap_<name>.png. A straight line means, in logx: a
# logarithmic decay; logy: an exponential decay; loglog: a power law.
PLOT_SCALES = {"logx": (True, False, "log gap, linear ΔP"),
               "logy": (False, True, "linear gap, log ΔP"),
               "loglog": (True, True, "log gap, log ΔP")}


def plot_relaxation(log_path: Path, scale: str = "logx") -> Path:
    """delta_P vs gap: mean +/- std over rounds, one curve per amplitude (a
    log from before the amplitude scan has one curve), with the axes of
    PLOT_SCALES[scale]."""
    log_x, log_y, note = PLOT_SCALES[scale]
    log = pd.read_csv(log_path)
    amplitudes = sorted(log["amplitude_v"].unique()) if "amplitude_v" in log else [None]
    if len(amplitudes) <= len(SERIES_COLORS):
        colors = SERIES_COLORS[:len(amplitudes)]
    else:  # more amplitudes than categorical colors: an ordered map instead of repeating them
        colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(amplitudes)))
    figure, axis = plt.subplots(figsize=(7.5, 5))
    for amplitude, color in zip(amplitudes, colors):
        subset = log if amplitude is None else log[log["amplitude_v"] == amplitude]
        stats = subset.groupby("gap_s")["delta_P"].agg(["mean", "std"]).reset_index()
        prefix = "" if amplitude is None else f"{amplitude:g} V: "
        axis.errorbar(stats["gap_s"] * 1e6, stats["mean"], yerr=stats["std"].fillna(0.0), color=color,
                      marker="o", markersize=7, linewidth=2, capsize=4,
                      label=f"{prefix}mean ± std of {subset['round'].nunique()} rounds")
    if log_y:
        # delta_P spans less than a decade: plain-number ticks at 1, 1.5, 2, ... x 10^n.
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(subs=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        axis.yaxis.set_minor_formatter(NullFormatter())
    if log_x:
        axis.set_xscale("log")
    axis.set(xlabel="Gap reset→P = P→U (us)", ylabel="ΔP = ∫(I1-I2)dt / area (C/m²)",
             title=f"PUND relaxation: ΔP vs gap ({note})")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    figure.tight_layout()
    output_path = log_path.with_name(f"relaxation_delta_p_vs_gap_{scale}.png")
    _save(figure, output_path)
    plt.close(figure)
    return output_path


def plot_all(log_path: Path) -> None:
    for scale in PLOT_SCALES:
        print(f"Plot ({scale}): {plot_relaxation(log_path, scale)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replot", type=Path, help="existing relaxation_log_*.csv to plot (no measurement)")
    args = parser.parse_args()
    if args.replot:
        plot_all(args.replot)
        return
    cfg = ExperimentConfig()
    cfg = dataclasses.replace(cfg, data_mother_path=cfg.data_mother_path / TEST_NAME, sweep_tag="relaxation")
    log_path = run_relaxation(cfg)
    if log_path.exists():
        plot_all(log_path)


if __name__ == "__main__":
    main()
