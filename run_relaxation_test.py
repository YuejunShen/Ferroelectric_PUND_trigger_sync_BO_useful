"""Relaxation test: delta_P vs the gaps of the PUND waveform (no BO). Edit
the settings below, then run this file.

gap1 (reset -> P) and gap2 (P -> U) are always set to the same value. After
the usual linear warm-up (ExperimentConfig.warmup_repeats, at config.py's
gaps, saved to warmup/ if save_warmup), every gap in GAPS_S is measured once
per round, in a new random order each round (so slow drift doesn't map onto
the gap axis), for ROUNDS rounds. Each gap is rounded to whole AWG samples --
the rise windows use the nominal gap, so a fractional sample would misalign
the U window against the U pulse.

Writes relaxation_log_<timestamp>.csv (one row per measurement), the
trace/rise CSVs, and relaxation_delta_p_vs_gap.png (delta_P = integral of
I1-I2 / area vs gap, mean +/- std over rounds). Replot an existing log:

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

from config import ExperimentConfig
from instruments import ArbGenerator
from measurement import apply_rise_and_measure
from run_drift_test import linear_ramp
from storage import _write_traces, save_run_config, save_warmup_measurement
from waveform import build_pund_waveform

# gap1 = gap2 values to test. Upper limit: the whole waveform must fit in
# maximum_arb_points at sample_rate_hz (1e6 points at 100 MSa/s = 10 ms), and
# the scope captures it at scope_time_step_s (1 ns -> 1e5 points per 100 us).
GAPS_S = np.logspace(np.log10(10e-9), np.log10(1e-3), num=40)
ROUNDS = 5
# Rise applied in every measurement: None = the linear rise, or a list of
# rise_points control voltages (e.g. a confirmed best protocol's V0..Vn).
PROTOCOL_V: list[float] | None = None
TEST_NAME = "relaxation_test"


def _with_gap(cfg: ExperimentConfig, gap_s: float) -> ExperimentConfig:
    return dataclasses.replace(cfg, waveform=dataclasses.replace(cfg.waveform, gap1_s=gap_s, gap2_s=gap_s))


def run_relaxation(cfg: ExperimentConfig) -> Path:
    sample_rate = cfg.waveform.sample_rate_hz
    gaps = sorted({round(gap * sample_rate) / sample_rate for gap in GAPS_S})
    voltages = linear_ramp(cfg) if PROTOCOL_V is None else np.asarray(PROTOCOL_V, dtype=float)
    for gap in gaps:  # fail now, not hours in, if a gap makes the waveform too long
        build_pund_waveform(voltages, _with_gap(cfg, gap).waveform)
    save_run_config(cfg)
    log_path = cfg.output_directory / f"relaxation_log_{cfg.run_timestamp}.csv"
    print(f"Relaxation test: gaps {[f'{gap * 1e9:g} ns' for gap in gaps]}, {ROUNDS} round(s), "
          f"rise {voltages}. Log: {log_path}")
    rng = np.random.default_rng(cfg.bo.random_seed)
    try:
        for warmup in range(1, cfg.warmup_repeats + 1):
            measurement = apply_rise_and_measure(linear_ramp(cfg), cfg)
            if cfg.save_warmup:
                save_warmup_measurement(measurement, warmup, cfg)
            print(f"Warm-up {warmup}/{cfg.warmup_repeats}: delta_P={measurement.delta_p_c_per_m2:.6g} C/m^2")
        with log_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["round", "gap_s", "logged_at", "Work", "delta_P", "trace_file", "rise_file"])
            for round_index in range(1, ROUNDS + 1):
                for gap in rng.permutation(gaps):
                    measurement = apply_rise_and_measure(voltages, _with_gap(cfg, gap))
                    trace_path, rise_path = _write_traces(
                        measurement, cfg.output_directory,
                        f"{cfg.run_timestamp}_gap{gap * 1e9:.0f}ns_round{round_index:02d}")
                    writer.writerow([round_index, gap, dt.datetime.now().isoformat(timespec="seconds"),
                                     measurement.work_j, measurement.delta_p_c_per_m2,
                                     trace_path.name, rise_path.name])
                    handle.flush()
                    print(f"Round {round_index}/{ROUNDS}, gap {gap * 1e9:g} ns: "
                          f"delta_P={measurement.delta_p_c_per_m2:.6g} C/m^2, Work={measurement.work_j:.6g} J")
    except KeyboardInterrupt:
        print("\nRelaxation test stopped by user (Ctrl+C).")
    finally:
        ArbGenerator(cfg).output(False)
        print("AWG output confirmed off.")
    return log_path


def plot_relaxation(log_path: Path) -> Path:
    """delta_P vs gap: every measurement (hollow) and mean +/- std over rounds."""
    log = pd.read_csv(log_path)
    stats = log.groupby("gap_s")["delta_P"].agg(["mean", "std"]).reset_index()
    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.scatter(log["gap_s"] * 1e6, log["delta_P"], s=30, facecolors="none", edgecolors="#2a78d6",
                 linewidths=1.0, label="Each measurement")
    axis.errorbar(stats["gap_s"] * 1e6, stats["mean"], yerr=stats["std"].fillna(0.0), color="#2a78d6",
                  marker="o", markersize=7, linewidth=2, capsize=4, label="Mean ± std of rounds")
    axis.set_xscale("log")
    axis.set(xlabel="Gap reset→P = P→U (us)", ylabel="ΔP = ∫(I1-I2)dt / area (C/m²)",
             title="PUND relaxation: ΔP vs gap")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    figure.tight_layout()
    output_path = log_path.with_name("relaxation_delta_p_vs_gap.png")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replot", type=Path, help="existing relaxation_log_*.csv to plot (no measurement)")
    args = parser.parse_args()
    if args.replot:
        print(f"Plot: {plot_relaxation(args.replot)}")
        return
    cfg = ExperimentConfig()
    cfg = dataclasses.replace(cfg, data_mother_path=cfg.data_mother_path / TEST_NAME, sweep_tag="relaxation")
    log_path = run_relaxation(cfg)
    if log_path.exists():
        print(f"Plot: {plot_relaxation(log_path)}")


if __name__ == "__main__":
    main()
