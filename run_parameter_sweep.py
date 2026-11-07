"""Grid-sweep runner for hyperparameter convergence tests.

Edit SWEEP and REPEATS below, then run this file. Any ExperimentConfig field
can be targeted by its dotted path (walking through the nested dataclasses --
instruments/waveform/bo/device -- down to the field itself). Give a field a
*list* of values to sweep over it; every combination of all list-valued
entries in SWEEP is run (a Cartesian product), each repeated REPEATS times.

Example: sweep the rise-shape discretization over 6/8/10/12/14 knots, twice
each:

    SWEEP = {"bo.rise_points": [6, 8, 10, 12, 14]}
    REPEATS = 2

Each of the 10 resulting runs is fully independent (its own output folder,
history CSV, BO instance) and its output folder name records exactly which
values it used (e.g. "..._rise_points10_rep2"). Every run also appends one
row to D:\\Ferroelectric_BTO_optimization\\run_log.csv (see storage.append_run_log)
recording when it started and every parameter above.
"""

import dataclasses
import itertools
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from plot_sweep_convergence import plot_convergence
from plot_sweep_results import summarize_sweep
from run_experiment import run_once
from storage import RUN_LOG_PATH
from waveform import build_pund_waveform

# Dotted-path config overrides. A list means "sweep over these values"; a
# bare (non-list) value overrides that field but does not multiply the
# number of runs.
SWEEP = {
    # Fixed: 6 Bernstein points won the 10/30 rise_points x rise_basis scan.
    "bo.rise_points": 8,
    "waveform.rise_basis": "bernstein",
    # Rounded to 10 ns = whole samples at 100 MSa/s (build_pund_waveform refuses fractions).
    "waveform.rise_time_s": np.round(np.logspace(np.log10(1e-6), np.log10(10e-6), num=7), 8),
    # Sets the P/U flat top, the reset pulse and the linear rise's end together.
    "waveform.amplitude_v": [1.2],
}
# How many times to repeat *each* combination of the values above (e.g. to
# average out BO's random_seed / measurement noise for a convergence test).
REPEATS = 5
# Not None: each run's BO upper bound (maximum_voltage_v) is amplitude_v +
# this, unless "waveform.maximum_voltage_v" is given in SWEEP itself (0.1
# reproduces config.py's 1.2 V -> 1.3 V). None: config.py's value.
BO_MAX_ABOVE_POSITIVE_V: float | None = 0.25
# Every run in this sweep is grouped under one subfolder of today's usual
# data_mother_path (config.py) -- e.g. D:\Ferroelectric_BTO_optimization\
# 20260907\rise_points_scan\ -- instead of scattered across separate
# per-run folders, so the whole sweep and its convergence plot live together.
SWEEP_NAME = "rise_time_amplitude_scan"
# Resume a stopped sweep: start at this run number (as printed in
# "=== Run N/total ===", earlier runs are skipped) and write into the
# original sweep's folder instead of today's. For a new sweep: START_RUN = 1,
# RESUME_FOLDER = None.
START_RUN = 1
RESUME_FOLDER = None
# Run numbers to skip because their (complete) run folders are already in
# SWEEP_FOLDER, e.g. copied over from a stopped sweep with identical settings.
# For a new sweep: set().
SKIP_RUNS: set[int] = set()
SWEEP_FOLDER = RESUME_FOLDER or ExperimentConfig().data_mother_path / SWEEP_NAME


def _set_nested(cfg, dotted_path: str, value):
    """dataclasses.replace(), but through a dotted path into nested (frozen)
    dataclasses, e.g. "bo.rise_points" -> cfg.bo.rise_points."""
    head, _, rest = dotted_path.partition(".")
    if not rest:
        return dataclasses.replace(cfg, **{head: value})
    child = _set_nested(getattr(cfg, head), rest, value)
    return dataclasses.replace(cfg, **{head: child})


def _values(values) -> list:
    """A list, tuple or numpy array (e.g. np.logspace(...)) means "sweep over
    these values" (numpy scalars become plain Python numbers); anything else
    is one fixed value."""
    if isinstance(values, (list, tuple, np.ndarray)):
        return [value.item() if isinstance(value, np.generic) else value for value in values]
    return [values]


def _expand(sweep: dict) -> list[dict]:
    """Cartesian product of every list-valued entry; scalar entries pass through unchanged."""
    keys = list(sweep.keys())
    axes = [_values(values) for values in sweep.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


def _label(combo: dict, repeat: int) -> str:
    # Only the swept (list-valued) fields; fixed overrides are in run_config.
    # 4 significant digits, so np.logspace values give short folder names.
    parts = [f"{dotted_path.split('.')[-1]}{value:.4g}" if isinstance(value, float)
             else f"{dotted_path.split('.')[-1]}{value}"
             for dotted_path, value in combo.items() if len(_values(SWEEP[dotted_path])) > 1]
    if REPEATS > 1:
        parts.append(f"rep{repeat}")
    return "_".join(parts)


def _build(combo: dict, repeat: int) -> ExperimentConfig:
    cfg = ExperimentConfig()
    # Group every run in this sweep under SWEEP_FOLDER instead of the
    # day's usual data_mother_path -- see output_directory in config.py.
    cfg = dataclasses.replace(cfg, data_mother_path=SWEEP_FOLDER)
    for dotted_path, value in combo.items():
        cfg = _set_nested(cfg, dotted_path, value)
    if BO_MAX_ABOVE_POSITIVE_V is not None and "waveform.maximum_voltage_v" not in combo:
        cfg = _set_nested(cfg, "waveform.maximum_voltage_v",
                          cfg.waveform.amplitude_v + BO_MAX_ABOVE_POSITIVE_V)
    return dataclasses.replace(cfg, sweep_tag=_label(combo, repeat))


def build_configs():
    # Repeats are the OUTER loop: every combo runs once before any repeats
    # start, so repeat 1 sweeps the whole parameter range, then repeat 2
    # sweeps it again, etc. -- rather than exhausting all repeats of one
    # combo before moving to the next.
    for repeat in range(1, REPEATS + 1):
        for combo in _expand(SWEEP):
            yield _build(combo, repeat), combo, repeat


def _validate_all() -> None:
    """Build every run's config and linear-rise waveform up front, so an
    out-of-range value fails now rather than hours into the sweep. (These
    configs are discarded -- each run's real one is built right before it
    starts, so its run_timestamp reflects when it actually ran.)"""
    for combo in _expand(SWEEP):
        cfg = _build(combo, repeat=1)
        linear = np.linspace(cfg.waveform.linear_minimum_voltage_v,
                             cfg.waveform.linear_maximum_voltage_v, cfg.bo.rise_points)
        build_pund_waveform(linear, cfg.waveform)


def _default_x_column() -> str | None:
    """The swept field to plot convergence against, if unambiguous (exactly
    one key in SWEEP actually varies); None if there's more than one, or none.
    rise_basis is not counted -- the plot already draws one curve per basis."""
    swept_keys = [key for key, values in SWEEP.items()
                  if len(_values(values)) > 1 and key != "waveform.rise_basis"]
    return swept_keys[0].split(".")[-1] if len(swept_keys) == 1 else None


def main() -> None:
    # Deliberately NOT list(build_configs()) -- that would construct every
    # ExperimentConfig() (and freeze every run_timestamp) up front, before
    # any run actually starts. Iterating the generator directly instead means
    # each cfg (and its run_timestamp) is created right before that run begins.
    _validate_all()
    total = len(_expand(SWEEP)) * REPEATS
    print(f"Sweep: {total} run(s) total. Output folder: {SWEEP_FOLDER}")
    first = True  # the first run actually started gets the start-of-program warm-up
    for index, (cfg, combo, repeat) in enumerate(build_configs(), start=1):
        if index < START_RUN or index in SKIP_RUNS:
            continue
        print(f"\n=== Run {index}/{total}: {cfg.sweep_tag} ===")
        stopped = run_once(cfg, sweep_label=cfg.sweep_tag, sweep_overrides=combo,
                            repeat_index=repeat, block_on_finish=False, initial_warmup=first)
        first = False
        if stopped:
            print(f"\nStop requested during run {index}/{total} -- ending sweep "
                  "early instead of starting the next run.")
            break

    try:
        # Each setting's best run (confirmed protocol) + tables and plots.
        summarize_sweep(SWEEP_FOLDER)
    except Exception as exc:
        print(f"\nCould not summarize the sweep: {exc}")

    x_column = _default_x_column()
    if x_column is None:
        print("\nMore than one swept parameter (or none) -- run plot_sweep_convergence.py "
              f"manually, e.g.:\n  python plot_sweep_convergence.py --sweep-folder \"{SWEEP_FOLDER}\" "
              "--x-column <field_name>")
        return
    try:
        plot_path = plot_convergence(RUN_LOG_PATH, x_column=x_column, sweep_folder=SWEEP_FOLDER)
        print(f"\nConvergence plot saved to {plot_path}")
    except Exception as exc:
        print(f"\nCould not generate convergence plot: {exc}")


if __name__ == "__main__":
    main()
