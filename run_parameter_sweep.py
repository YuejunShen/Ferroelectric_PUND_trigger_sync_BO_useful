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

from config import ExperimentConfig
from plot_sweep_convergence import plot_convergence
from run_experiment import run_once
from storage import RUN_LOG_PATH

# Dotted-path config overrides. A list means "sweep over these values"; a
# bare (non-list) value overrides that field but does not multiply the
# number of runs.
SWEEP = {
    "bo.rise_points": [14,12,10,8,6,4],
}
# How many times to repeat *each* combination of the values above (e.g. to
# average out BO's random_seed / measurement noise for a convergence test).
REPEATS = 4
# Every run in this sweep is grouped under one subfolder of today's usual
# data_mother_path (config.py) -- e.g. D:\Ferroelectric_BTO_optimization\
# 20260907\rise_points_scan\ -- instead of scattered across separate
# per-run folders, so the whole sweep and its convergence plot live together.
SWEEP_NAME = "rise_points_scan"
SWEEP_FOLDER = ExperimentConfig().data_mother_path / SWEEP_NAME


def _set_nested(cfg, dotted_path: str, value):
    """dataclasses.replace(), but through a dotted path into nested (frozen)
    dataclasses, e.g. "bo.rise_points" -> cfg.bo.rise_points."""
    head, _, rest = dotted_path.partition(".")
    if not rest:
        return dataclasses.replace(cfg, **{head: value})
    child = _set_nested(getattr(cfg, head), rest, value)
    return dataclasses.replace(cfg, **{head: child})


def _expand(sweep: dict) -> list[dict]:
    """Cartesian product of every list-valued entry; scalar entries pass through unchanged."""
    keys = list(sweep.keys())
    axes = [values if isinstance(values, list) else [values] for values in sweep.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes)]


def _label(combo: dict, repeat: int) -> str:
    parts = [f"{dotted_path.split('.')[-1]}{value}" for dotted_path, value in combo.items()]
    if REPEATS > 1:
        parts.append(f"rep{repeat}")
    return "_".join(parts)


def build_configs():
    # Repeats are the OUTER loop: every combo runs once before any repeats
    # start, so repeat 1 sweeps the whole parameter range, then repeat 2
    # sweeps it again, etc. -- rather than exhausting all repeats of one
    # combo before moving to the next.
    for repeat in range(1, REPEATS + 1):
        for combo in _expand(SWEEP):
            cfg = ExperimentConfig()
            # Group every run in this sweep under SWEEP_FOLDER instead of the
            # day's usual data_mother_path -- see output_directory in config.py.
            cfg = dataclasses.replace(cfg, data_mother_path=SWEEP_FOLDER)
            for dotted_path, value in combo.items():
                cfg = _set_nested(cfg, dotted_path, value)
            cfg = dataclasses.replace(cfg, sweep_tag=_label(combo, repeat))
            yield cfg, combo, repeat


def _default_x_column() -> str | None:
    """The swept field to plot convergence against, if unambiguous (exactly
    one key in SWEEP actually varies); None if there's more than one, or none."""
    swept_keys = [key for key, values in SWEEP.items() if isinstance(values, list) and len(values) > 1]
    return swept_keys[0].split(".")[-1] if len(swept_keys) == 1 else None


def main() -> None:
    # Deliberately NOT list(build_configs()) -- that would construct every
    # ExperimentConfig() (and freeze every run_timestamp) up front, before
    # any run actually starts. Iterating the generator directly instead means
    # each cfg (and its run_timestamp) is created right before that run begins.
    total = len(_expand(SWEEP)) * REPEATS
    print(f"Sweep: {total} run(s) total. Output folder: {SWEEP_FOLDER}")
    for index, (cfg, combo, repeat) in enumerate(build_configs(), start=1):
        print(f"\n=== Run {index}/{total}: {cfg.sweep_tag} ===")
        run_once(cfg, sweep_label=cfg.sweep_tag, sweep_overrides=combo,
                 repeat_index=repeat, block_on_finish=False)

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
