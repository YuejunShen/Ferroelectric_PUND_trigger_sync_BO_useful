"""Run several full drift tests back to back, one per parameter setting, with
a fixed wait between settings. Edit SWEEP / WAIT_BETWEEN_SETTINGS_S below,
then run this file.

SWEEP keys are dotted paths, same as run_parameter_sweep.py:
  - any ExperimentConfig field, e.g. "waveform.positive_amplitude_v"
  - any drift-protocol field (run_drift_test.DriftProtocol) as "drift.<field>",
    e.g. "drift.intra_batch_delay_s"
A list means "sweep over these values" (every combination of all list-valued
entries is run); a bare value overrides that field for every run. Anything
not in SWEEP keeps its default from config.py / run_drift_test.py.

Example: drift tests at three flat-top/ramp-end voltages, 30 min apart:

    SWEEP = {
        "waveform.positive_amplitude_v": [0.4, 0.5, 0.6],
        "waveform.linear_maximum_voltage_v": 0.5,
    }
    WAIT_BETWEEN_SETTINGS_S = 1800

Each setting is one complete run_drift_once() (all batches + rests) in its
own subfolder of SWEEP_FOLDER, named after its values (e.g.
"..._positive_amplitude_v0.4"). With MATCH_RESET_TO_POSITIVE /
MATCH_RAMP_TO_POSITIVE, the reset pulse (-positive_amplitude_v) and the end of
the linear rise (+positive_amplitude_v) follow the swept amplitude, overriding
reset_voltage_v / linear_maximum_voltage_v from config.py.
The AWG output is off during the wait.
"Stop safely" or Ctrl+C (during a run or during the wait) ends the sweep.
"""

import dataclasses
import time

from config import ExperimentConfig
from instruments import ArbGenerator
from run_drift_test import DriftProtocol, linear_ramp, run_drift_once
from run_parameter_sweep import _expand, _set_nested
from waveform import build_pund_waveform

# Dotted-path overrides; see the module docstring.
SWEEP = {
    "waveform.positive_amplitude_v": [0.5,0.75,1],
}
# How many times to repeat each combination. Repeats are the outer loop: the
# whole list of settings runs once, then again, etc.
REPEATS = 1
# Wait after each setting's drift test finishes, before the next one starts
# (lets the device recover between settings). Not applied after the last one.
WAIT_BETWEEN_SETTINGS_S: float = 3600*2
# True: also wait WAIT_BETWEEN_SETTINGS_S before the FIRST setting (e.g. to let
# the device rest after whatever ran before this sweep). False: start at once.
WAIT_BEFORE_FIRST: bool = False
# True: each setting's reset pulse is -positive_amplitude_v (same magnitude as
# the P/U pulses), unless "waveform.reset_voltage_v" is given in SWEEP itself.
# False: reset_voltage_v keeps its config.py value.
MATCH_RESET_TO_POSITIVE: bool = True
# True: each setting's linear P/U rise ramps 0 -> positive_amplitude_v, so it
# meets the flat top with no jump, unless "waveform.linear_maximum_voltage_v"
# is given in SWEEP itself. False: the ramp ends at config.py's
# linear_maximum_voltage_v regardless of the amplitude.
MATCH_RAMP_TO_POSITIVE: bool = True
SWEEP_NAME = "drift_sweep"
SWEEP_FOLDER = ExperimentConfig().data_mother_path / SWEEP_NAME

DRIFT_PREFIX = "drift."


def _label(combo: dict, repeat: int) -> str:
    parts = [f"{dotted_path.split('.')[-1]}{value}" for dotted_path, value in combo.items()]
    if REPEATS > 1:
        parts.append(f"rep{repeat}")
    return "_".join(parts)


def _build(combo: dict, repeat: int) -> tuple[ExperimentConfig, DriftProtocol]:
    cfg = dataclasses.replace(ExperimentConfig(), data_mother_path=SWEEP_FOLDER)
    protocol = DriftProtocol()
    for dotted_path, value in combo.items():
        if dotted_path.startswith(DRIFT_PREFIX):
            protocol = _set_nested(protocol, dotted_path[len(DRIFT_PREFIX):], value)
        else:
            cfg = _set_nested(cfg, dotted_path, value)
    amplitude = cfg.waveform.positive_amplitude_v
    if MATCH_RESET_TO_POSITIVE and "waveform.reset_voltage_v" not in combo:
        cfg = _set_nested(cfg, "waveform.reset_voltage_v", -amplitude)
    if MATCH_RAMP_TO_POSITIVE and "waveform.linear_maximum_voltage_v" not in combo:
        cfg = _set_nested(cfg, "waveform.linear_maximum_voltage_v", amplitude)
    cfg = dataclasses.replace(cfg, sweep_tag=_label(combo, repeat))
    return cfg, protocol


def _validate_all(combos: list[dict]) -> None:
    """Build every setting's config and waveform up front, so an out-of-range
    value fails now rather than hours into the sweep. (The configs built here
    are discarded -- each run's real config is built right before it starts,
    so its run_timestamp reflects when it actually ran.)"""
    for combo in combos:
        cfg, _ = _build(combo, repeat=1)
        build_pund_waveform(linear_ramp(cfg), cfg.waveform)


def _output_off() -> None:
    """Turn the AWG output off and confirm it (ArbGenerator.output reads the
    state back). Raises if it can't be confirmed, so the sweep stops instead of
    sitting through an hours-long wait with the device possibly still driven."""
    ArbGenerator(ExperimentConfig()).output(False)
    print("AWG output confirmed off.")


def _wait(seconds: float) -> None:
    """Confirm the AWG output is off, then sleep for `seconds`, printing the
    remaining time each minute. Ctrl+C raises KeyboardInterrupt out of here to
    end the sweep."""
    _output_off()
    end = time.monotonic() + seconds
    while (remaining := end - time.monotonic()) > 0:
        print(f"  {remaining / 60:.1f} min until next setting (Ctrl+C to end sweep)")
        time.sleep(min(60.0, remaining))


def main() -> None:
    combos = _expand(SWEEP)
    _validate_all(combos)
    total = len(combos) * REPEATS
    print(f"Drift sweep: {total} setting(s), {WAIT_BETWEEN_SETTINGS_S}s wait between"
          + (" (and before the first)" if WAIT_BEFORE_FIRST else "")
          + f". Output folder: {SWEEP_FOLDER}")

    index = 0
    try:
        for repeat in range(1, REPEATS + 1):
            for combo in combos:
                index += 1
                if (index > 1 or WAIT_BEFORE_FIRST) and WAIT_BETWEEN_SETTINGS_S > 0:
                    print(f"\nWaiting {WAIT_BETWEEN_SETTINGS_S}s before setting {index}/{total}...")
                    _wait(WAIT_BETWEEN_SETTINGS_S)
                cfg, protocol = _build(combo, repeat)
                print(f"\n=== Setting {index}/{total}: {cfg.sweep_tag} ===")
                stopped = run_drift_once(cfg, protocol, block_on_finish=False)
                if stopped:
                    print(f"\nStopped during setting {index}/{total} -- ending sweep.")
                    return
    except KeyboardInterrupt:
        print(f"\nSweep stopped by user (Ctrl+C) after setting {index}/{total}.")
        return
    finally:
        # Covers every way out of the sweep -- including a crash inside
        # run_drift_once, which skips that function's own final shutdown.
        try:
            _output_off()
        except Exception as exc:
            print(f"WARNING: could not confirm AWG output is off: {exc} -- "
                  "turn it off at the instrument.")
    print(f"\nDrift sweep complete: {total} setting(s) in {SWEEP_FOLDER}")


if __name__ == "__main__":
    main()
