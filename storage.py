"""Persistence of raw traces and compact BO history."""

import csv
import dataclasses
import datetime as dt
import json
from pathlib import Path

import numpy as np

from config import ExperimentConfig
from measurement import Measurement

# One persistent table across all runs and all days -- deliberately outside
# any single day's data_mother_path folder, so it keeps accumulating rows
# regardless of which day/device/sweep is currently running.
RUN_LOG_PATH = Path(r"D:\Ferroelectric_BTO_optimization\run_log.csv")


def append_run_log(cfg: ExperimentConfig, sweep_label: str = "",
                    sweep_overrides: dict | None = None, repeat_index: int | None = None) -> None:
    """Append one row recording when this run started and which parameters it
    used. Called once per run (single runs and every run of a sweep alike),
    so D:\\Ferroelectric_BTO_optimization\\run_log.csv accumulates a complete
    history of every experiment ever launched."""
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RUN_LOG_PATH.exists()
    with RUN_LOG_PATH.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["logged_at", "run_timestamp", "device_name", "mode",
                             "rise_points", "initial_random_measurements", "bo_steps",
                             "p_lower_relative_tolerance", "sweep_label", "repeat_index",
                             "sweep_overrides_json", "output_directory"])
        writer.writerow([
            dt.datetime.now().isoformat(timespec="seconds"),
            cfg.run_timestamp,
            cfg.device.device_name,
            cfg.bo.mode,
            cfg.bo.rise_points,
            cfg.bo.initial_random_measurements,
            cfg.bo.bo_steps,
            cfg.bo.p_lower_relative_tolerance,
            sweep_label,
            repeat_index if repeat_index is not None else "",
            json.dumps(sweep_overrides or {}),
            str(cfg.output_directory),
        ])


def save_run_config(cfg: ExperimentConfig) -> None:
    """Write every scan parameter (instrument, waveform, BO, device settings) to a log file."""
    cfg.output_directory.mkdir(parents=True, exist_ok=True)
    config_path = cfg.output_directory / f"run_config_{cfg.run_timestamp}.json"
    payload = dataclasses.asdict(cfg)
    payload["output_directory"] = str(cfg.output_directory)
    payload["history_file"] = str(cfg.history_file)
    with config_path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def save_measurement(measurement: Measurement, iteration: int, result: dict, cfg: ExperimentConfig) -> None:
    cfg.output_directory.mkdir(parents=True, exist_ok=True)
    trace_path = cfg.output_directory / (
        f"trace_{cfg.run_timestamp}_iteration_{iteration:03d}.csv"
    )
    with trace_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        # voltage_v is whichever the run used for analysis (raw or
        # sense-resistor-corrected); voltage_measured_v/voltage_resistor_v
        # are always the raw measured voltage and I*R_sense, so the
        # correction (if any) stays reconstructable from the CSV alone.
        writer.writerow(["time_s", "current_a", "voltage_v", "voltage_measured_v", "voltage_resistor_v"])
        writer.writerows(zip(measurement.time_s, measurement.current_a, measurement.voltage_v,
                             measurement.voltage_measured_v, measurement.voltage_resistor_v))
    rise_path = cfg.output_directory / (
        f"rise_{cfg.run_timestamp}_iteration_{iteration:03d}.csv"
    )
    with rise_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rise_time_s", "rise_voltage_v", "rise_voltage_measured_v",
                         "rise_voltage_resistor_v", "rise_current_diff_a"])
        writer.writerows(zip(measurement.rise_time_s, measurement.rise_voltage_v,
                             measurement.rise_voltage_measured_v, measurement.rise_voltage_resistor_v,
                             measurement.rise_current_diff_a))
    new_file = not cfg.history_file.exists()
    with cfg.history_file.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["evaluation", "trace_file", "rise_file",
                             # V0..Vn is what BO proposed/commanded -- this is
                             # what Xopt's GP is fit against. Measured_V0..Vn
                             # is what the scope actually measured at those
                             # same knot times; the two can differ (AWG DAC
                             # quantization, slew-rate limiting, cabling, ...).
                             *[f"V{i}" for i in range(measurement.proposed_control_voltages_v.size)],
                             *[f"Measured_V{i}" for i in range(measurement.measured_control_voltages_v.size)],
                             "Work", "Work_NORM", "delta_P", "delta_P_NORM",
                             "P_target", "P_lower_bound", "P_violation", "P_lower", "P_lower_NORM",
                             "P_SCALE", "WORK_SCALE",
                             "feasible", "phase", "phase_evaluation"])
        writer.writerow([iteration, trace_path.name, rise_path.name,
                         *measurement.proposed_control_voltages_v,
                         *measurement.measured_control_voltages_v,
                         result["Work"], result["Work_NORM"],
                         result["delta_P"], result["delta_P_NORM"], result["P_target"],
                         result["P_lower_bound"], result["P_violation"], result["P_lower"], result["P_lower_NORM"],
                         result["P_SCALE"], result["WORK_SCALE"],
                         result["feasible"], result["phase"],
                         result["phase_evaluation"]])
