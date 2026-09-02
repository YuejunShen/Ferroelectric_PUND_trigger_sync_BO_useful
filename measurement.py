"""Upload one rise-shape protocol and return the reduced PUND trace."""

from dataclasses import dataclass
import time

import numpy as np

from config import ExperimentConfig
from instruments import ArbGenerator, configured_scope, read_scope_channel
from pulse_subtraction import (measured_voltage_on_first_rise, subtract_pund_pulses,
                                work_and_polarization_change)
from waveform import build_pund_waveform


@dataclass(frozen=True)
class Measurement:
    time_s: np.ndarray
    current_a: np.ndarray
    voltage_v: np.ndarray
    # voltage_v is whichever one downstream analysis (Work, etc.) uses -- raw
    # or sense-resistor-corrected, per DeviceConfig.correct_voltage_for_sense_resistor.
    # These two are always kept regardless of that setting, so the correction
    # stays fully auditable:
    voltage_measured_v: np.ndarray   # raw scope-measured voltage
    voltage_resistor_v: np.ndarray   # current_a * current_sense_resistor_ohm
    rise_time_s: np.ndarray
    rise_voltage_v: np.ndarray
    rise_voltage_measured_v: np.ndarray
    rise_voltage_resistor_v: np.ndarray
    rise_current_diff_a: np.ndarray
    work_j: float
    delta_p_c_per_m2: float
    # What BO asked for (fed into the waveform builder / logged as Xopt's input).
    proposed_control_voltages_v: np.ndarray
    # What the scope actually measured at those same knot times -- can differ
    # from proposed_control_voltages_v due to AWG DAC quantization, slew-rate
    # limiting, cabling, etc.
    measured_control_voltages_v: np.ndarray


def apply_rise_and_measure(control_voltages_v: np.ndarray, cfg: ExperimentConfig) -> Measurement:
    arb, timing = build_pund_waveform(control_voltages_v, cfg.waveform)
    if cfg.scope_capture_periods < 1:
        raise ValueError("scope_capture_periods must be at least 1")
    capture_duration_s = cfg.scope_capture_periods * arb.duration_s
    generator = ArbGenerator(cfg)
    generator.upload(arb)
    try:
        with configured_scope(cfg, capture_duration_s) as scope:
            # Arm first; the ARB sync/trigger output marks buffer restart (t=0).
            channels = (cfg.instruments.current_channel, cfg.instruments.voltage_channel,
                        cfg.instruments.trigger_channel)
            scope.write(":DIGITIZE " + ",".join(f"CHANNEL{ch}" for ch in channels))
            generator.output(True)
            time.sleep(cfg.output_on_wait_s)
            scope.query("*OPC?")
            time_i, raw_i = read_scope_channel(scope, cfg.instruments.current_channel)
            time_v, raw_v = read_scope_channel(scope, cfg.instruments.voltage_channel)
            time_trigger, trigger_v = read_scope_channel(scope, cfg.instruments.trigger_channel)
    finally:
        generator.output(False)

    if not (time_i.size == time_v.size == time_trigger.size):
        raise RuntimeError("Scope returned unequal current, voltage, and trigger lengths")
    current = raw_i * cfg.device.effective_current_gain_a_per_v
    voltage_measured = raw_v * cfg.instruments.voltage_scale_factor
    voltage_resistor = current * cfg.device.current_sense_resistor_ohm
    if cfg.device.correct_voltage_for_sense_resistor:
        # The voltage probe's node includes the current-sense resistor's IR
        # drop; subtract it to recover the voltage across the device alone.
        # Everything downstream (rise_voltage_v, measured_control_voltages_v,
        # and therefore the Work integral) uses this corrected voltage.
        voltage = voltage_measured - voltage_resistor
    else:
        voltage = voltage_measured

    # Locate the measured positive trigger-out edge; it is the AWG buffer restart (t=0).
    level = cfg.instruments.trigger_level_v
    crossings = np.flatnonzero((trigger_v[:-1] < level) & (trigger_v[1:] >= level))
    if crossings.size:
        index = int(crossings[0])
        dv = trigger_v[index + 1] - trigger_v[index]
        fraction = (level - trigger_v[index]) / dv
        crossing_s = time_trigger[index] + fraction * (time_trigger[index + 1] - time_trigger[index])
    else:
        above = np.flatnonzero(trigger_v >= level)
        if above.size == 0:
            raise RuntimeError("Could not locate the channel-3 trigger-out edge")
        crossing_s = time_trigger[int(above[0])]
    relative_time = time_i - crossing_s

    rise_time_s, current_diff_a = subtract_pund_pulses(
        relative_time, current, timing, cfg.scope_capture_periods
    )
    _, rise_voltage_v = measured_voltage_on_first_rise(
        relative_time, voltage, timing, cfg.scope_capture_periods
    )
    _, rise_voltage_measured_v = measured_voltage_on_first_rise(
        relative_time, voltage_measured, timing, cfg.scope_capture_periods
    )
    _, rise_voltage_resistor_v = measured_voltage_on_first_rise(
        relative_time, voltage_resistor, timing, cfg.scope_capture_periods
    )
    work_j, delta_p = work_and_polarization_change(
        rise_time_s, current_diff_a, rise_voltage_v, cfg.device.electrode_area_m2
    )
    proposed = np.asarray(control_voltages_v, dtype=float)
    # Same nominal knot times build_shaped_rise used to lay out the proposed
    # vector; read the measured rise voltage back at those instants.
    control_t = np.linspace(0.0, cfg.waveform.rise_time_s, proposed.size)
    measured_controls = np.interp(control_t, rise_time_s, rise_voltage_v)
    return Measurement(
        relative_time, current, voltage, voltage_measured, voltage_resistor,
        rise_time_s, rise_voltage_v, rise_voltage_measured_v, rise_voltage_resistor_v, current_diff_a,
        work_j, delta_p,
        proposed.copy(), measured_controls,
    )
