"""Pure-numpy PUND current subtraction and work/polarization metrics,
restricted to the rise window of the positive pulses. Ported from
20260820/BTO_F9_wo_curAmp_PUND_test2/plot_PUND_pulse_subtraction_gap1.py's
process_file, scoped to a single in-memory trace instead of a CSV file, and
scoped to [0, rise_time_s] instead of the full pulse width."""

import numpy as np

from waveform import PundTiming

RISE_TRACE_POINTS = 1000


def _validate_trace(time_s, values, name):
    if time_s.shape != values.shape or time_s.ndim != 1 or time_s.size < 2:
        raise ValueError(f"time_s and {name} must be equal-length 1D arrays")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing")


def subtract_pund_pulses(time_s: np.ndarray, current_a: np.ndarray, timing: PundTiming,
                         periods: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """I_diff(t) = I_first_pulse(t) - I_second_pulse(t) over [0, rise_time_s],
    averaged across `periods` repeats of the captured AWG buffer."""
    time_s, current_a = np.asarray(time_s, dtype=float), np.asarray(current_a, dtype=float)
    _validate_trace(time_s, current_a, "current_a")
    if periods < 1:
        raise ValueError("periods must be at least 1")
    relative_time = np.linspace(0.0, timing.rise_time_s, RISE_TRACE_POINTS)
    differences = []
    for period in range(periods):
        period_start = period * timing.period_duration_s
        first_times = period_start + timing.first_rise_start_s + relative_time
        second_times = period_start + timing.second_rise_start_s + relative_time
        if first_times[0] < time_s[0] or second_times[-1] > time_s[-1]:
            continue
        first_current = np.interp(first_times, time_s, current_a)
        second_current = np.interp(second_times, time_s, current_a)
        differences.append(first_current - second_current)
    if not differences:
        raise ValueError("No complete positive-pulse pair found in the captured time range")
    return relative_time, np.mean(differences, axis=0)


def measured_voltage_on_first_rise(time_s: np.ndarray, voltage_v: np.ndarray, timing: PundTiming,
                                   periods: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Raw measured voltage during the first positive pulse's rise, over
    [0, rise_time_s], averaged across `periods` repeats. P and U are driven
    with the identical shaped rise, so this equals the second pulse's
    measured voltage up to noise."""
    time_s, voltage_v = np.asarray(time_s, dtype=float), np.asarray(voltage_v, dtype=float)
    _validate_trace(time_s, voltage_v, "voltage_v")
    if periods < 1:
        raise ValueError("periods must be at least 1")
    relative_time = np.linspace(0.0, timing.rise_time_s, RISE_TRACE_POINTS)
    traces = []
    for period in range(periods):
        period_start = period * timing.period_duration_s
        first_times = period_start + timing.first_rise_start_s + relative_time
        if first_times[0] < time_s[0] or first_times[-1] > time_s[-1]:
            continue
        traces.append(np.interp(first_times, time_s, voltage_v))
    if not traces:
        raise ValueError("No complete first-pulse rise found in the captured time range")
    return relative_time, np.mean(traces, axis=0)


def _trapz(y, x):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def work_and_polarization_change(relative_time: np.ndarray, current_diff_a: np.ndarray,
                                 voltage_v: np.ndarray, area_m2: float) -> tuple[float, float]:
    """work_j = integral(V * I_diff dt) over the rise window (the objective).
    delta_p_c_per_m2 = integral(I_diff dt) / area (the constrained change of
    polarization)."""
    relative_time, current_diff_a, voltage_v = (
        np.asarray(a, dtype=float) for a in (relative_time, current_diff_a, voltage_v)
    )
    if not (relative_time.shape == current_diff_a.shape == voltage_v.shape and relative_time.size >= 2):
        raise ValueError("relative_time, current_diff_a, and voltage_v must be equal-length arrays")
    if area_m2 <= 0:
        raise ValueError("area_m2 must be positive")
    work_j = _trapz(voltage_v * current_diff_a, relative_time)
    delta_p_c_per_m2 = _trapz(current_diff_a, relative_time) / area_m2
    return work_j, delta_p_c_per_m2
