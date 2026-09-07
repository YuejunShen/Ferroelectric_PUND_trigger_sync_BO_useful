"""Pure functions for constructing the physical PUND ARB waveform.

Pulse train: pre-baseline -> reset pulse (negative, linear ramps) -> gap1 ->
positive pulse 1 (BO-shaped rise -> flat top -> linear fall) -> gap2 ->
positive pulse 2 (identical BO-shaped rise -> flat top -> linear fall) ->
post-baseline. Only the rise of the two positive pulses is arbitrary; both
positive pulses share the exact same rise shape, which is required for PUND
leakage cancellation (subtracting pulse 2 from pulse 1 only isolates the
switching current if both pulses were driven with an identical stimulus).

The flat top's amplitude is fixed at -reset_voltage_v (magnitude-symmetric
with the reset pulse), independent of wherever the arbitrary rise ends up --
the rise is concatenated directly into the flat top with no forced
continuity, so BO is free to shape the approach without that shape also
having to hit the target amplitude at its last knot.
"""

from dataclasses import dataclass

import numpy as np

from config import PUNDWaveformConfig


@dataclass(frozen=True)
class ArbWaveform:
    voltage_v: np.ndarray
    sample_rate_hz: float

    @property
    def duration_s(self) -> float:
        return self.voltage_v.size / self.sample_rate_hz


@dataclass(frozen=True)
class PundTiming:
    """Offsets (seconds, relative to the AWG buffer restart / t=0) needed to
    locate each positive pulse's rise window in a captured trace."""
    pulse_width_s: float
    rise_time_s: float
    first_rise_start_s: float
    second_rise_start_s: float
    period_duration_s: float


def _constant(value: float, duration_s: float, sample_rate_hz: float) -> np.ndarray:
    if duration_s < 0:
        raise ValueError("Durations cannot be negative")
    count = int(round(duration_s * sample_rate_hz))
    return np.full(max(0, count), value, dtype=float)


def _ramp(start: float, stop: float, duration_s: float, sample_rate_hz: float) -> np.ndarray:
    if duration_s < 0:
        raise ValueError("Durations cannot be negative")
    count = int(round(duration_s * sample_rate_hz))
    if count <= 0:
        return np.empty(0, dtype=float)
    return np.linspace(start, stop, count, endpoint=False, dtype=float)


def build_shaped_rise(control_voltages_v: np.ndarray, rise_time_s: float, sample_rate_hz: float) -> np.ndarray:
    """Interpolate BO control knots onto the rise's sample grid. The last
    knot is just the rise's own endpoint -- it does not set the pulse's
    flat-top voltage (see build_pund_waveform), so no continuity with the
    following constant segment is implied or required."""
    controls = np.asarray(control_voltages_v, dtype=float)
    if controls.ndim != 1 or controls.size < 2:
        raise ValueError("control_voltages_v must be a 1D array with at least 2 points")
    if not np.all(np.isfinite(controls)):
        raise ValueError("Voltage points must be finite")
    count = int(round(rise_time_s * sample_rate_hz))
    if count <= 0:
        raise ValueError("rise_time_s must be positive")
    sample_t = np.arange(count) / sample_rate_hz
    control_t = np.linspace(sample_t[0], sample_t[-1], controls.size)
    return np.interp(sample_t, control_t, controls)


def positive_rise_windows(cfg: PUNDWaveformConfig) -> PundTiming:
    """Offsets of each positive pulse's rise onset, relative to the AWG
    buffer restart (t=0 = start of pre_baseline_s)."""
    reset_pulse_duration = 2.0 * cfg.pulse_width_s + 2.0 * cfg.rise_time_s
    positive_pulse_duration = cfg.pulse_width_s + 2.0 * cfg.rise_time_s
    first_rise_start = cfg.pre_baseline_s + reset_pulse_duration + cfg.gap1_s
    second_rise_start = first_rise_start + positive_pulse_duration + cfg.gap2_s
    period_duration = (second_rise_start + positive_pulse_duration
                       + cfg.post_baseline_s)
    return PundTiming(
        pulse_width_s=cfg.pulse_width_s,
        rise_time_s=cfg.rise_time_s,
        first_rise_start_s=first_rise_start,
        second_rise_start_s=second_rise_start,
        period_duration_s=period_duration,
    )


def build_pund_waveform(control_voltages_v: np.ndarray, cfg: PUNDWaveformConfig) -> tuple[ArbWaveform, PundTiming]:
    # Point count is whatever control_voltages_v provides -- how many knots
    # to use is a BO search-space decision (BOConfig.rise_points), not a
    # waveform-building one; build_shaped_rise below only requires >= 2.
    controls = np.asarray(control_voltages_v, dtype=float)
    allowed_minimum = min(cfg.linear_minimum_voltage_v, cfg.minimum_voltage_v)
    allowed_maximum = max(cfg.linear_maximum_voltage_v, cfg.maximum_voltage_v)
    if np.any((controls < allowed_minimum) | (controls > allowed_maximum)):
        raise ValueError("A rise voltage point is outside the configured linear/BO ranges")
    if np.any(np.abs(controls) > cfg.safety_max_voltage_v):
        raise ValueError("A rise voltage point exceeds safety_max_voltage_v")

    # Flat-top amplitude is fixed, magnitude-symmetric with the reset pulse --
    # independent of the arbitrary rise's own endpoint (controls[-1]), which
    # is concatenated directly into it with no forced continuity.
    positive_amplitude = -cfg.reset_voltage_v
    shaped_rise = build_shaped_rise(controls, cfg.rise_time_s, cfg.sample_rate_hz)
    flat_top = _constant(positive_amplitude, cfg.pulse_width_s, cfg.sample_rate_hz)
    fall = _ramp(positive_amplitude, 0.0, cfg.rise_time_s, cfg.sample_rate_hz)
    positive_pulse = np.concatenate((shaped_rise, flat_top, fall))

    reset_rise = _ramp(0.0, cfg.reset_voltage_v, cfg.rise_time_s, cfg.sample_rate_hz)
    reset_flat = _constant(cfg.reset_voltage_v, 2.0 * cfg.pulse_width_s, cfg.sample_rate_hz)
    reset_fall = _ramp(cfg.reset_voltage_v, 0.0, cfg.rise_time_s, cfg.sample_rate_hz)
    reset_pulse = np.concatenate((reset_rise, reset_flat, reset_fall))

    voltage = np.concatenate((
        _constant(0.0, cfg.pre_baseline_s, cfg.sample_rate_hz),
        reset_pulse,
        _constant(0.0, cfg.gap1_s, cfg.sample_rate_hz),
        positive_pulse,
        _constant(0.0, cfg.gap2_s, cfg.sample_rate_hz),
        positive_pulse,
        _constant(0.0, cfg.post_baseline_s, cfg.sample_rate_hz),
    ))
    if voltage.size > cfg.maximum_arb_points:
        raise ValueError(f"ARB waveform has {voltage.size} points; limit is {cfg.maximum_arb_points}")
    timing = positive_rise_windows(cfg)
    return ArbWaveform(voltage_v=voltage, sample_rate_hz=cfg.sample_rate_hz), timing


def normalize_for_generator(voltage_v: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Encode physical volts using the waveform's required Vpp and offset."""
    samples = np.asarray(voltage_v, dtype=float)
    if not np.all(np.isfinite(samples)):
        raise ValueError("Waveform voltages must be finite")
    low, high = float(np.min(samples)), float(np.max(samples))
    if np.isclose(low, high):
        return np.zeros_like(samples), 0.02, low
    vpp = high - low
    offset = (high + low) / 2.0
    normalized = 2.0 * (samples - low) / vpp - 1.0
    return normalized, vpp, offset
