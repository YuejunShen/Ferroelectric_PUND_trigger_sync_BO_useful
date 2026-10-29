"""Configuration for the PUND rise-shape optimization experiment."""

from dataclasses import dataclass, field
import datetime as dt
import time
from pathlib import Path


@dataclass(frozen=True)
class InstrumentConfig:
    function_generator_address: str = "USB0::0x0957::0x2707::MY57301577::INSTR"
    oscilloscope_address: str = "USB0::0x0957::0x9005::MY50140103::INSTR"
    function_generator_timeout_ms: int = 100_000
    oscilloscope_timeout_ms: int = 100_000
    current_channel: int = 1
    voltage_channel: int = 2
    trigger_channel: int = 3
    # current_input_impedance_ohm: float = 50.0
    # current_termination_correction: float = 2.0
    voltage_scale_factor: float = 1.0
    current_scale_v_per_div: float = 0.02
    voltage_scale_v_per_div: float = 1
    current_offset_v: float = 0.0
    voltage_offset_v: float = 0.0
    trigger_scale_v_per_div: float = 1.0
    trigger_offset_v: float = 0.0
    trigger_level_v: float = 0.5
    average_enabled: bool = True
    average_count: int = 256
    scope_time_step_s: float = 1e-9


@dataclass(frozen=True)
class DeviceConfig:
    device_name: str = "BTO_PLD219_2"
    electrode_area_m2: float = (20e-6) ** 2
    current_preamplifier_gain_a_per_v: float = 1.0 / 50.0
    use_current_amplifier: bool = False
    # The voltage channel measures V_device + I * current_sense_resistor_ohm,
    # since the current-sense path (through this resistor) shares its node
    # with the voltage probe. When True, that IR drop is subtracted out to
    # recover the voltage actually across the ferroelectric device alone --
    # this corrected voltage is then used everywhere downstream, including
    # the Work integral. When False (default), voltage_v stays exactly what
    # the scope measured (device + sense-resistor drop), as before.
    correct_voltage_for_sense_resistor: bool = True
    current_sense_resistor_ohm: float = 50.0

    @property
    def effective_current_gain_a_per_v(self) -> float:
        return self.current_preamplifier_gain_a_per_v * (2.0 if self.use_current_amplifier else 1.0)


@dataclass(frozen=True)
class PUNDWaveformConfig:
    sample_rate_hz: float = 10e7
    pulse_width_s: float = 2e-6
    # Duration of the two BO-shaped positive (P/U) rises.
    rise_time_s: float = 2e-6
    # Duration of every other ramp: the reset pulse's rise and fall and the
    # two positive pulses' falls. None (default) reuses rise_time_s, as before.
    other_ramp_time_s: float | None = 0.1e-6
    pre_baseline_s: float = 2e-6
    post_baseline_s: float = 2e-6
    gap1_s: float = 0.5e-6
    gap2_s: float = 0.5e-6
    # Fixed (not BO-controlled) negative reset-pulse voltage: must fully pole
    # the device regardless of whatever positive rise voltages BO is testing.
    reset_voltage_v: float = -1
    # Fixed (not BO-controlled) flat-top amplitude of the two positive (P/U)
    # pulses, independent of reset_voltage_v's magnitude -- the rise is
    # concatenated directly into this flat top with no forced continuity
    # (see waveform.build_pund_waveform), so BO is free to shape the approach
    # without that shape also having to hit this amplitude at its last knot.
    positive_amplitude_v: float = 1
    # How the BOConfig.rise_points BO variables define the rise V(t):
    # "points"    -- voltages at evenly spaced times, linearly interpolated.
    # "bernstein" -- coefficients of a degree (rise_points - 1) Bernstein
    #                polynomial. V(t) always stays between the smallest and
    #                largest coefficient, so the BO voltage bounds hold without
    #                clipping; V(0) is the first and V(end) the last coefficient.
    #                Evenly spaced coefficients give the same linear ramp as
    #                "points", so the initial linear rise is identical in both.
    rise_basis: str = "bernstein"
    linear_minimum_voltage_v: float = 0.0
    linear_maximum_voltage_v: float = 1
    minimum_voltage_v: float = 0  # BO lower bound
    maximum_voltage_v: float = 1  # BO upper bound
    safety_max_voltage_v: float = 2.0  # hard ceiling, independent of the BO search bounds
    maximum_arb_points: int = 1_000_000

    def __post_init__(self):
        if self.reset_voltage_v >= 0.0:
            raise ValueError("reset_voltage_v must be negative")
        if self.positive_amplitude_v <= 0.0:
            raise ValueError("positive_amplitude_v must be positive")
        if self.rise_basis not in ("points", "bernstein"):
            raise ValueError(f"Unknown rise_basis {self.rise_basis!r}")
        if self.other_ramp_time_s is not None and self.other_ramp_time_s <= 0.0:
            raise ValueError("other_ramp_time_s must be positive (or None to reuse rise_time_s)")
        # minimum_voltage_v/linear_minimum_voltage_v may be negative (rise
        # shapes are free to dip below 0 V; PUND leakage cancellation only
        # requires both positive pulses share the identical control vector,
        # regardless of sign) -- bounded symmetrically by safety_max_voltage_v,
        # same as the positive ceiling, reset_voltage_v, and positive_amplitude_v below.
        allowed_max = max(self.linear_maximum_voltage_v, self.maximum_voltage_v)
        allowed_min = min(self.linear_minimum_voltage_v, self.minimum_voltage_v)
        if (allowed_max > self.safety_max_voltage_v
                or -allowed_min > self.safety_max_voltage_v
                or -self.reset_voltage_v > self.safety_max_voltage_v
                or self.positive_amplitude_v > self.safety_max_voltage_v):
            raise ValueError("linear/BO/reset/positive-amplitude voltage exceeds safety_max_voltage_v")

    @property
    def other_ramp_s(self) -> float:
        """Non-P/U-rise ramp time actually used (other_ramp_time_s, or rise_time_s if None)."""
        return self.rise_time_s if self.other_ramp_time_s is None else self.other_ramp_time_s


@dataclass(frozen=True)
class BOConfig:
    # Number of voltage knots (or Bernstein coefficients, see
    # PUNDWaveformConfig.rise_basis) parameterizing the shared rise shape, i.e. the
    # dimensionality of the BO search space (both positive pulses reuse the
    # same rise_points-length control vector -- see waveform.build_shaped_rise).
    rise_points: int = 10
    initial_random_measurements: int =30
    bo_steps: int = 400
    # Shifts the BO feasibility boundary relative to P_TARGET (the first,
    # linear-rise evaluation's delta_P), as a fraction of |P_TARGET|:
    # negative LOOSENS the bound (delta_P may fall below P_TARGET by that
    # fraction and still be searched as feasible), positive TIGHTENS it,
    # 0.0 means the search boundary sits exactly at P_TARGET. Does not
    # affect the "feasible" column, which always reports delta_P >= P_TARGET.
    p_lower_relative_tolerance: float = 1e-3
    random_seed: int = field(default_factory=lambda: time.time_ns() % (2**32))
    n_restarts: int = 5
    max_iter: int = 100
    # None avoids Botorch's Windows threadpoolctl timeout wrapper.
    max_time_s: float | None = None
    # "optimize" (default): constrained Expected Improvement, as before --
    # minimizes Work_NORM subject to the P_lower_NORM feasibility constraint.
    # "explore": unconstrained Bayesian exploration (BayesianExplorationGenerator).
    # The P_lower_NORM constraint is dropped entirely and Work_NORM/delta_P_NORM
    # stop being an objective/observable pair to optimize -- both become equal-
    # standing EXPLORE targets, and the generator proposes whichever untried
    # protocol the joint GP over (Work_NORM, delta_P_NORM) is currently least
    # certain about. The goal isn't a minimum, it's mapping which protocols
    # (V0..Vn) produce which combinations of Work and delta_P -- feed the
    # resulting bo_history_*.csv to plot_landscape.py to see that map.
    mode: str = "optimize"

    def __post_init__(self):
        if self.rise_points < 2:
            raise ValueError("rise_points must be at least 2")
        if self.mode not in ("optimize", "explore"):
            raise ValueError(f"mode must be 'optimize' or 'explore', got {self.mode!r}")

    @property
    def total_measurements(self) -> int:
        return 1 + self.initial_random_measurements + self.bo_steps


@dataclass(frozen=True)
class ExperimentConfig:
    instruments: InstrumentConfig = field(default_factory=InstrumentConfig)
    waveform: PUNDWaveformConfig = field(default_factory=PUNDWaveformConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    run_timestamp: str = field(
        default_factory=lambda: dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    data_mother_path: Path = field(
        default_factory=lambda: Path(r"D:\Ferroelectric_BTO_optimization") / dt.datetime.now().strftime("%Y%m%d")
    )
    # Number of full AWG-buffer repeats to capture in one digitize call and
    # average over in pulse_subtraction.subtract_pund_pulses.
    scope_capture_periods: int = 1
    output_on_wait_s: float = 0.05
    # Number of times to apply the initial linear-rise waveform (unmeasured,
    # unsaved, not fed to BO) before the real measurement loop starts, to let
    # the device settle into a repeatable switching state. 0 (default) skips
    # this and starts measuring immediately, as before.
    warmup_repeats: int = 0
    enable_control_panel: bool = True
    # Human-readable tag describing which swept parameter values this run
    # uses (e.g. "rise_points10_rep2"), set by run_parameter_sweep.py so each
    # run's output folder self-documents its settings. Empty for a normal,
    # non-swept run (folder naming is unchanged in that case).
    sweep_tag: str = ""

    @property
    def output_directory(self) -> Path:
        # HHMM from run_timestamp so repeat runs (same device, same day) get their own folder.
        time_of_day = self.run_timestamp.split("_")[1][:4]
        suffix = f"_{self.sweep_tag}" if self.sweep_tag else ""
        return self.data_mother_path / f"BO_pund_rise_{self.device.device_name}_{time_of_day}{suffix}"

    @property
    def history_file(self) -> Path:
        return self.output_directory / f"bo_history_{self.run_timestamp}.csv"
