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
    voltage_scale_v_per_div: float = 0.3
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
    device_name: str = "BTO_F9"
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
    sample_rate_hz: float = 5e7
    pulse_width_s: float = 2e-6
    rise_time_s: float = 1e-6
    pre_baseline_s: float = 2e-6
    post_baseline_s: float = 2e-6
    gap1_s: float = 1e-6
    gap2_s: float = 1e-6
    # Fixed (not BO-controlled) negative reset-pulse voltage: must fully pole
    # the device regardless of whatever positive rise voltages BO is testing.
    reset_voltage_v: float = -0.8
    linear_minimum_voltage_v: float = 0.0
    linear_maximum_voltage_v: float = 0.8
    minimum_voltage_v: float = 0  # BO lower bound
    maximum_voltage_v: float = 1  # BO upper bound
    safety_max_voltage_v: float = 1.0  # hard ceiling, independent of the BO search bounds
    maximum_arb_points: int = 1_000_000

    def __post_init__(self):
        if self.reset_voltage_v >= 0.0:
            raise ValueError("reset_voltage_v must be negative")
        # minimum_voltage_v/linear_minimum_voltage_v may be negative (rise
        # shapes are free to dip below 0 V; PUND leakage cancellation only
        # requires both positive pulses share the identical control vector,
        # regardless of sign) -- bounded symmetrically by safety_max_voltage_v,
        # same as the positive ceiling and reset_voltage_v below.
        allowed_max = max(self.linear_maximum_voltage_v, self.maximum_voltage_v)
        allowed_min = min(self.linear_minimum_voltage_v, self.minimum_voltage_v)
        if (allowed_max > self.safety_max_voltage_v
                or -allowed_min > self.safety_max_voltage_v
                or -self.reset_voltage_v > self.safety_max_voltage_v):
            raise ValueError("linear/BO/reset voltage exceeds safety_max_voltage_v")


@dataclass(frozen=True)
class BOConfig:
    # Number of voltage knots parameterizing the shared rise shape, i.e. the
    # dimensionality of the BO search space (both positive pulses reuse the
    # same rise_points-length control vector -- see waveform.build_shaped_rise).
    rise_points: int = 10
    initial_random_measurements: int = 100
    bo_steps: int = 300
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

    def __post_init__(self):
        if self.rise_points < 2:
            raise ValueError("rise_points must be at least 2")

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
    data_mother_path: Path = Path(r"D:\Ferroelectric_BTO_optimization\20260824")
    # Number of full AWG-buffer repeats to capture in one digitize call and
    # average over in pulse_subtraction.subtract_pund_pulses.
    scope_capture_periods: int = 1
    output_on_wait_s: float = 1.0
    enable_control_panel: bool = True

    @property
    def output_directory(self) -> Path:
        # HHMM from run_timestamp so repeat runs (same device, same day) get their own folder.
        time_of_day = self.run_timestamp.split("_")[1][:4]
        return self.data_mother_path / f"BO_pund_rise_{self.device.device_name}_{time_of_day}"

    @property
    def history_file(self) -> Path:
        return self.output_directory / f"bo_history_{self.run_timestamp}.csv"
