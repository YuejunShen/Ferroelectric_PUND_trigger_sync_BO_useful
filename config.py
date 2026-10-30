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
    # True: subtract the sense-resistor IR drop from the measured voltage (used for Work).
    correct_voltage_for_sense_resistor: bool = True
    current_sense_resistor_ohm: float = 50.0

    @property
    def effective_current_gain_a_per_v(self) -> float:
        return self.current_preamplifier_gain_a_per_v * (2.0 if self.use_current_amplifier else 1.0)


@dataclass(frozen=True)
class PUNDWaveformConfig:
    sample_rate_hz: float = 10e7
    pulse_width_s: float = 2e-6
    rise_time_s: float = 1e-6  # tau: the BO-shaped P/U rise
    other_ramp_time_s: float | None = 0.1e-6  # reset rise/fall and P/U falls; None = rise_time_s
    pre_baseline_s: float = 2e-6
    post_baseline_s: float = 2e-6
    gap1_s: float = 1e-6
    gap2_s: float = 1e-6
    reset_voltage_v: float = -0.7  # fixed, must fully pole the device
    positive_amplitude_v: float = 0.7  # fixed P/U flat top
    rise_basis: str = "points"  # "points" (linear interpolation) or "bernstein" (polynomial)
    linear_minimum_voltage_v: float = 0.0
    linear_maximum_voltage_v: float = 0.7
    minimum_voltage_v: float = 0  # BO lower bound
    maximum_voltage_v: float = 1.0  # BO upper bound
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
    # "optimize": minimize Work subject to delta_P >= linear-rise target (constrained EI)
    # "pareto":   minimize Work and maximize delta_P together, no threshold;
    #             pick the result from the Pareto front plot at the end
    # "explore":  map Work and delta_P over the search space, no optimization
    mode: str = "optimize"
    rise_points: int = 10  # number of BO voltages describing the rise
    # linear-rise measurements averaged into the target: the first, the rest at random among the random runs
    linear_repeats: int = 10
    initial_random_measurements: int = 30
    # "uniform": every voltage random in the BO range (average voltage clusters mid-range)
    # "spread":  random shapes whose average voltages span the whole BO range, more often high, in random order
    random_design: str = "spread"
    bo_steps: int = 400
    # Drift correction: one extra linear rise after every this many BO steps
    # (added on top of bo_steps); every row is then normalized against the
    # linear rises measured around it instead of the initial-phase mean.
    # 0 = off (target fixed by the initial linear repeats).
    reference_interval: int = 10
    # Linear rises averaged into each point of the drift reference (centered, in time order)
    reference_window: int = 5
    # "optimize": feasibility bound, "pareto": reference delta_P = target * (1 + this)
    p_lower_relative_tolerance: float = 2e-3
    random_seed: int = field(default_factory=lambda: time.time_ns() % (2**32))
    n_restarts: int = 5
    max_iter: int = 100
    use_low_noise_prior: bool = True  # True: GP assumes near-noiseless data (September behaviour)
    max_time_s: float | None = None  # None avoids Botorch's Windows threadpoolctl timeout wrapper

    def __post_init__(self):
        if self.rise_points < 2:
            raise ValueError("rise_points must be at least 2")
        if self.linear_repeats < 1:
            raise ValueError("linear_repeats must be at least 1")
        if self.mode not in ("optimize", "explore", "pareto"):
            raise ValueError(f"mode must be 'optimize', 'explore' or 'pareto', got {self.mode!r}")
        if self.random_design not in ("uniform", "spread"):
            raise ValueError(f"random_design must be 'uniform' or 'spread', got {self.random_design!r}")
        if self.reference_interval < 0:
            raise ValueError("reference_interval must be >= 0 (0 = no drift correction)")
        if self.reference_window < 1:
            raise ValueError("reference_window must be at least 1")

    @property
    def reference_measurements(self) -> int:
        """Linear references interleaved into the BO phase (one after each full block)."""
        return self.bo_steps // self.reference_interval if self.reference_interval else 0

    @property
    def total_measurements(self) -> int:
        return (self.linear_repeats + self.initial_random_measurements + self.bo_steps
                + self.reference_measurements)


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
    scope_capture_periods: int = 1  # AWG periods captured and averaged per measurement
    output_on_wait_s: float = 0.05
    # Linear-rise repeats before measuring, to settle the device ("pareto" also uses them as reference)
    warmup_repeats: int = 100
    enable_control_panel: bool = True
    sweep_tag: str = ""  # set by run_parameter_sweep.py

    @property
    def output_directory(self) -> Path:
        time_of_day = self.run_timestamp.split("_")[1][:4]
        suffix = f"_{self.sweep_tag}" if self.sweep_tag else ""
        return self.data_mother_path / f"BO_pund_rise_{self.device.device_name}_{time_of_day}{suffix}"

    @property
    def history_file(self) -> Path:
        return self.output_directory / f"bo_history_{self.run_timestamp}.csv"
