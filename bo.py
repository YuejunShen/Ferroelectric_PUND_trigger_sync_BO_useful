"""Xopt constrained-EI optimizer for the PUND rise shape, adapted from
test1_trigger_sync/bo.py: minimizes switching work subject to the measured
change of polarization staying at or above the linear-rise reference."""

from dataclasses import dataclass, field
from contextlib import nullcontext
import sys
import numpy as np
import pandas as pd
from packaging.version import Version

from config import BOConfig, PUNDWaveformConfig

try:
    import xopt
    from xopt.generators.bayesian import ExpectedImprovementGenerator, BayesianExplorationGenerator
    from xopt.generators.bayesian.mobo import MOBOGenerator
    from xopt.vocs import VOCS
except ImportError:
    xopt = None
    ExpectedImprovementGenerator = None
    BayesianExplorationGenerator = None
    MOBOGenerator = None
    VOCS = None


if sys.platform == "win32" and xopt is not None:
    # BoTorch always asks threadpoolctl to enumerate loaded BLAS DLLs before
    # fitting. That Windows API call is intermittently unavailable on this lab
    # PC. Only disable the thread-count wrapper; optimization itself is unchanged.
    import botorch.optim.utils.timeout as _botorch_timeout

    _botorch_timeout.threadpool_limits = lambda *args, **kwargs: nullcontext()

# "linear rise": the initial-phase repeats; "linear reference": the drift
# references interleaved into the BO phase (BOConfig.reference_interval).
LINEAR_PHASES = ("linear rise", "linear reference")


@dataclass
class BayesianRiseOptimizer:
    waveform_cfg: PUNDWaveformConfig
    bo_cfg: BOConfig
    data: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    p_target: float | None = field(default=None, init=False)
    p_lower_bound: float | None = field(default=None, init=False)
    p_scale: float | None = field(default=None, init=False)
    work_scale: float | None = field(default=None, init=False)
    p_lower_norm_threshold: float | None = field(default=None, init=False)
    _generator: object | None = field(default=None, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    # random_design="spread": shuffled target average voltages, one per random measurement.
    _spread_means_v: np.ndarray | None = field(default=None, init=False, repr=False)
    # True where an initial-phase evaluation is a linear rise: evaluation 0,
    # plus linear_repeats - 1 random slots among the random measurements.
    _linear_slots: np.ndarray = field(init=False, repr=False)
    # (evaluation index, Work, delta_P) of the linear-rise reference; see _fit_reference.
    _reference: tuple | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if ExpectedImprovementGenerator is None:
            raise RuntimeError("Xopt is required: pip install xopt")
        if Version(xopt.__version__) < Version("3.2.1"):
            raise RuntimeError(f"Xopt {xopt.__version__} is unsupported; use .\\run_in_xopt_env.ps1")
        self.input_names = [f"V{i}" for i in range(self.bo_cfg.rise_points)]
        self._rng = np.random.default_rng(self.bo_cfg.random_seed)
        n_initial = self.bo_cfg.linear_repeats + self.bo_cfg.initial_random_measurements
        self._linear_slots = np.zeros(n_initial, dtype=bool)
        self._linear_slots[0] = True
        later = self._rng.choice(np.arange(1, n_initial), self.bo_cfg.linear_repeats - 1, replace=False)
        self._linear_slots[later] = True

    def _make_generator(self):
        variables = {name: [self.waveform_cfg.minimum_voltage_v,
                            self.waveform_cfg.maximum_voltage_v]
                     for name in self.input_names}
        if self.bo_cfg.mode == "explore":
            # No feasibility constraint and no single objective to minimize --
            # Work_NORM and delta_P_NORM are equal-standing EXPLORE targets,
            # so the generator chases whichever untried protocol the joint GP
            # is currently most uncertain about, across both outputs at once.
            self.vocs = VOCS(
                variables=variables,
                objectives={"Work_NORM": "EXPLORE", "delta_P_NORM": "EXPLORE"},
                observables=["Work", "delta_P", "P_violation"],
            )
            generator = BayesianExplorationGenerator(vocs=self.vocs)
        elif self.bo_cfg.mode == "pareto":
            # No threshold: Work and delta_P are traded off directly. Only
            # shapes beating the reference -- the mean linear-rise Work, and
            # P_lower_bound (P_TARGET plus p_lower_relative_tolerance) -- on
            # both add hypervolume, so that part of the Pareto front is what BO maps.
            self.vocs = VOCS(
                variables=variables,
                objectives={"Work_NORM": "MINIMIZE", "delta_P_NORM": "MAXIMIZE"},
                observables=["Work", "delta_P", "P_violation"],
            )
            generator = MOBOGenerator(
                vocs=self.vocs,
                reference_point={"Work_NORM": 1.0, "delta_P_NORM": self.p_lower_norm_threshold},
            )
        else:
            self.vocs = VOCS(
                variables=variables,
                objectives={"Work_NORM": "MINIMIZE"},
                constraints={"P_lower_NORM": ["GREATER_THAN", self.p_lower_norm_threshold]},
                observables=["Work", "delta_P", "delta_P_NORM", "P_violation"],
            )
            generator = ExpectedImprovementGenerator(vocs=self.vocs)
        generator.gp_constructor.use_low_noise_prior = self.bo_cfg.use_low_noise_prior
        generator.numerical_optimizer.n_restarts = self.bo_cfg.n_restarts
        generator.numerical_optimizer.max_iter = self.bo_cfg.max_iter
        generator.numerical_optimizer.max_time = self.bo_cfg.max_time_s
        self._generator = generator

    def _linear_shape(self):
        return np.linspace(self.waveform_cfg.linear_minimum_voltage_v,
                           self.waveform_cfg.linear_maximum_voltage_v,
                           self.bo_cfg.rise_points)

    def _is_reference(self, evaluation):
        """True for a BO-phase evaluation that is a drift-reference linear
        rise: one after every reference_interval BO steps."""
        interval = self.bo_cfg.reference_interval
        after_initial = evaluation - self._linear_slots.size
        return interval > 0 and after_initial >= 0 and (after_initial + 1) % (interval + 1) == 0

    def propose(self):
        count = len(self.data)
        low, high = (self.waveform_cfg.minimum_voltage_v,
                     self.waveform_cfg.maximum_voltage_v)
        if count < self._linear_slots.size:
            if self._linear_slots[count]:
                return self._linear_shape()
            if self.bo_cfg.random_design == "spread":
                random_index = int(np.count_nonzero(~self._linear_slots[:count]))
                return self._spread_shape(random_index, low, high)
            return self._rng.uniform(low, high, self.bo_cfg.rise_points)
        if self._is_reference(count):
            return self._linear_shape()
        inputs = self._generator.generate(1)[0]
        return np.array([inputs[name] for name in self.input_names], dtype=float)

    def _spread_shape(self, index, low, high):
        """Random shape whose average voltage is the index-th of N levels
        across [low, high], denser toward high voltage (probability of a level
        proportional to its height above low), shuffled so slow device drift
        is not correlated with voltage. Uniform random voltages average out to
        ~mid-range in many dimensions; this instead shows the GP shapes from
        barely switching to fully switched. The shape's deviations from its
        average are shrunk, never clipped, so it stays in bounds with its
        average exactly on target. For "bernstein" the control average is
        exactly the rise's time-average voltage."""
        n_random = self.bo_cfg.initial_random_measurements
        if self._spread_means_v is None:
            levels = low + (high - low) * np.sqrt((np.arange(n_random) + 0.5) / n_random)
            self._spread_means_v = self._rng.permutation(levels)
        target = self._spread_means_v[index]
        deviation = self._rng.uniform(low, high, self.bo_cfg.rise_points)
        deviation -= deviation.mean()
        room_up = (high - target) / deviation.max() if deviation.max() > 0 else np.inf
        room_down = (target - low) / -deviation.min() if deviation.min() < 0 else np.inf
        shape = target + deviation * min(1.0, room_up, room_down)
        # Removes only floating-point overshoot (~1e-16 V) at a bound, which
        # waveform.build_pund_waveform's exact range check would reject.
        return np.clip(shape, low, high)

    def observe(self, proposed_voltages_v, work_j, delta_p_c_per_m2):
        # The GP is fit against what was proposed/commanded, not the
        # independently measured-at-knot-times voltage (see measurement.py's
        # Measurement.measured_control_voltages_v) -- otherwise measurement
        # noise on the inputs would corrupt the surrogate model's input space.
        protocol = np.asarray(proposed_voltages_v, dtype=float)
        if protocol.shape != (self.bo_cfg.rise_points,):
            raise ValueError("Proposed voltage control-point count does not match Xopt")
        n_initial = self._linear_slots.size
        evaluation = len(self.data)
        if evaluation < n_initial and self._linear_slots[evaluation]:
            # The linear_repeats linear rises: evaluation 0, then scattered
            # among the random measurements, so slow drift over the initial
            # phase averages into the target.
            phase = "linear rise"
            phase_evaluation = int(np.count_nonzero(self._linear_slots[:evaluation]))
        elif evaluation < n_initial:
            phase = "random initial"
            phase_evaluation = int(np.count_nonzero(~self._linear_slots[:evaluation])) + 1
        else:
            phase = "linear reference" if self._is_reference(evaluation) else "BO optimization"
            phase_evaluation = int(np.count_nonzero(self.data["phase"] == phase)) + 1
        row = dict(zip(self.input_names, protocol))
        row.update({"Work": float(work_j), "delta_P": float(delta_p_c_per_m2),
                    "phase": phase, "phase_evaluation": phase_evaluation})
        self.data = pd.concat([self.data, pd.DataFrame([row], index=[evaluation])])
        if phase in LINEAR_PHASES:
            self._fit_reference()
        # A new linear rise moves the reference of earlier rows too, so every
        # row is renormalized; between linear rises this changes only the new one.
        self.data = self._normalized(self.data)
        if evaluation == n_initial - 1 or phase == "linear reference":
            # Initial phase over, or the drift reference just moved: hand
            # every (re)normalized row to a fresh generator.
            self._make_generator()
            self._generator.add_data(self.data)
        elif evaluation >= n_initial:
            self._generator.add_data(self.data.loc[[evaluation]])
        return self.data.loc[evaluation].to_dict()

    def _fit_reference(self):
        """Linear-rise reference Work and delta_P (P_TARGET) as a function of
        evaluation index. reference_interval = 0: the mean of every linear rise
        so far, so initial-phase rows are final once the initial phase ends.
        Otherwise a centered mean of reference_window consecutive linear rises,
        interpolated between them and held flat after the newest one, so each
        row is compared against the device as it was when it was measured."""
        linear = self.data[self.data["phase"].isin(LINEAR_PHASES)]
        work = linear["Work"].to_numpy(dtype=float)
        delta_p = linear["delta_P"].to_numpy(dtype=float)
        if self.bo_cfg.reference_interval == 0:
            evaluations, work, delta_p = np.zeros(1), np.array([work.mean()]), np.array([delta_p.mean()])
        else:
            evaluations = linear.index.to_numpy(dtype=float)
            work = _centered_mean(work, self.bo_cfg.reference_window)
            delta_p = _centered_mean(delta_p, self.bo_cfg.reference_window)
        self._reference = (evaluations, work, delta_p)
        self._set_target(float(work[-1]), float(delta_p[-1]))

    def _set_target(self, work_j, delta_p):
        """Newest reference. p_lower_relative_tolerance shifts the BO boundary
        relative to P_TARGET, as a fraction of |P_TARGET|: negative LOOSENS,
        positive TIGHTENS, 0.0 puts it exactly at P_TARGET ("optimize": the
        feasibility bound; "pareto": the reference delta_P). Normalized, that
        boundary is the same for every row however far the reference drifts."""
        self.p_target = delta_p
        self.p_lower_bound = self.p_target + self.bo_cfg.p_lower_relative_tolerance * abs(self.p_target)
        self.p_scale = abs(self.p_target) if self.p_target != 0.0 else 1.0
        self.work_scale = abs(work_j) if work_j != 0.0 else 1.0
        self.p_lower_norm_threshold = self.p_lower_bound / self.p_scale

    def _normalized(self, data):
        """data with every target/normalized column filled in against each
        row's own reference (see _fit_reference)."""
        evaluations, work_reference, p_reference = self._reference
        at = data.index.to_numpy(dtype=float)
        p_target = np.interp(at, evaluations, p_reference)
        work_target = np.interp(at, evaluations, work_reference)
        p_scale = np.where(p_target != 0.0, np.abs(p_target), 1.0)
        work_scale = np.where(work_target != 0.0, np.abs(work_target), 1.0)
        p_lower_bound = p_target + self.bo_cfg.p_lower_relative_tolerance * np.abs(p_target)
        delta_p = data["delta_P"].to_numpy(dtype=float)
        delta_p_norm = delta_p / p_scale
        return data.assign(
            Work_NORM=data["Work"].to_numpy(dtype=float) / work_scale,
            delta_P_NORM=delta_p_norm,
            P_target=p_target,
            P_lower_bound=p_lower_bound,
            P_violation=np.maximum(p_lower_bound - delta_p, 0.0),
            P_lower=delta_p,
            P_lower_NORM=delta_p_norm,
            P_SCALE=p_scale,
            WORK_SCALE=work_scale,
            # "feasible" reports whether delta_P reached the true physical
            # target, independent of p_lower_relative_tolerance -- that
            # tolerance only shifts what the BO constraint itself searches
            # for (p_lower_bound above), it does not redefine what counts as
            # a successful result for reporting.
            feasible=delta_p >= p_target,
        )


def _centered_mean(values, window):
    """Mean of the window values centered on each one, truncated at the ends."""
    half = window // 2
    return np.array([values[max(0, i - half):i - half + window].mean() for i in range(values.size)])
