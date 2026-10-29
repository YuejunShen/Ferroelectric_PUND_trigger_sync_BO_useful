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
    from xopt.vocs import VOCS
except ImportError:
    xopt = None
    ExpectedImprovementGenerator = None
    BayesianExplorationGenerator = None
    VOCS = None


if sys.platform == "win32" and xopt is not None:
    # BoTorch always asks threadpoolctl to enumerate loaded BLAS DLLs before
    # fitting. That Windows API call is intermittently unavailable on this lab
    # PC. Only disable the thread-count wrapper; optimization itself is unchanged.
    import botorch.optim.utils.timeout as _botorch_timeout

    _botorch_timeout.threadpool_limits = lambda *args, **kwargs: nullcontext()


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

    def __post_init__(self):
        if ExpectedImprovementGenerator is None:
            raise RuntimeError("Xopt is required: pip install xopt")
        if Version(xopt.__version__) < Version("3.2.1"):
            raise RuntimeError(f"Xopt {xopt.__version__} is unsupported; use .\\run_in_xopt_env.ps1")
        self.input_names = [f"V{i}" for i in range(self.bo_cfg.rise_points)]
        self._rng = np.random.default_rng(self.bo_cfg.random_seed)

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

    def propose(self):
        count = len(self.data)
        low, high = (self.waveform_cfg.minimum_voltage_v,
                     self.waveform_cfg.maximum_voltage_v)
        if count == 0:
            return np.linspace(self.waveform_cfg.linear_minimum_voltage_v,
                               self.waveform_cfg.linear_maximum_voltage_v,
                               self.bo_cfg.rise_points)
        if count <= self.bo_cfg.initial_random_measurements:
            return self._rng.uniform(low, high, self.bo_cfg.rise_points)
        inputs = self._generator.generate(1)[0]
        return np.array([inputs[name] for name in self.input_names], dtype=float)

    def observe(self, proposed_voltages_v, work_j, delta_p_c_per_m2):
        # The GP is fit against what was proposed/commanded, not the
        # independently measured-at-knot-times voltage (see measurement.py's
        # Measurement.measured_control_voltages_v) -- otherwise measurement
        # noise on the inputs would corrupt the surrogate model's input space.
        protocol = np.asarray(proposed_voltages_v, dtype=float)
        if protocol.shape != (self.bo_cfg.rise_points,):
            raise ValueError("Proposed voltage control-point count does not match Xopt")
        work_j, delta_p = float(work_j), float(delta_p_c_per_m2)
        if self.p_target is None:
            # delta_P of the first (linear-rise) evaluation defines P_TARGET.
            # p_lower_relative_tolerance shifts the feasibility boundary
            # relative to P_TARGET, as a fraction of |P_TARGET|: negative
            # LOOSENS the bound (delta_P may fall below P_TARGET by that
            # fraction and still be searched as feasible by BO), positive
            # TIGHTENS it, 0.0 means the BO search boundary sits exactly at
            # P_TARGET.
            self.p_target = delta_p
            tolerance = self.bo_cfg.p_lower_relative_tolerance * abs(self.p_target)
            self.p_lower_bound = self.p_target + tolerance
            self.p_scale = abs(self.p_target) if self.p_target != 0.0 else 1.0
            self.work_scale = abs(work_j) if work_j != 0.0 else 1.0
            self.p_lower_norm_threshold = self.p_lower_bound / self.p_scale
            self._make_generator()
        delta_p_norm = delta_p / self.p_scale
        work_norm = work_j / self.work_scale
        evaluation = len(self.data)
        if evaluation == 0:
            phase, phase_evaluation = "linear rise", 0
        elif evaluation <= self.bo_cfg.initial_random_measurements:
            phase, phase_evaluation = "random initial", evaluation
        else:
            phase = "BO optimization"
            phase_evaluation = evaluation - self.bo_cfg.initial_random_measurements
        row = dict(zip(self.input_names, protocol))
        row.update({
            "Work": work_j,
            "Work_NORM": work_norm,
            "delta_P": delta_p,
            "delta_P_NORM": delta_p_norm,
            "P_target": self.p_target,
            "P_lower_bound": self.p_lower_bound,
            "P_violation": max(self.p_lower_bound - delta_p, 0.0),
            "P_lower": delta_p,
            "P_lower_NORM": delta_p_norm,
            "P_SCALE": self.p_scale,
            "WORK_SCALE": self.work_scale,
            # "feasible" reports whether delta_P reached the true physical
            # target, independent of p_lower_relative_tolerance -- that
            # tolerance only shifts what the BO constraint itself searches
            # for (p_lower_bound above), it does not redefine what counts as
            # a successful result for reporting.
            "feasible": bool(delta_p >= self.p_target),
            "phase": phase,
            "phase_evaluation": phase_evaluation,
        })
        new_data = pd.DataFrame([row], index=[evaluation])
        self.data = pd.concat([self.data, new_data])
        self._generator.add_data(new_data)
        return row
