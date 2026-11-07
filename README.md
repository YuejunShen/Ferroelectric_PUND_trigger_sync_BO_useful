# PUND Rise-Shape Bayesian Optimization

Instrument-control software that searches for the voltage-rise shape of a PUND
pulse pair that switches a ferroelectric capacitor with the least work, while
switching at least as much polarization as a linear rise.

> [!CAUTION]
> This program drives laboratory instruments and applies voltage to a connected
> device. Before enabling an output, review in `config.py` the instrument
> addresses, channel scaling, current gain, electrode area, waveform timing and
> voltage limits. Validate the waveform and trigger on a scope without a device first.

## How it works

The AWG (Keysight 33500B series, true ARB mode, `sample_rate_hz`) plays one period:

```text
pre-baseline -> reset (-amplitude) -> gap1 -> P pulse -> gap2 -> U pulse -> post-baseline
P, U pulse = shaped rise (rise_time_s) -> flat top (+amplitude, pulse_width_s) -> linear fall
```

- P and U share the identical rise. Only that rise is optimized; the reset,
  flat tops and falls are fixed (`amplitude_v` sets reset, flat top and the end of the
  linear reference rise; `other_ramp_time_s` sets the non-optimized ramps).
- The rise's control values (`BOConfig.rise_points` of them) are, by
  `rise_basis`, either voltage knots joined linearly (`"points"`) or
  Bernstein-polynomial coefficients (`"bernstein"`, smooth, bounded by the
  coefficients). The rise is concatenated straight into the flat top, so its
  last value does not have to equal the amplitude.
- The AWG SYNC output triggers the scope (channel 3); its edge is t = 0. The
  scope averages `average_count` periods. The program subtracts the U current
  from the P current over the rise window `[rise start, rise start + rise_time_s]`:
  - **Work** `= ∫ V·(I_P − I_U) dt` — objective, minimized. V has the
    sense-resistor IR drop removed if `correct_voltage_for_sense_resistor`.
  - **ΔP** `= ∫ (I_P − I_U) dt / electrode_area_m2` — constraint.

### One run (`run_experiment.run_once`)

1. **Warm-up:** `warmup_repeats` linear rises to settle the device
   (saved to `warmup/` if `save_warmup`).
2. **Initial phase:** `linear_repeats` linear rises (their mean ΔP is the target)
   mixed with `initial_random_measurements` random shapes (`random_design="spread"`
   spans the whole voltage range).
3. **BO phase:** `bo_steps` steps of Xopt constrained Expected Improvement,
   constrained to ΔP ≥ target × (1 + `p_lower_relative_tolerance`), so proposals land above the target.
   - **Drift correction:** a linear reference rise is measured every
     `reference_interval` steps, and each row is normalized against the linear rises
     around it (`reference_window`).
   - **No feasible point yet:** extra random shapes are measured, up to
     `max_extra_random_measurements`; if none meets the bound, it is lowered to the closest ΔP reached.
4. **Confirmation:** the `confirm_top` lowest-Work distinct protocols with ΔP ≥ target
   (the target itself; the tolerance above only steers BO), topped up if needed with those below the
   target by at most `confirm_noise_sd` s.d. of the repeated linear rises' ΔP (measurement noise) that
   need less Work than linear, closest first (`selected_by`), are re-measured `confirm_repeats` times
   (a candidate counts if its mean ΔP ≥ target − the same noise allowance), in shuffled order with one linear reference per round. They are ranked by
   mean Work in `confirm_summary_*.csv`. This removes the low bias of a single noisy minimum.
5. **Post-run outputs:** an evolution video and a PCA landscape plot (plus a Pareto plot in `"pareto"` mode).

`BOConfig.mode`:

| Mode | What it does |
| --- | --- |
| `"optimize"` | Constrained minimization (the default). |
| `"pareto"` | Minimizes Work and maximizes ΔP together. |
| `"explore"` | Maps the space without optimizing. |

Default run: 150 warm-up + 315 BO-run + 31 confirmation measurements, about 40 min.

## Install

Requirements:
- Windows with Python ≥ 3.10.
- VISA (Keysight IO Libraries or NI-VISA).
- A Keysight 33500B-series AWG and an Infiniium scope.
- FFmpeg, for MP4 videos.

```powershell
python -m pip install -r requirements.txt   # matplotlib numpy packaging pandas pyvisa xopt
```

`run_in_xopt_env.ps1` runs `run_experiment.py` in the lab's Conda environment
(`bto-xopt-latest`); edit its path for another machine.

## Configure (`config.py`)

| Section | Key settings |
| --- | --- |
| `InstrumentConfig` | VISA addresses, channels (1 current, 2 voltage, 3 trigger), vertical scales, `average_count`, `scope_time_step_s` |
| `DeviceConfig` | `device_name`, `electrode_area_m2`, current gain (50 Ω sense), IR-drop correction |
| `PUNDWaveformConfig` | `sample_rate_hz`, `rise_time_s`, `pulse_width_s`, `gap1_s`/`gap2_s`, baselines, `amplitude_v`, `rise_basis`, BO voltage bounds, `safety_max_voltage_v`, `maximum_arb_points` |
| `BOConfig` | `mode`, `rise_points`, phase sizes (above), drift reference, confirmation, tolerance, GP options |
| `ExperimentConfig` | `data_mother_path` (`D:\Ferroelectric_BTO_optimization\<date>`), `warmup_repeats`, `save_warmup`, control panel |

Rules for the settings:
- Keep `rise_time_s` and every gap a whole number of samples at `sample_rate_hz`
  (multiples of 10 ns at 100 MSa/s). The waveform is built from rounded sample counts, but the
  analysis windows use the nominal times, so a fractional sample misaligns U
  against P and leaves an edge spike in I1−I2.
- Keep `responsive_control_panel = False` for long unattended sweeps; the
  worker-thread panel crashed Tk. "Stop safely" still works between measurements.

## Scripts

Each script has its settings at the top; edit them, then run it.

| Script | Purpose |
| --- | --- |
| `run_experiment.py` | One BO run with `config.py`. |
| `run_parameter_sweep.py` | Grid sweep. `SWEEP` maps dotted config paths (e.g. `"waveform.rise_time_s"`) to a value or a list/array of values. Every combination runs `REPEATS` times (repeats as the outer loop) into `<date>\SWEEP_NAME\`. Resume with `START_RUN`/`RESUME_FOLDER`. At the end it calls `summarize_sweep` and the convergence plot. |
| `run_drift_test.py` | The same linear rise repeated in batches with growing rests, to track Work/ΔP drift and recovery (no BO). |
| `run_drift_sweep.py` | Several drift tests back to back (`"drift.<field>"` keys allowed), with an output-off wait between them. |
| `run_relaxation_test.py` | ΔP vs `gap1 = gap2` (reset→P = P→U) for each amplitude in `AMPLITUDES_V` (±flat top/reset). For each amplitude it runs its own warm-up, then `GAPS_S` (rounded to whole samples) × `ROUNDS`, shuffled per round, with the linear rise or `PROTOCOL_V`. Writes `relaxation_log_*.csv` and `relaxation_delta_p_vs_gap_{logx,logy,loglog}.png` (one curve per amplitude); `--replot <log>` replots. |

Every BO run (single or in a sweep) appends one row to `D:\Ferroelectric_BTO_optimization\run_log.csv`.

## Outputs (per run folder `BO_pund_rise_<device>_<HHMM>[_<sweep tag>]`)

| File | Contents |
| --- | --- |
| `run_config_*.json` | Every setting used. |
| `bo_history_*.csv` | One row per measurement: commanded `V0..Vn`, scope-measured `Measured_V0..Vn`, Work, ΔP, normalized values, feasibility, `phase`, confirmation `candidate`. |
| `trace_*_iteration_NNN.csv` | Full scope trace: time, current, voltage (used / raw / IR drop). |
| `rise_*_iteration_NNN.csv` | Rise window, 1000 points: voltage, I1−I2. |
| `confirm_summary_*.csv` | Ranked confirmed protocols. |
| `warmup/` | Warm-up traces and the `warmup_*.csv` table. |
| `*.mp4`, `*_landscape.png` | Evolution video, PCA landscape. |

## Analysis (no hardware needed)

| Script | Output |
| --- | --- |
| `plot_sweep_results.py --sweep-folder <folder> [--x … --series …]` | Best confirmed protocol per setting. Writes `sweep_all_runs.csv` and `sweep_best_per_setting.csv`. Plots: Work reduction vs x; absolute Work (log-log with power-law fit, written to `sweep_power_law_fit.csv`) and ΔP vs x; best-protocol V, I1−I2 and P(t) vs time and vs t/τ (20 ns cut at each window end). |
| `plot_sweep_convergence.py` | Work ratio vs a swept `run_log.csv` column. |
| `plot_sweep_figure.py --sweep-folder <folder> [--output-dir <folder>]` | Publication sweep figure plus the polynomial-coefficient distributions below. |
| `plot_sweep_polynomial_coefficients.py --sweep-folder <folder> [--output-dir <folder>]` | One panel per power order, showing coefficient distributions across confirmed best protocols at each rise time. Exact Bernstein-to-power conversion: `V(s) = sum(a_k * s**k)`, `s = t/tau`; all `a_k` are in volts. Dots are independent runs; boxes show median and IQR, whiskers the full range. Saves PDF/SVG/PNG and per-run/summary CSV tables. |
| `plot_history_traces.py`, `make_evolution_video.py` | All traces of one run, as static plots or video. |
| `plot_landscape.py` | Work/ΔP on the top-2 principal components of `V0..Vn`. |
| `plot_pareto.py` | Work vs ΔP with the measured Pareto front. |
| `plot_drift_test.py` | Drift plot and I1−I2 video from `drift_log_*.csv`. |

The single-run scripts take `--history <bo_history csv>` or `--data-folder <run folder>`; without arguments they use the newest history they find (see each script's `--help`).

## Code layout

| File | Role |
| --- | --- |
| `config.py` | All settings (frozen dataclasses). |
| `waveform.py` | Building the PUND ARB waveform: `build_pund_waveform`, `rise_voltage`, `positive_rise_windows`. |
| `instruments.py` | AWG upload/output (`ArbGenerator`), scope setup and readout. |
| `measurement.py` | `apply_rise_and_measure`: upload, capture, align to trigger, subtract, Work/ΔP. |
| `pulse_subtraction.py` | P−U subtraction on the rise window, Work/ΔP integrals. |
| `bo.py` | `BayesianRiseOptimizer`: phases, drift reference, Xopt constrained EI, confirmation. |
| `storage.py` | Run config, history, traces, warm-up, run log. |
| `control_panel.py` | Live panel with a "Stop safely" button. |

## Known caveats

- **End of the rise window:** the window ends exactly where the AWG steps
  from the rise's last value to the flat top. The finite edge (about 10–20 ns) leaks into
  the last points, and into the first points when the rise starts above 0 V.
  This changes Work by up to about 2 % and ΔP by up to about 0.6 % for non-linear protocols.
- **Data location:** raw data stays outside the repository (`.gitignore`). The SCPI commands assume
  the instruments listed above.
