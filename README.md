# PUND Rise-Shape Bayesian Optimization

Hardware-control software for optimizing the shared voltage-rise shape of a
PUND pulse pair. The experiment minimizes measured ferroelectric switching
work while requiring the polarization change to remain at or above a reference
measurement.

> [!CAUTION]
> This program controls laboratory instruments and can apply voltage to a
> connected device. Review the instrument addresses, channel scaling, current
> conversion, device area, load setting, waveform timing, and voltage limits in
> `config.py` before enabling an output. First validate the waveform and trigger
> behavior on an oscilloscope without a device connected.

## How it works

The arbitrary waveform generator (AWG) produces:

```text
baseline -> negative reset -> gap -> P pulse -> gap -> U pulse -> baseline
```

The P and U pulses use the same shaped rise. Bayesian optimization selects
the rise's control values; the pulse plateau and fall remain fixed.
`PUNDWaveformConfig.rise_basis` sets what those values mean: `"points"`
(default) treats them as voltage knots joined by straight lines, and
`"bernstein"` treats them as coefficients of a smooth Bernstein polynomial
that always stays between the smallest and largest coefficient. The
scope captures both pulses, and the program subtracts their currents
(`I_diff = I_P - I_U`) to estimate switching behavior over the rise window.

For each proposed shape, the program computes:

- `work_j = integral(V_device * I_diff dt)` (objective, minimized)
- `delta_p_c_per_m2 = integral(I_diff dt) / electrode_area_m2` (constraint)

The first evaluation is a linear reference rise. Its measured `delta_P` sets
the constraint target. Random initial evaluations are followed by constrained
Expected Improvement through Xopt.

## Requirements

- Windows with Python 3.10 or newer
- A VISA implementation (for example, Keysight IO Libraries Suite or NI-VISA)
- A VISA-addressable Agilent/Keysight arbitrary waveform generator
- An Agilent/Keysight Infiniium oscilloscope
- FFmpeg if MP4 generation is desired

Install the Python packages in an isolated environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The included `run_in_xopt_env.ps1` is machine-specific and expects an existing
Conda environment at the path declared inside that script. Edit the path or
run `python run_experiment.py` from your own environment.

## Configure the experiment

All experiment settings live in `config.py`:

1. Set `InstrumentConfig.function_generator_address` and
   `oscilloscope_address` to the VISA resources on your system.
2. Verify scope channel assignments, vertical scales, offsets, trigger level,
   averaging, and time step.
3. Set `DeviceConfig.device_name`, `electrode_area_m2`, and the current
   conversion settings. With
   `correct_voltage_for_sense_resistor = True`, the configured sense-resistor
   drop is removed before calculating work.
4. Set `ExperimentConfig.data_mother_path` to the desired output location.
5. Review waveform timing and all voltage values. The current defaults search
   from `0` to `1 V`, use a `-0.8 V` reset, and enforce a `1 V` absolute safety
   ceiling.
6. Choose the number of rise knots and evaluations in `BOConfig`. The defaults
   perform 401 measurements: one linear reference, 100 random measurements,
   and 300 Bayesian-optimization steps.

The configured P and U rises must remain identical for the subtraction to be
meaningful. Cross-check a saved linear-rise measurement with an independently
validated PUND analysis before trusting the reported metrics.

## Run

With the hardware connected and the configuration reviewed:

```powershell
python run_experiment.py
```

The live control panel displays the measured voltage and subtracted current,
the optimization history, and the next proposed rise. Set
`ExperimentConfig.enable_control_panel = False` for headless operation.

Each run receives its own timestamp and output folder. The program saves:

- full captured traces (`trace_*.csv`)
- rise-window traces (`rise_*.csv`)
- compact optimization history (`bo_history_*.csv`)
- the effective run configuration (`run_config_*.json`)
- an evolution video when a compatible Matplotlib writer is available

The history records both proposed knots (`V0`, `V1`, ...) and scope-measured
values at the same times (`Measured_V0`, `Measured_V1`, ...), making DAC,
bandwidth, slew-rate, and cabling differences visible.

Stop the experiment with `Ctrl+C` if the observed waveform or current is
unexpected. Confirm the instrument output is off before touching the setup.

## Analyze an existing run without hardware

The plotting tools only read saved files and do not communicate with the
instruments:

```powershell
python plot_history_traces.py
python make_evolution_video.py
```

With no arguments, each script searches beneath `data_mother_path` for the
newest history file. Target a particular run with either option:

```powershell
python plot_history_traces.py --history D:\path\to\bo_history_TIMESTAMP.csv
python make_evolution_video.py --data-folder D:\path\to\run_folder
```

`plot_history_traces.py` is the quicker feedback loop while changing plot
layouts. Generate the video after the static figures look correct.

## Project layout

| File | Purpose |
| --- | --- |
| `run_experiment.py` | Sequential measurement and optimization loop |
| `config.py` | Instrument, device, waveform, optimizer, and output settings |
| `waveform.py` | PUND waveform construction and AWG normalization |
| `instruments.py` | VISA communication, waveform upload, and scope acquisition |
| `measurement.py` | Applies a proposal and reduces the acquired traces |
| `pulse_subtraction.py` | Aligns/subtracts P and U pulses and calculates metrics |
| `bo.py` | Xopt constrained Expected Improvement workflow |
| `storage.py` | Run configuration, traces, and history persistence |
| `control_panel.py` | Live experiment visualization |
| `plot_history_traces.py` | Static post-run plots |
| `make_evolution_video.py` | Animated optimization history |

## Notes

- The AWG sync output drives the default scope trigger channel. Its positive
  edge marks the AWG-buffer restart, at the beginning of the pre-baseline.
- Raw experimental data is intentionally excluded by `.gitignore`; keep it in
  the configured external data directory.
- Instrument commands and assumptions are tailored to the equipment used to
  develop this project. Verify SCPI compatibility before substituting models.
