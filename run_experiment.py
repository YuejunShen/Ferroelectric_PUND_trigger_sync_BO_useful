"""Sequential BO measurement loop. Edit config.py, then run this file.

For a hyperparameter convergence test (multiple parameter values, each
possibly repeated), use run_parameter_sweep.py instead -- it calls run_once()
below for every combination."""

import numpy as np

from bo import LINEAR_PHASES, BayesianRiseOptimizer
from config import ExperimentConfig
from control_panel import ControlPanel
from make_evolution_video import default_output_path, make_video
from measurement import apply_rise_and_measure
from plot_landscape import plot_landscape
from plot_pareto import plot_pareto
from storage import append_run_log, rewrite_history_references, save_measurement, save_run_config
from waveform import rise_voltage


def run_once(cfg: ExperimentConfig, sweep_label: str = "", sweep_overrides: dict | None = None,
             repeat_index: int | None = None, block_on_finish: bool = True) -> bool:
    """Returns True if the user clicked "Stop safely" on the control panel
    during this run (checked between iterations, never mid-measurement) --
    run_parameter_sweep.py uses this to stop the whole sweep rather than
    launching another run."""
    save_run_config(cfg)
    append_run_log(cfg, sweep_label=sweep_label, sweep_overrides=sweep_overrides, repeat_index=repeat_index)
    optimizer = BayesianRiseOptimizer(cfg.waveform, cfg.bo)
    panel = ControlPanel(cfg)
    proposed_voltage = optimizer.propose()
    panel.show_next(proposed_voltage, "Initial linear rise")
    if cfg.warmup_repeats > 0:
        print(f"Applying linear warm-up rise {cfg.warmup_repeats} time(s) before measuring...")
        for warmup in range(1, cfg.warmup_repeats + 1):
            panel.run_responsive(apply_rise_and_measure, proposed_voltage, cfg)
            panel.set_status(f"Linear warm-up {warmup}/{cfg.warmup_repeats}")
            if panel.stop_requested:
                print(f"Stop requested -- ending during warm-up ({warmup}/{cfg.warmup_repeats}), "
                      "before any measurement was saved.")
                break
    for iteration in range(0 if panel.stop_requested else cfg.bo.total_measurements):
        print(f"Iteration {iteration}: applying {proposed_voltage}")
        measurement = panel.run_responsive(apply_rise_and_measure, proposed_voltage, cfg)
        result = optimizer.observe(
            measurement.proposed_control_voltages_v,
            measurement.work_j,
            measurement.delta_p_c_per_m2,
        )
        save_measurement(measurement, iteration, result, cfg)
        if result["phase"] in LINEAR_PHASES:
            rewrite_history_references(cfg, optimizer.data)
        proposed = measurement.proposed_control_voltages_v
        commanded_at_knots = rise_voltage(proposed, np.linspace(0.0, 1.0, proposed.size),
                                          cfg.waveform.rise_basis)
        knot_error = np.abs(measurement.measured_control_voltages_v - commanded_at_knots)
        print(f"Work={result['Work']:.8g} J, delta_P={result['delta_P']:.8g} C/m^2, "
              f"feasible={result['feasible']}, phase={result['phase']}, "
              f"max|measured-proposed|={knot_error.max():.4g} V")
        proposed_voltage = panel.run_responsive(optimizer.propose)
        panel.update(measurement, result, proposed_voltage, iteration)

        if panel.stop_requested:
            print(f"Stop requested -- finishing cleanly after iteration {iteration}.")
            break

    if not cfg.history_file.exists():
        # Stopped during warm-up: nothing was measured, so nothing to plot.
        panel.finish(block=block_on_finish)
        return panel.stop_requested

    try:
        video_path = make_video(cfg.history_file, default_output_path(cfg.history_file),
                                cfg.device.electrode_area_m2, cfg.waveform)
        print(f"Evolution video saved to {video_path}")
    except Exception as exc:
        print(f"Could not generate evolution video: {exc}")

    try:
        landscape_path = plot_landscape(cfg.history_file)
        print(f"Landscape plot saved to {landscape_path}")
    except Exception as exc:
        print(f"Could not generate landscape plot: {exc}")

    if cfg.bo.mode == "pareto":
        try:
            pareto_path = plot_pareto(cfg.history_file)
            print(f"Pareto front plot saved to {pareto_path}")
        except Exception as exc:
            print(f"Could not generate Pareto front plot: {exc}")

    panel.finish(block=block_on_finish)
    return panel.stop_requested


def main() -> None:
    run_once(ExperimentConfig())


if __name__ == "__main__":
    main()
