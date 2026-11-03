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
from storage import (append_run_log, rewrite_history_references, save_confirmation_summary,
                     save_measurement, save_run_config, save_warmup_measurement)
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
            measurement = panel.run_responsive(apply_rise_and_measure, proposed_voltage, cfg)
            if cfg.save_warmup:
                save_warmup_measurement(measurement, warmup, cfg)
            panel.set_status(f"Linear warm-up {warmup}/{cfg.warmup_repeats}")
            if panel.stop_requested:
                print(f"Stop requested -- ending during warm-up ({warmup}/{cfg.warmup_repeats}), "
                      "before any measurement was saved.")
                break
    planned = 0 if panel.stop_requested else cfg.bo.total_measurements
    iteration = 0
    # Extra random measurements (no feasible point yet, see bo.py) extend the run.
    while iteration < planned + optimizer.extra_random_measurements:
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
        iteration += 1

    if (cfg.bo.mode == "optimize" and cfg.bo.confirm_top > 0
            and not panel.stop_requested and cfg.history_file.exists()):
        confirm_top_protocols(cfg, optimizer, panel, iteration)

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


def confirm_top_protocols(cfg: ExperimentConfig, optimizer: BayesianRiseOptimizer,
                          panel: ControlPanel, iteration: int) -> None:
    """Re-measure BO's best few protocols (see BOConfig.confirm_top) and write
    confirm_summary_<timestamp>.csv ranking them by mean Work."""
    candidates = optimizer.confirmation_candidates()
    if candidates.empty:
        print("No feasible protocol to confirm -- skipping the confirmation phase.")
        return
    plan = optimizer.confirmation_plan(candidates)
    print(f"\nConfirming the top {len(candidates)} protocol(s) (evaluations "
          f"{', '.join(map(str, candidates['source_evaluation']))}): "
          f"{cfg.bo.confirm_repeats} rounds, {len(plan)} measurements")
    for step, (protocol, candidate) in enumerate(plan):
        label = "linear reference" if candidate is None else f"candidate {candidate}"
        print(f"Iteration {iteration}: confirmation {step + 1}/{len(plan)}, {label}")
        measurement = panel.run_responsive(apply_rise_and_measure, protocol, cfg)
        result = optimizer.observe_confirmation(
            measurement.proposed_control_voltages_v,
            measurement.work_j,
            measurement.delta_p_c_per_m2,
            candidate,
        )
        save_measurement(measurement, iteration, result, cfg)
        if result["phase"] in LINEAR_PHASES:
            rewrite_history_references(cfg, optimizer.data)
        print(f"Work={result['Work']:.8g} J, delta_P={result['delta_P']:.8g} C/m^2, "
              f"feasible={result['feasible']}")
        next_protocol = plan[step + 1][0] if step + 1 < len(plan) else protocol
        panel.update(measurement, result, next_protocol, iteration)
        iteration += 1
        if panel.stop_requested:
            print("Stop requested -- ending the confirmation phase early.")
            break
    summary = optimizer.confirmation_summary(candidates)
    path = save_confirmation_summary(cfg, summary)
    print(summary[["rank", "candidate", "source_evaluation", "bo_Work_NORM", "n", "Work_NORM_mean",
                   "Work_NORM_std", "delta_P_NORM_mean", "feasible_fraction"]].to_string(index=False))
    best = summary.iloc[0]
    print(f"Confirmed best: candidate {int(best['candidate'])} (evaluation "
          f"{int(best['source_evaluation'])}), Work reduction {best['work_reduction_pct']:.1f} "
          f"+/- {100 * best['Work_NORM_sem']:.1f} % (mean +/- s.e.m.)"
          + ("" if best["meets_target"] else " -- NONE reached the delta_P target on average"))
    print(f"Confirmation summary saved to {path}")


def main() -> None:
    run_once(ExperimentConfig())


if __name__ == "__main__":
    main()
