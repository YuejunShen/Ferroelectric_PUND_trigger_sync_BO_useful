"""Sequential BO measurement loop. Edit config.py, then run this file."""

import numpy as np

from bo import BayesianRiseOptimizer
from config import ExperimentConfig
from control_panel import ControlPanel
from make_evolution_video import default_output_path, make_video
from measurement import apply_rise_and_measure
from storage import save_measurement, save_run_config


def main() -> None:
    cfg = ExperimentConfig()
    save_run_config(cfg)
    optimizer = BayesianRiseOptimizer(cfg.waveform, cfg.bo)
    panel = ControlPanel(cfg)
    proposed_voltage = optimizer.propose()
    panel.show_next(proposed_voltage, "Initial linear rise")
    for iteration in range(cfg.bo.total_measurements):
        print(f"Iteration {iteration}: applying {proposed_voltage}")
        measurement = apply_rise_and_measure(proposed_voltage, cfg)
        result = optimizer.observe(
            measurement.proposed_control_voltages_v,
            measurement.work_j,
            measurement.delta_p_c_per_m2,
        )
        save_measurement(measurement, iteration, result, cfg)
        knot_error = np.abs(measurement.measured_control_voltages_v
                            - measurement.proposed_control_voltages_v)
        print(f"Work={result['Work']:.8g} J, delta_P={result['delta_P']:.8g} C/m^2, "
              f"feasible={result['feasible']}, phase={result['phase']}, "
              f"max|measured-proposed|={knot_error.max():.4g} V")
        proposed_voltage = optimizer.propose()
        panel.update(measurement, result, proposed_voltage, iteration)

    try:
        video_path = make_video(cfg.history_file, default_output_path(cfg.history_file),
                                cfg.device.electrode_area_m2, cfg.waveform)
        print(f"Evolution video saved to {video_path}")
    except Exception as exc:
        print(f"Could not generate evolution video: {exc}")

    panel.finish()


if __name__ == "__main__":
    main()
