"""Sample-drift test: repeatedly apply the SAME fixed linear voltage ramp
(no BO, no search) in BATCHES, tracking how Work and delta_P drift over
repeated identical stress. Edit the settings below, then run this file.

Protocol: MEASUREMENTS_PER_BATCH measurements, INTRA_BATCH_DELAY_S apart,
then an INTER_BATCH_REST_S rest, then the next batch -- repeated for
NUM_BATCHES batches (None = run batches until interrupted with Ctrl+C).

The ramp used is np.linspace(linear_minimum_voltage_v, linear_maximum_voltage_v,
rise_points) from config.py -- the same "initial linear rise" run_experiment.py
uses as its very first BO evaluation.
"""

import csv
import dataclasses
import datetime as dt
import time

import numpy as np

from config import ExperimentConfig
from measurement import apply_rise_and_measure

# Measurements taken back-to-back within one batch.
MEASUREMENTS_PER_BATCH: int = 200
# Pause between consecutive measurements within a batch, in seconds.
INTRA_BATCH_DELAY_S: float = 2.0
# Rest between batches, in seconds (e.g. to see recovery after stress). This is
# the rest AFTER batch 0; it then grows by REST_GROWTH_FACTOR each subsequent
# batch (1.0 = constant rest, as before; e.g. 2.0 doubles it every batch:
# 60s, 120s, 240s, ...), capped at REST_MAX_S (None = uncapped).
INITIAL_REST_S: float = 10
REST_GROWTH_FACTOR: float = 2
REST_MAX_S: float | None = None
# Number of batches to run. None runs batches until interrupted (Ctrl+C).
NUM_BATCHES: int | None = 5
DRIFT_NAME = "drift_test"


def rest_duration_s(batches_completed: int) -> float:
    """Rest before the batch after `batches_completed` full batches, i.e. 0 ->
    the rest after batch 0, 1 -> the rest after batch 1, etc."""
    duration = INITIAL_REST_S * (REST_GROWTH_FACTOR ** batches_completed)
    return duration if REST_MAX_S is None else min(duration, REST_MAX_S)


def linear_ramp(cfg: ExperimentConfig) -> np.ndarray:
    return np.linspace(cfg.waveform.linear_minimum_voltage_v,
                        cfg.waveform.linear_maximum_voltage_v,
                        cfg.bo.rise_points)


def main() -> None:
    cfg = ExperimentConfig()
    cfg = dataclasses.replace(cfg, data_mother_path=cfg.data_mother_path / DRIFT_NAME)
    output_dir = cfg.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"drift_log_{cfg.run_timestamp}.csv"

    voltages = linear_ramp(cfg)
    print(f"Drift test: repeatedly applying fixed linear ramp {voltages}.")
    print(f"Batches of {MEASUREMENTS_PER_BATCH}, {INTRA_BATCH_DELAY_S}s apart, "
          f"with a growing rest between batches (starts at {INITIAL_REST_S}s, "
          f"x{REST_GROWTH_FACTOR} per batch"
          + (f", capped at {REST_MAX_S}s" if REST_MAX_S is not None else "") + ").")
    print(f"Logging to {log_path}")

    figure = ax_work = ax_delta_p = plt = None
    if cfg.enable_control_panel:
        import matplotlib.pyplot as plt
        plt.ion()
        figure, (ax_work, ax_delta_p) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        figure.canvas.manager.set_window_title("Drift test")

    works, deltas_p, elapsed_s = [], [], []
    batch_start_times = []
    start = time.monotonic()

    def refresh_plot():
        if plt is None:
            return
        ax_work.clear()
        ax_work.plot(elapsed_s, works, marker="o", color="tab:blue")
        ax_work.set(title="Work drift", ylabel="Work (J)")
        ax_work.grid(True, alpha=0.3)
        ax_delta_p.clear()
        ax_delta_p.plot(elapsed_s, deltas_p, marker="o", color="tab:purple")
        ax_delta_p.set(title="delta_P drift", xlabel="Elapsed time (s)", ylabel="delta_P (C/m^2)")
        ax_delta_p.grid(True, alpha=0.3)
        for axis in (ax_work, ax_delta_p):
            for batch_time in batch_start_times[1:]:
                axis.axvline(batch_time, color="0.6", linestyle="--", linewidth=1)
        figure.tight_layout()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.pause(0.05)

    with log_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(["batch", "batch_iteration", "iteration", "timestamp",
                             "elapsed_s", "Work", "delta_P"])

        iteration = 0
        batch = 0
        try:
            while NUM_BATCHES is None or batch < NUM_BATCHES:
                batch_start_times.append(time.monotonic() - start)
                print(f"\n--- Batch {batch} ---")
                for batch_iteration in range(MEASUREMENTS_PER_BATCH):
                    measurement = apply_rise_and_measure(voltages, cfg)
                    now = time.monotonic() - start
                    print(f"[batch {batch}, {batch_iteration}] t={now:.1f}s "
                          f"Work={measurement.work_j:.6g} J "
                          f"delta_P={measurement.delta_p_c_per_m2:.6g} C/m^2")
                    writer.writerow([batch, batch_iteration, iteration,
                                     dt.datetime.now().isoformat(timespec="seconds"),
                                     f"{now:.3f}", measurement.work_j, measurement.delta_p_c_per_m2])
                    handle.flush()

                    works.append(measurement.work_j)
                    deltas_p.append(measurement.delta_p_c_per_m2)
                    elapsed_s.append(now)
                    refresh_plot()

                    iteration += 1
                    is_last_in_batch = batch_iteration == MEASUREMENTS_PER_BATCH - 1
                    if not is_last_in_batch and INTRA_BATCH_DELAY_S > 0:
                        time.sleep(INTRA_BATCH_DELAY_S)

                rest_s = rest_duration_s(batch)
                batch += 1
                if (NUM_BATCHES is None or batch < NUM_BATCHES) and rest_s > 0:
                    print(f"Resting {rest_s:.1f}s before next batch...")
                    time.sleep(rest_s)
        except KeyboardInterrupt:
            print("\nDrift test stopped by user.")

    if plt is not None:
        figure.suptitle("Drift test complete - close this window to exit")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
