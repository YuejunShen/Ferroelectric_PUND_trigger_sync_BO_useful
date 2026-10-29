"""Sample-drift test: repeatedly apply the SAME fixed linear voltage ramp
(no BO, no search) in BATCHES, tracking how Work and delta_P drift over
repeated identical stress. Edit the settings below, then run this file.

Protocol: MEASUREMENTS_PER_BATCH measurements, INTRA_BATCH_DELAY_S apart,
then an INTER_BATCH_REST_S rest, then the next batch -- repeated for
NUM_BATCHES batches (None = run batches until interrupted with Ctrl+C).

The ramp used is np.linspace(linear_minimum_voltage_v, linear_maximum_voltage_v,
rise_points) from config.py -- the same "initial linear rise" run_experiment.py
uses as its very first BO evaluation.

To run several drift tests back to back with different settings, use
run_drift_sweep.py, which calls run_drift_once() below once per setting.
"""

import csv
import dataclasses
import datetime as dt
import json
import time

import numpy as np

from config import ExperimentConfig
from instruments import ArbGenerator
from measurement import apply_rise_and_measure, Measurement
from plot_drift_test import default_video_output_path, make_drift_video, plot_drift
from storage import save_run_config

# Measurements taken back-to-back within one batch.
MEASUREMENTS_PER_BATCH: int = 100
# Pause between consecutive measurements within a batch, in seconds.
INTRA_BATCH_DELAY_S: float = 2.0
# Rest between batches, in seconds (e.g. to see recovery after stress). This is
# the rest AFTER batch 0; it then grows by REST_GROWTH_FACTOR each subsequent
# batch (1.0 = constant rest, as before; e.g. 2.0 doubles it every batch:
# 60s, 120s, 240s, ...), capped at REST_MAX_S (None = uncapped).
INITIAL_REST_S: float = 10
REST_GROWTH_FACTOR: float = 4
REST_MAX_S: float | None = None
# Number of batches to run. None runs batches until interrupted (Ctrl+C).
NUM_BATCHES: int | None = 6
DRIFT_NAME = "drift_test"


@dataclasses.dataclass(frozen=True)
class DriftProtocol:
    """The batch sizing/timing settings above, bundled so run_drift_sweep.py
    can override them per run (as "drift.<field>" sweep keys)."""
    measurements_per_batch: int = MEASUREMENTS_PER_BATCH
    intra_batch_delay_s: float = INTRA_BATCH_DELAY_S
    initial_rest_s: float = INITIAL_REST_S
    rest_growth_factor: float = REST_GROWTH_FACTOR
    rest_max_s: float | None = REST_MAX_S
    num_batches: int | None = NUM_BATCHES

    def rest_duration_s(self, batches_completed: int) -> float:
        """Rest before the batch after `batches_completed` full batches, i.e. 0 ->
        the rest after batch 0, 1 -> the rest after batch 1, etc."""
        duration = self.initial_rest_s * (self.rest_growth_factor ** batches_completed)
        return duration if self.rest_max_s is None else min(duration, self.rest_max_s)


def linear_ramp(cfg: ExperimentConfig) -> np.ndarray:
    return np.linspace(cfg.waveform.linear_minimum_voltage_v,
                        cfg.waveform.linear_maximum_voltage_v,
                        cfg.bo.rise_points)


def save_trace(measurement: Measurement, cfg: ExperimentConfig, batch: int, batch_iteration: int) -> None:
    """Raw scope trace + rise-window trace for one measurement, same columns
    as storage.save_measurement's per-iteration CSVs, so drift-test traces can
    be inspected the same way as a normal BO run's."""
    tag = f"{cfg.run_timestamp}_batch{batch:03d}_meas{batch_iteration:03d}"
    trace_path = cfg.output_directory / f"trace_{tag}.csv"
    with trace_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "current_a", "voltage_v", "voltage_measured_v", "voltage_resistor_v"])
        writer.writerows(zip(measurement.time_s, measurement.current_a, measurement.voltage_v,
                             measurement.voltage_measured_v, measurement.voltage_resistor_v))
    rise_path = cfg.output_directory / f"rise_{tag}.csv"
    with rise_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rise_time_s", "rise_voltage_v", "rise_voltage_measured_v",
                         "rise_voltage_resistor_v", "rise_current_diff_a"])
        writer.writerows(zip(measurement.rise_time_s, measurement.rise_voltage_v,
                             measurement.rise_voltage_measured_v, measurement.rise_voltage_resistor_v,
                             measurement.rise_current_diff_a))


def save_drift_params(cfg: ExperimentConfig, protocol: DriftProtocol, voltages: np.ndarray) -> None:
    """Every drift-protocol setting (batch sizing/timing) plus the actual
    fixed ramp applied, alongside save_run_config()'s instrument/waveform/
    BO/device dump -- together these fully document one drift-test run."""
    params_path = cfg.output_directory / f"drift_params_{cfg.run_timestamp}.json"
    payload = dataclasses.asdict(protocol)
    payload["linear_ramp_voltages_v"] = voltages.tolist()
    with params_path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def run_drift_once(cfg: ExperimentConfig, protocol: DriftProtocol = DriftProtocol(),
                   block_on_finish: bool = True) -> bool:
    """One full drift test (all batches + rests) with `cfg`/`protocol`, saving
    into cfg.output_directory. Returns True if the user stopped it early
    ("Stop safely" button or Ctrl+C) -- run_drift_sweep.py uses this to end
    the whole sweep rather than moving on to the next setting."""
    output_dir = cfg.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"drift_log_{cfg.run_timestamp}.csv"
    rests_path = output_dir / f"drift_rests_{cfg.run_timestamp}.csv"

    voltages = linear_ramp(cfg)
    save_run_config(cfg)
    save_drift_params(cfg, protocol, voltages)
    print(f"Drift test: repeatedly applying fixed linear ramp {voltages}.")
    print(f"Batches of {protocol.measurements_per_batch}, {protocol.intra_batch_delay_s}s apart, "
          f"with a growing rest between batches (starts at {protocol.initial_rest_s}s, "
          f"x{protocol.rest_growth_factor} per batch"
          + (f", capped at {protocol.rest_max_s}s" if protocol.rest_max_s is not None else "") + ").")
    print(f"Logging to {log_path}")
    print(f"Run config + drift params saved under {output_dir}")

    stop_requested = [False]

    figure = ax_work = ax_delta_p = plt = None
    if cfg.enable_control_panel:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        plt.ion()
        figure, (ax_work, ax_delta_p) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        figure.subplots_adjust(bottom=0.18)
        figure.canvas.manager.set_window_title(
            f"Drift test - {cfg.sweep_tag}" if cfg.sweep_tag else "Drift test")

        def _on_stop_clicked(_event):
            stop_requested[0] = True
            stop_button.label.set_text("Stopping after this\nmeasurement...")
            stop_button.ax.set_facecolor("0.8")
            figure.canvas.draw_idle()

        button_ax = figure.add_axes([0.35, 0.02, 0.3, 0.07])
        stop_button = Button(button_ax, "Stop safely", color="tab:red", hovercolor="0.9")
        stop_button.on_clicked(_on_stop_clicked)

    works, deltas_p, elapsed_s = [], [], []
    rest_intervals = []
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
            y_low, y_high = axis.get_ylim()
            label_y = y_low + 0.97 * (y_high - y_low)
            for index, (rest_start, rest_end) in enumerate(rest_intervals):
                axis.axvspan(rest_start, rest_end, color="0.6", alpha=0.3,
                            label="Rest" if index == 0 else None)
                duration = rest_end - rest_start
                axis.text((rest_start + rest_end) / 2.0, label_y, f"{duration:.0f}s",
                          ha="center", va="top", fontsize=8, color="0.3")
            axis.set_ylim(y_low, y_high)
            if rest_intervals:
                axis.legend(loc="upper left", fontsize=8)
        figure.tight_layout()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        plt.pause(0.05)

    rests_handle = rests_path.open("a", newline="")
    rests_writer = csv.writer(rests_handle)
    if rests_handle.tell() == 0:
        rests_writer.writerow(["batch_before", "rest_start_s", "rest_end_s", "planned_rest_s", "completed",
                               "output_off_confirmed_at_start", "output_on_at_end"])

    with log_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if handle.tell() == 0:
            writer.writerow(["batch", "batch_iteration", "iteration", "timestamp",
                             "elapsed_s", "Work", "delta_P", "trace_file", "rise_file"])

        iteration = 0
        batch = 0
        try:
            while (protocol.num_batches is None or batch < protocol.num_batches) and not stop_requested[0]:
                print(f"\n--- Batch {batch} ---")
                for batch_iteration in range(protocol.measurements_per_batch):
                    # apply_rise_and_measure() owns the AWG/scope connections
                    # for exactly this one measurement and always closes them
                    # (its own try/finally) before returning -- so the stop
                    # check below only ever fires between measurements, never
                    # mid-measurement, and the instruments are always left in
                    # a clean, reconnectable state.
                    measurement = apply_rise_and_measure(voltages, cfg)
                    now = time.monotonic() - start
                    print(f"[batch {batch}, {batch_iteration}] t={now:.1f}s "
                          f"Work={measurement.work_j:.6g} J "
                          f"delta_P={measurement.delta_p_c_per_m2:.6g} C/m^2")
                    save_trace(measurement, cfg, batch, batch_iteration)
                    tag = f"{cfg.run_timestamp}_batch{batch:03d}_meas{batch_iteration:03d}"
                    writer.writerow([batch, batch_iteration, iteration,
                                     dt.datetime.now().isoformat(timespec="seconds"),
                                     f"{now:.3f}", measurement.work_j, measurement.delta_p_c_per_m2,
                                     f"trace_{tag}.csv", f"rise_{tag}.csv"])
                    handle.flush()

                    works.append(measurement.work_j)
                    deltas_p.append(measurement.delta_p_c_per_m2)
                    elapsed_s.append(now)
                    refresh_plot()

                    iteration += 1
                    if stop_requested[0]:
                        print("Stop requested -- finishing cleanly after this measurement.")
                        break
                    is_last_in_batch = batch_iteration == protocol.measurements_per_batch - 1
                    if not is_last_in_batch and protocol.intra_batch_delay_s > 0:
                        time.sleep(protocol.intra_batch_delay_s)

                if stop_requested[0]:
                    break

                rest_s = protocol.rest_duration_s(batch)
                batch_before_rest = batch
                batch += 1
                if (protocol.num_batches is None or batch < protocol.num_batches) and rest_s > 0:
                    print(f"Resting {rest_s:.1f}s before next batch...")
                    # Confirmed (read-back) off at the start of the rest, and
                    # re-queried at its end, both logged -- so the rests CSV
                    # itself proves the device was not driven during the rest.
                    ArbGenerator(cfg).output(False)
                    rest_start = time.monotonic() - start
                    # Deadline-based: plt.pause() takes far longer than its
                    # nominal 1 ms, so counting fixed 0.1 s sleeps made rests
                    # run ~1.5x longer than planned.
                    rest_deadline = time.monotonic() + rest_s
                    while not stop_requested[0] and time.monotonic() < rest_deadline:
                        time.sleep(min(0.1, max(0.0, rest_deadline - time.monotonic())))
                        if plt is not None:
                            plt.pause(0.001)
                    rest_end = time.monotonic() - start
                    output_on_at_end = ArbGenerator(cfg).output_enabled()
                    if output_on_at_end:
                        print("WARNING: AWG output was found ON at the end of this rest.")
                    rest_intervals.append((rest_start, rest_end))
                    rests_writer.writerow([batch_before_rest, f"{rest_start:.3f}", f"{rest_end:.3f}",
                                          rest_s, not stop_requested[0], True, output_on_at_end])
                    rests_handle.flush()
                    refresh_plot()
        except KeyboardInterrupt:
            print("\nDrift test stopped by user (Ctrl+C).")
            stop_requested[0] = True
        finally:
            rests_handle.close()

    # Defensive final shutdown -- apply_rise_and_measure already closes its
    # own AWG/scope sessions after every measurement, so this is normally a
    # no-op, but it guarantees the AWG output is off before you walk away.
    try:
        ArbGenerator(cfg).output(False)
    except Exception as exc:
        print(f"Could not confirm AWG output is off: {exc}")
    else:
        print("AWG output confirmed off. Safe to start the next run.")

    try:
        plot_path = plot_drift(log_path, rests_path)
        print(f"Drift plot saved to {plot_path}")
    except Exception as exc:
        print(f"Could not generate drift plot: {exc}")

    try:
        video_path = make_drift_video(log_path, rests_path, default_video_output_path(log_path))
        print(f"Drift I1-I2 evolution video saved to {video_path}")
    except Exception as exc:
        print(f"Could not generate drift I1-I2 video: {exc}")

    if plt is not None:
        if block_on_finish:
            figure.suptitle("Drift test complete - close this window to exit")
            stop_button.label.set_text("Stop safely")
            stop_button.ax.set_facecolor("0.85")
            plt.ioff()
            plt.show()
        else:
            # The plot was already saved by plot_drift() above; close the live
            # window so a sweep doesn't pile up one window per setting.
            plt.close(figure)
    return stop_requested[0]


def main() -> None:
    cfg = ExperimentConfig()
    cfg = dataclasses.replace(cfg, data_mother_path=cfg.data_mother_path / DRIFT_NAME)
    run_drift_once(cfg)


if __name__ == "__main__":
    main()
