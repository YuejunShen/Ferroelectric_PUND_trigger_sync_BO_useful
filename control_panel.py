"""Live, iteration-by-iteration visualization for the PUND rise-shape BO experiment."""

import numpy as np
from waveform import rise_voltage


class ControlPanel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = cfg.enable_control_panel
        self.works = []
        self.deltas_p = []
        self.feasible = []
        # Set when the user clicks "Stop safely" -- checked by run_experiment.py
        # between iterations (never mid-measurement, since apply_rise_and_measure
        # always closes its own AWG/scope sessions before returning), so the
        # run always ends with the instruments in a clean, reconnectable state.
        self.stop_requested = False
        if not self.enabled:
            return
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.plt = plt
        plt.ion()
        self.figure, axes = plt.subplots(2, 2, figsize=(13, 8))
        self.ax_voltage, self.ax_current, self.ax_next, self.ax_history = axes.ravel()
        self.ax_charge = self.ax_history.twinx()
        self.figure.canvas.manager.set_window_title("PUND Rise-Shape Bayesian Optimization")
        self.figure.subplots_adjust(bottom=0.12)
        self.figure.tight_layout(pad=3.0, rect=(0.0, 0.08, 1.0, 1.0))

        stop_ax = self.figure.add_axes([0.4, 0.01, 0.2, 0.05])
        self.stop_button = Button(stop_ax, "Stop safely", color="tab:red", hovercolor="0.9")
        self.stop_button.on_clicked(self._on_stop_clicked)

    def _on_stop_clicked(self, _event):
        self.stop_requested = True
        self.stop_button.label.set_text("Stopping after this\nmeasurement...")
        self.stop_button.ax.set_facecolor("0.8")
        self.figure.canvas.draw_idle()

    def show_next(self, control_voltages_v, status="Waiting for measurement"):
        if not self.enabled:
            return
        duration = self.cfg.waveform.rise_time_s
        control_t = np.linspace(0.0, duration, len(control_voltages_v))
        dense_t = np.linspace(0.0, duration, 1000)
        basis = self.cfg.waveform.rise_basis
        dense_v = rise_voltage(control_voltages_v, dense_t / duration, basis)
        self.ax_next.clear()
        self.ax_next.plot(dense_t * 1e6, dense_v, color="tab:orange", linewidth=2)
        self.ax_next.scatter(control_t * 1e6, control_voltages_v, color="tab:red", zorder=3)
        self.ax_next.set(title=f"Next BO rise choice ({basis}{len(control_voltages_v)})",
                         xlabel="Rise time (us)", ylabel="Voltage (V)")
        self.ax_next.set_xlim(0.0, duration * 1e6)
        self.ax_next.grid(True, alpha=0.3)
        self.figure.suptitle(status)
        self._refresh()

    def set_status(self, status):
        """Update the title and let the window handle pending clicks (e.g. "Stop
        safely") during loops that don't otherwise redraw, such as the warm-up."""
        if not self.enabled:
            return
        self.figure.suptitle(status)
        self.figure.canvas.draw_idle()
        self.plt.pause(0.01)

    def run_responsive(self, function, *args):
        """Call function(*args) on a worker thread while keeping the window's
        event loop running, so the panel stays restorable/clickable during a
        long measurement or GP fit instead of freezing ("Not Responding")."""
        if not self.enabled:
            return function(*args)
        import threading

        outcome = {}

        def work():
            try:
                outcome["value"] = function(*args)
            except BaseException as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while worker.is_alive():
            # start_event_loop (not plt.pause) so the window is not re-shown
            # or pulled to the front on every tick.
            self.figure.canvas.start_event_loop(0.1)
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    def _refresh(self):
        self.figure.tight_layout(pad=3.0)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.05)

    def update(self, measurement, result, next_voltages_v, iteration):
        if not self.enabled:
            return
        self.ax_voltage.clear()
        self.ax_voltage.plot(measurement.rise_time_s * 1e6, measurement.rise_voltage_v, color="tab:blue")
        self.ax_voltage.set(title="Measured voltage (rise window)", xlabel="Rise time (us)", ylabel="Voltage (V)")
        self.ax_voltage.grid(True, alpha=0.3)

        self.ax_current.clear()
        self.ax_current.plot(measurement.rise_time_s * 1e6, measurement.rise_current_diff_a,
                             color="tab:green", label="Subtracted current I1-I2")
        self.ax_current.set(title="Subtracted current (rise window)", xlabel="Rise time (us)", ylabel="Current (A)")
        self.ax_current.grid(True, alpha=0.3)
        self.ax_current.legend(loc="best")

        self.works.append(float(result["Work"]))
        self.deltas_p.append(float(result["delta_P"]))
        self.feasible.append(bool(result["feasible"]))
        self.ax_history.clear()
        self.ax_charge.clear()
        self.ax_history.yaxis.set_label_position("left")
        self.ax_history.yaxis.tick_left()
        self.ax_history.spines["left"].set_position(("axes", 0.0))
        self.ax_charge.yaxis.set_label_position("right")
        self.ax_charge.yaxis.tick_right()
        self.ax_charge.spines["right"].set_position(("axes", 1.0))
        indices = np.arange(len(self.works))
        colors = ["tab:green" if value else "tab:red" for value in self.feasible]
        self.ax_history.plot(indices, self.works, color="0.5", linewidth=1)
        self.ax_history.scatter(indices, self.works, c=colors, s=24)
        self.ax_history.set(title="Work history (green = polarization feasible)",
                            xlabel="Evaluation", ylabel="Work = Integral V*I_diff dt (J)")
        self.ax_history.tick_params(axis="y", colors="0.25")
        self.ax_history.grid(True, alpha=0.3)
        self.ax_charge.plot(indices, self.deltas_p, color="tab:purple", marker="o",
                            markersize=3, label="Measured delta_P")
        self.ax_charge.axhline(float(result["P_target"]), color="tab:red",
                               linestyle="--", label="Target delta_P")
        self.ax_charge.set_ylabel("delta_P (C/m^2)", color="tab:purple")
        self.ax_charge.tick_params(axis="y", colors="tab:purple")
        self.ax_charge.legend(loc="lower left")

        status = (f"Evaluation {iteration} | {result['phase']} | "
                  f"delta_P={result['delta_P']:.4g} C/m^2 | target={result['P_target']:.4g} C/m^2 | "
                  f"feasible={result['feasible']}")
        self.show_next(next_voltages_v, status)

    def finish(self, block: bool = True):
        """block=False closes the window immediately instead of waiting for the
        user -- used by run_parameter_sweep.py so a multi-run sweep proceeds
        unattended instead of stalling on every run's plot window."""
        if not self.enabled:
            return
        self.figure.suptitle("Experiment complete - close this window to exit")
        self.stop_button.label.set_text("Stop safely")
        self.stop_button.ax.set_facecolor("0.85")
        self._refresh()
        self.plt.ioff()
        if block:
            self.plt.show()
        else:
            self.plt.close(self.figure)
