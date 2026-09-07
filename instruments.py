"""Thin VISA drivers for the Agilent ARB generator and Infiniium scope."""

from contextlib import contextmanager

import numpy as np

from config import ExperimentConfig
from waveform import ArbWaveform, normalize_for_generator

try:
    import pyvisa
except ImportError:  # Pure waveform/BO modules remain testable without lab drivers.
    pyvisa = None


def _resource_manager():
    if pyvisa is None:
        raise RuntimeError("pyvisa is required to communicate with the instruments")
    return pyvisa.ResourceManager()


class ArbGenerator:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def upload(self, waveform: ArbWaveform) -> None:
        # Vpp and offset follow the complete waveform range. The plateau samples
        # are transformed by the same affine mapping, so they still decode to
        # reset_voltage_v and -reset_voltage_v for every BO rise protocol.
        normalized, vpp, offset = normalize_for_generator(waveform.voltage_v)
        rm = _resource_manager()
        fg = rm.open_resource(self.cfg.instruments.function_generator_address)
        fg.timeout = self.cfg.instruments.function_generator_timeout_ms
        fg.write_termination = fg.read_termination = "\n"
        try:
            fg.write("*RST")
            fg.write("*CLS")
            fg.write("DATA VOLATILE," + ",".join(f"{value:.8g}" for value in normalized))
            fg.write("FUNC:USER VOLATILE")
            fg.write("FUNC USER")
            fg.write("OUTP:SYNC ON")
            fg.write(f"FREQ {1.0 / waveform.duration_s:.12g}")
            fg.write("VOLT:UNIT VPP")
            fg.write(f"VOLT {vpp:.12g}")
            fg.write(f"VOLT:OFFS {offset:.12g}")
            fg.query("*OPC?")
        finally:
            fg.close()
            # ResourceManager instances may share one VISA session. Closing it
            # here can invalidate an oscilloscope acquisition running elsewhere.

    def output(self, enabled: bool) -> None:
        rm = _resource_manager()
        fg = rm.open_resource(self.cfg.instruments.function_generator_address)
        fg.timeout = self.cfg.instruments.function_generator_timeout_ms
        try:
            fg.write("OUTP ON" if enabled else "OUTP OFF")
        finally:
            fg.close()


@contextmanager
def configured_scope(cfg: ExperimentConfig, capture_duration_s: float):
    rm = _resource_manager()
    scope = rm.open_resource(cfg.instruments.oscilloscope_address)
    scope.timeout = cfg.instruments.oscilloscope_timeout_ms
    scope.write_termination = scope.read_termination = "\n"
    points = int(round(capture_duration_s / cfg.instruments.scope_time_step_s)) + 1
    try:
        scope.write("*RST;*CLS")
        for channel in (cfg.instruments.current_channel, cfg.instruments.voltage_channel,
                        cfg.instruments.trigger_channel):
            scope.write(f":CHANNEL{channel}:DISPLAY ON")
        scope.write(f":CHANNEL{cfg.instruments.current_channel}:IMPEDANCE FIFTY")
        scope.write(f":CHANNEL{cfg.instruments.current_channel}:SCALE {cfg.instruments.current_scale_v_per_div}")
        scope.write(f":CHANNEL{cfg.instruments.voltage_channel}:SCALE {cfg.instruments.voltage_scale_v_per_div}")
        scope.write(f":CHANNEL{cfg.instruments.trigger_channel}:SCALE {cfg.instruments.trigger_scale_v_per_div}")
        scope.write(f":CHANNEL{cfg.instruments.current_channel}:OFFSET {cfg.instruments.current_offset_v}")
        scope.write(f":CHANNEL{cfg.instruments.voltage_channel}:OFFSET {cfg.instruments.voltage_offset_v}")
        scope.write(f":CHANNEL{cfg.instruments.trigger_channel}:OFFSET {cfg.instruments.trigger_offset_v}")
        scope.write(f":TIMEBASE:SCALE {capture_duration_s / 10.0}")
        scope.write(":TIMEBASE:REFERENCE LEFT")
        scope.write(":TIMEBASE:POSITION 0")
        scope.write(":WAVEFORM:FORMAT ASCII")
        scope.write(":ACQUIRE:SRATE:AUTO OFF")
        scope.write(f":ACQUIRE:SRATE {1.0 / cfg.instruments.scope_time_step_s}")
        scope.write(f":ACQUIRE:POINTS {points}")
        scope.write(f":WAVEFORM:POINTS {points}")
        scope.write(":ACQUIRE:MODE HRESOLUTION")
        if cfg.instruments.average_enabled:
            scope.write(":ACQUIRE:AVERAGE ON")
            scope.write(f":ACQUIRE:AVERAGE:COUNT {cfg.instruments.average_count}")
        else:
            scope.write(":ACQUIRE:AVERAGE OFF")
        scope.write(":TRIGGER:MODE EDGE")
        scope.write(f":TRIGGER:EDGE:SOURCE CHANNEL{cfg.instruments.trigger_channel}")
        scope.write(":TRIGGER:EDGE:SLOPE POSITIVE")
        scope.write(f":TRIGGER:LEVEL CHANNEL{cfg.instruments.trigger_channel},{cfg.instruments.trigger_level_v}")
        scope.write(":TRIGGER:SWEEP TRIGGERED")
        yield scope
    finally:
        try:
            scope.write(":RUN")
        except (pyvisa.errors.InvalidSession, pyvisa.errors.VisaIOError):
            # Never replace the actual acquisition error with a cleanup error.
            pass
        try:
            scope.close()
        except (pyvisa.errors.InvalidSession, pyvisa.errors.VisaIOError):
            pass
        finally:
            rm.close()


def read_scope_channel(scope, channel: int) -> tuple[np.ndarray, np.ndarray]:
    scope.write(f":WAVEFORM:SOURCE CHANNEL{channel}")
    scope.write(":WAVEFORM:FORMAT ASCII")
    dt = float(scope.query(":WAVEFORM:XINCREMENT?"))
    origin = float(scope.query(":WAVEFORM:XORIGIN?"))
    values = np.fromstring(scope.query(":WAVEFORM:DATA?").strip(), sep=",")
    return origin + np.arange(values.size) * dt, values
