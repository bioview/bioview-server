import queue
from typing import Dict, List

import numpy as np
import uhd

from bioview_common import log_print, PausableWorker
from bioview_common.signal_schemes import SignalScheme

INIT_DELAY = 0.05  # 50mS initial delay before transmit

TX_PARAMS = {
    "tx_gain",
    "tx_amplitude",
    "tx_phase",
    "if_freq",
    "calibration",
    "calibration.enabled",
    "signal_scheme",
    "fmcw",
    "pulsed_doppler",
}


class TransmitWorker(PausableWorker):
    def __init__(
        self,
        usrp,
        tx_gain: List[float],
        tx_channels: List[int],
        samp_rate: int,
        tx_streamer,
        scheme: SignalScheme,
        cmd_queue: queue.Queue,
        global_tx_offset: int = 0,
        running: bool = False,
        logger=None,
    ):
        super().__init__()

        self.logger = logger
        self.tx_gain = tx_gain
        self.tx_channels = tx_channels
        self.samp_rate = samp_rate
        self.usrp = usrp
        self.tx_streamer = tx_streamer
        self.scheme = scheme
        self.cmd_queue = cmd_queue
        self.global_tx_offset = global_tx_offset
        self.running = running

        self.tx_metadata = None
        self._sample_idx = 0
        self._use_cyclic = scheme.cycle_length() is not None
        self.tx_waveform = None
        self.tx_buffer_size = self.tx_streamer.get_max_num_samps()

        if self._use_cyclic:
            self._build_cyclic_buffer()
        else:
            self.tx_waveform = np.zeros(
                (len(tx_channels), self.tx_buffer_size), dtype=np.complex64
            )

    def _build_cyclic_buffer(self):
        period = self.scheme.cycle_length()
        len_buf = max(period * 20, self.tx_buffer_size)
        self.tx_waveform = self.scheme.generate(len_buf, 0)

    def _generate_chunk(self, n: int) -> np.ndarray:
        if self._use_cyclic and self.tx_waveform is not None:
            start = self._sample_idx % self.tx_waveform.shape[1]
            end = start + n
            if end <= self.tx_waveform.shape[1]:
                chunk = self.tx_waveform[:, start:end]
            else:
                part1 = self.tx_waveform[:, start:]
                part2 = self.tx_waveform[:, : end - self.tx_waveform.shape[1]]
                chunk = np.hstack([part1, part2])
            self._sample_idx += n
            return chunk
        chunk = self.scheme.generate(n, self._sample_idx)
        self._sample_idx += n
        return chunk

    def _apply_command(self, param: str, val):
        if param == "tx_gain":
            if val != self.tx_gain:
                for idx, chan in enumerate(self.tx_channels):
                    self.usrp.set_tx_gain(val[idx], chan)
            self.tx_gain = val
        elif param in TX_PARAMS or param.startswith("calibration."):
            self.scheme.update_param(param, val)
            if param in ("if_freq", "calibration", "calibration.enabled", "signal_scheme"):
                self._use_cyclic = self.scheme.cycle_length() is not None
                if self._use_cyclic:
                    self._build_cyclic_buffer()
        elif param == "tx_amplitude":
            amps = val if isinstance(val, list) else [val]
            local_amps = amps[
                self.global_tx_offset : self.global_tx_offset + len(self.tx_channels)
            ]
            self.scheme.update_param("tx_amplitude", local_amps)
        elif param == "if_freq":
            freqs = val if isinstance(val, list) else [val]
            local_freqs = freqs[
                self.global_tx_offset : self.global_tx_offset + len(self.tx_channels)
            ]
            self.scheme.update_param("if_freq", local_freqs)
        elif param == "tx_phase":
            phases = val if isinstance(val, list) else [val]
            local_phases = phases[self.global_tx_offset : self.global_tx_offset + len(self.tx_channels)]
            self.scheme.update_param("tx_phase", local_phases)
        elif param == "set_calibration_enabled":
            self.scheme.set_calibration_enabled(bool(val))
            self._use_cyclic = self.scheme.cycle_length() is not None
            if self._use_cyclic:
                self._build_cyclic_buffer()

    def set_global_tx_param(self, global_tx_idx: int, param: str, value):
        local_idx = global_tx_idx - self.global_tx_offset
        if local_idx < 0 or local_idx >= len(self.tx_channels):
            return
        if param == "phase":
            phases = [0.0] * self.scheme.get_num_tx_channels()
            for i in range(len(phases)):
                phases[i] = getattr(self.scheme, "tx_phase_deg", [0.0] * len(phases))[i]
            phases[local_idx] = value
            self.scheme.update_param("tx_phase", phases)
        elif param == "amplitude":
            amps = [self.scheme.get_tx_amplitude(i) for i in range(self.scheme.get_num_tx_channels())]
            amps[local_idx] = value
            self.scheme.update_param("tx_amplitude", amps)

    def work(self):
        log_print(self.logger, "debug", "Transmission Started")
        self.tx_metadata = uhd.types.TXMetadata()
        self.tx_metadata.start_of_burst = True
        self.tx_metadata.end_of_burst = False
        self.tx_metadata.has_time_spec = True
        self.tx_metadata.time_spec = uhd.types.TimeSpec(
            self.usrp.get_time_now().get_real_secs() + INIT_DELAY
        )

        while self.is_running:
            try:
                current_command = self.cmd_queue.get_nowait()
                self._apply_command(
                    current_command["param"], current_command["value"]
                )
            except queue.Empty:
                pass

            try:
                if self._use_cyclic:
                    buffer_iter = self._generate_chunk(self.tx_buffer_size)
                else:
                    if self.tx_waveform is None or self.tx_waveform.shape[1] != self.tx_buffer_size:
                        self.tx_waveform = self._generate_chunk(self.tx_buffer_size)
                    else:
                        self.tx_waveform = self._generate_chunk(self.tx_buffer_size)
                    buffer_iter = self.tx_waveform

                num_samps = self.tx_streamer.send(buffer_iter, self.tx_metadata)
            except RuntimeError as ex:
                log_print(self.logger, "error", f"Runtime error in transmit: {ex}")
                continue

            self.tx_metadata.start_of_burst = False
            self.tx_metadata.has_time_spec = False

            if num_samps < self.tx_buffer_size:
                log_print(
                    self.logger, "warning", f"Tx Sent only {num_samps} samples"
                )

        self.tx_metadata.end_of_burst = True
        n_ch = len(self.tx_channels)
        self.tx_streamer.send(
            np.zeros((n_ch, self.tx_buffer_size), dtype=np.complex64),
            self.tx_metadata,
        )
        log_print(self.logger, "debug", "Transmission Stopped")

    def cleanup(self):
        if self.tx_metadata is not None:
            try:
                self.tx_metadata.end_of_burst = True
                n_ch = len(self.tx_channels)
                self.tx_streamer.send(
                    np.zeros((n_ch, self.tx_buffer_size), dtype=np.complex64),
                    self.tx_metadata,
                )
                log_print(self.logger, "debug", "Transmission burst ended cleanly")
            except Exception as ex:
                log_print(self.logger, "error", f"Error ending transmission burst: {ex}")
