import queue
from typing import List

import numpy as np
import uhd
from bioview_common import PausableWorker, log_print
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
    "global_tx_phase",
    "global_tx_amplitude",
    "global_tx_gain",
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
            gains = val if isinstance(val, list) else [val]
            local_gains = gains[
                self.global_tx_offset : self.global_tx_offset + len(self.tx_channels)
            ]
            if len(local_gains) < len(self.tx_channels):
                local_gains = list(local_gains) + [
                    local_gains[-1] if local_gains else 0
                ] * (len(self.tx_channels) - len(local_gains))
            if local_gains != self.tx_gain:
                for idx, chan in enumerate(self.tx_channels):
                    self.usrp.set_tx_gain(local_gains[idx], chan)
            self.tx_gain = local_gains
        elif param in ("global_tx_phase", "global_tx_amplitude", "global_tx_gain"):
            self._apply_global_tx_param(param, val)
        # These lists arrive in *global* Tx indexing and must be sliced to this
        # device's window. They are all in TX_PARAMS, so they must precede the
        # generic branch below or device 2 gets device 1's values.
        elif param in ("tx_amplitude", "if_freq", "tx_phase"):
            values = val if isinstance(val, (list, tuple)) else [val]
            local = list(
                values[
                    self.global_tx_offset : self.global_tx_offset + len(self.tx_channels)
                ]
            )
            if not local:
                return
            self.scheme.update_param(param, local)
            if param == "if_freq":
                self._refresh_cyclic()
        elif param in TX_PARAMS or param.startswith("calibration."):
            self.scheme.update_param(param, val)
            if param in ("calibration", "calibration.enabled", "signal_scheme"):
                self._refresh_cyclic()
        elif param == "set_calibration_enabled":
            self.scheme.set_calibration_enabled(bool(val))
            self._refresh_cyclic()

    def _refresh_cyclic(self):
        self._use_cyclic = self.scheme.cycle_length() is not None
        if self._use_cyclic:
            self._build_cyclic_buffer()

    def _local_idx(self, global_tx_idx: int):
        local_idx = global_tx_idx - self.global_tx_offset
        if local_idx < 0 or local_idx >= len(self.tx_channels):
            return None
        return local_idx

    def get_global_tx_param(self, global_tx_idx: int, param: str):
        """Read back a single Tx's phase (deg) or amplitude, or None if not ours."""
        local_idx = self._local_idx(global_tx_idx)
        if local_idx is None:
            return None
        if param == "phase":
            phases = getattr(self.scheme, "tx_phase_deg", None)
            if not phases or local_idx >= len(phases):
                return 0.0
            return float(phases[local_idx])
        if param == "amplitude":
            return float(self.scheme.get_tx_amplitude(local_idx))
        if param == "gain":
            return float(self.tx_gain[local_idx])
        return None

    def set_global_tx_param(self, global_tx_idx: int, param: str, value):
        """Queue a per-Tx phase/amplitude change.

        This must go through ``cmd_queue`` rather than mutating ``self.scheme``
        directly: the caller (the backend's DPIC balance) runs on a different
        thread from ``work()``, which reads the scheme's parameter lists on every
        buffer. Direct mutation races with waveform generation.
        """
        if self._local_idx(global_tx_idx) is None:
            return
        if param == "phase":
            self.cmd_queue.put(
                {"param": "global_tx_phase", "value": (global_tx_idx, float(value))}
            )
        elif param == "amplitude":
            self.cmd_queue.put(
                {"param": "global_tx_amplitude", "value": (global_tx_idx, float(value))}
            )
        elif param == "gain":
            # Analog Tx gain on a single channel. DPIC needs this to reach
            # direct paths that a digital weight of at most 1.0 cannot cancel.
            self.cmd_queue.put(
                {"param": "global_tx_gain", "value": (global_tx_idx, float(value))}
            )

    def _apply_global_tx_param(self, param: str, value):
        global_tx_idx, val = value
        local_idx = self._local_idx(global_tx_idx)
        if local_idx is None:
            return
        n_ch = self.scheme.get_num_tx_channels()
        if param == "global_tx_phase":
            existing = list(getattr(self.scheme, "tx_phase_deg", []) or [])
            phases = [
                float(existing[i]) if i < len(existing) else 0.0 for i in range(n_ch)
            ]
            if local_idx >= len(phases):
                return
            phases[local_idx] = float(val)
            self.scheme.update_param("tx_phase", phases)
        elif param == "global_tx_amplitude":
            amps = [float(self.scheme.get_tx_amplitude(i)) for i in range(n_ch)]
            if local_idx >= len(amps):
                return
            amps[local_idx] = float(val)
            self.scheme.update_param("tx_amplitude", amps)
        elif param == "global_tx_gain":
            if local_idx >= len(self.tx_channels):
                return
            self.usrp.set_tx_gain(float(val), self.tx_channels[local_idx])
            self.tx_gain = list(self.tx_gain)
            self.tx_gain[local_idx] = float(val)

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
            # Drain fully: a DPIC step enqueues phase and amplitude together
            # and they must land on the same buffer.
            while True:
                try:
                    current_command = self.cmd_queue.get_nowait()
                except queue.Empty:
                    break
                self._apply_command(current_command["param"], current_command["value"])

            try:
                if self._use_cyclic:
                    buffer_iter = self._generate_chunk(self.tx_buffer_size)
                else:
                    if (
                        self.tx_waveform is None
                        or self.tx_waveform.shape[1] != self.tx_buffer_size
                    ):
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
                log_print(self.logger, "warning", f"Tx Sent only {num_samps} samples")

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
