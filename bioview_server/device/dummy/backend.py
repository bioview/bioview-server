"""
Virtual "dummy" device backend.

It generates a configurable number of phase-shifted sine waves at a given
sampling rate and streams them through the exact same path a real device uses
(display_queue -> DisplayWorker -> data_output_queue -> server -> client). This
lets the whole connect / initialize / stream / display / save flow be exercised
without any physical hardware attached.
"""

import time
import queue
import multiprocessing as mp

import numpy as np

from bioview_common import DataSource, DeviceStatus, PausableWorker, log_print

from bioview_server.datatypes import Backend


class SineWaveWorker(PausableWorker):
    """Continuously synthesizes a (num_channels, chunk_size) block of phase-shifted
    sine waves at roughly real time and pushes it onto the display queue. Phase is
    accumulated across chunks so the waveform stays continuous between blocks."""

    def __init__(
        self,
        samp_rate: float,
        num_channels: int,
        signal_freq: float,
        amplitude: float,
        noise_std: float,
        chunk_duration: float,
        display_queue: mp.Queue,
        logger=None,
    ):
        super().__init__(logger=logger)
        self.samp_rate = float(samp_rate)
        self.num_channels = int(num_channels)
        self.signal_freq = float(signal_freq)
        self.amplitude = float(amplitude)
        self.noise_std = float(noise_std)
        self.display_queue = display_queue

        # At least one sample per chunk
        self.chunk_size = max(1, int(round(self.samp_rate * float(chunk_duration))))
        self.chunk_duration = self.chunk_size / self.samp_rate

        # Per-channel constant phase offset so the channels are visibly shifted
        # relative to one another (evenly spread across a full period).
        self.phase_offsets = (
            2.0 * np.pi * np.arange(self.num_channels) / max(1, self.num_channels)
        )

        # Running sample index for cross-chunk phase continuity, and a monotonic
        # pacing clock so we emit data at approximately real time.
        self._sample_idx = 0
        self._next_emit = None

    def work(self):
        if self.display_queue is None:
            return

        now = time.monotonic()
        if self._next_emit is None:
            self._next_emit = now

        # Build the time vector for this chunk (continuous across chunks)
        n = np.arange(self._sample_idx, self._sample_idx + self.chunk_size)
        t = n / self.samp_rate

        # (num_channels, chunk_size) phase-shifted sine waves
        angle = 2.0 * np.pi * self.signal_freq * t
        chunk = self.amplitude * np.sin(
            angle[np.newaxis, :] + self.phase_offsets[:, np.newaxis]
        )

        if self.noise_std > 0:
            chunk = chunk + np.random.normal(
                0.0, self.noise_std, size=chunk.shape
            )

        self._sample_idx += self.chunk_size

        try:
            self.display_queue.put_nowait(np.ascontiguousarray(chunk, dtype=float))
        except queue.Full:
            log_print(self.logger, "warning", "[DUMMY] Display queue full; dropping chunk")

        # Pace to real time without drifting
        self._next_emit += self.chunk_duration
        sleep_for = self._next_emit - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # We fell behind; reset the pacing clock to avoid a runaway catch-up
            self._next_emit = time.monotonic()


class DummyBackend(Backend):
    def __init__(
        self,
        group_id: str,
        samp_rate: int,
        num_channels: int,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None,
        signal_freq: float = 1.0,
        amplitude: float = 1.0,
        noise_std: float = 0.0,
        chunk_duration: float = 0.05,
    ):
        super().__init__(
            group_id=group_id,
            response_queue=response_queue,
            data_output_queue=data_output_queue,
        )
        self.samp_rate = samp_rate
        self.num_channels = int(num_channels)
        self.signal_freq = signal_freq
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.chunk_duration = chunk_duration

        # The generating worker (created lazily inside the subprocess on start)
        self.generator_worker = None

        # Populate the data sources advertised to the client
        self.populate_data_sources()

    def populate_data_sources(self):
        # The display buffer on the client is sized by disp_freq; using samp_rate
        # keeps the plotted window temporally accurate for this virtual device.
        for ch in range(self.num_channels):
            source = DataSource(
                group_id=self.group_id,
                channel=ch,
                label=f"{self.group_id} Ch{ch + 1}",
                disp_freq=float(self.samp_rate),
            )
            self.data_sources.add(source)

    def _initialize(self):
        # Nothing to connect to; a virtual device is always ready.
        log_print(self.logger, "debug", f"[DUMMY] Initialized {self.group_id} "
                  f"({self.num_channels} ch @ {self.samp_rate} Hz)")
        self.status = DeviceStatus.CONNECTED
        return True

    def _start_streaming(self):
        # Start/resume the display worker (set up by the base class on START_STREAMING)
        if self.display_worker is not None:
            if not self.display_worker.is_alive():
                self.display_worker.start()
            self.display_worker.resume()

        # Start/resume the optional server-side save worker (off by default)
        if self.save_worker is not None:
            if not self.save_worker.is_alive():
                self.save_worker.start()
            self.save_worker.resume()

        # Create the signal generator once, then resume it
        if self.generator_worker is None:
            self.generator_worker = SineWaveWorker(
                samp_rate=self.samp_rate,
                num_channels=self.num_channels,
                signal_freq=self.signal_freq,
                amplitude=self.amplitude,
                noise_std=self.noise_std,
                chunk_duration=self.chunk_duration,
                display_queue=self.display_queue,
                logger=self.logger,
            )

        if not self.generator_worker.is_alive():
            self.generator_worker.start()
        self.generator_worker.resume()

        self.status = DeviceStatus.STREAMING
        log_print(self.logger, "debug", f"[DUMMY] Streaming started for {self.group_id}")
        return True

    def _stop_streaming(self):
        if self.generator_worker is not None:
            self.generator_worker.pause()

        if self.display_worker is not None:
            self.display_worker.pause()

        if self.save_worker is not None:
            self.save_worker.pause()

        self.status = DeviceStatus.CONNECTED
        log_print(self.logger, "debug", f"[DUMMY] Streaming stopped for {self.group_id}")
        return True

    def _queue_param_update(self, params):
        # Allow a couple of live tweaks while streaming for testing convenience.
        if self.generator_worker is None:
            return
        for param, value in (params or {}).items():
            if param == "signal_freq":
                self.generator_worker.signal_freq = float(value)
            elif param == "amplitude":
                self.generator_worker.amplitude = float(value)
            elif param == "noise_std":
                self.generator_worker.noise_std = float(value)

    def _disconnect(self):
        if self.generator_worker is not None:
            self.generator_worker.stop()
            self.generator_worker = None

        if self.display_worker is not None:
            self.display_worker.stop()

        if self.save_worker is not None:
            self.save_worker.stop()

        self.status = DeviceStatus.DISCONNECTED
        return True
