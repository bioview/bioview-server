"""Generate synthetic MIMO RX streams for dummy RF simulation."""

from __future__ import annotations

import queue
import time
from typing import Dict, List

import numpy as np

from bioview_common import PausableWorker, log_print
from bioview_common.signal_schemes.base import SignalScheme

from .rf_simulation import MimoChannelModel

SAVE_BUFFER_SCALE = 20


class DummyRfWorker(PausableWorker):
    """Produce per-device RX buffers from virtual TX schemes and a channel model."""

    def __init__(
        self,
        samp_rate: float,
        hardware: Dict[str, dict],
        rx_device_order: List[str],
        rx_queues: Dict[str, queue.Queue],
        schemes_by_device: Dict[str, SignalScheme],
        global_tx_to_device: Dict[int, tuple],
        global_tx_offsets: Dict[str, int],
        channel_model: MimoChannelModel,
        chunk_duration: float,
        logger=None,
    ):
        super().__init__(logger=logger)
        self.samp_rate = float(samp_rate)
        self.hardware = hardware
        self.rx_device_order = rx_device_order
        self.rx_queues = rx_queues
        self.schemes_by_device = schemes_by_device
        self.global_tx_to_device = global_tx_to_device
        self.global_tx_offsets = global_tx_offsets
        self.channel_model = channel_model

        base_chunk = max(1, int(round(self.samp_rate * float(chunk_duration))))
        self.chunk_size = base_chunk * SAVE_BUFFER_SCALE
        self.chunk_duration = self.chunk_size / self.samp_rate

        self._sample_idx = 0
        self._next_emit = None
        self._rx_offsets = {}
        offset = 0
        for dev_name in rx_device_order:
            self._rx_offsets[dev_name] = offset
            offset += len(hardware[dev_name].get("rx_channels", [0]))

    def _generate_global_tx(self, n_samples: int, start_sample: int) -> np.ndarray:
        num_tx = len(self.channel_model.if_freq)
        tx_all = np.zeros((num_tx, n_samples), dtype=np.complex64)
        for dev_name, scheme in self.schemes_by_device.items():
            offset = self.global_tx_offsets[dev_name]
            wave = scheme.generate(n_samples, start_sample)
            for local in range(wave.shape[0]):
                tx_all[offset + local] = wave[local]
        return tx_all

    def work(self):
        now = time.monotonic()
        if self._next_emit is None:
            self._next_emit = now

        n_samples = self.chunk_size
        start = self._sample_idx
        tx_all = self._generate_global_tx(n_samples, start)
        rx_global = self.channel_model.synthesize(
            tx_all,
            self.schemes_by_device,
            self.global_tx_to_device,
            start,
            n_samples,
        )
        self._sample_idx += n_samples

        for dev_name in self.rx_device_order:
            rx_offset = self._rx_offsets[dev_name]
            n_rx = len(self.hardware[dev_name].get("rx_channels", [0]))
            device_buf = np.ascontiguousarray(
                rx_global[rx_offset : rx_offset + n_rx], dtype=np.complex64
            )
            try:
                self.rx_queues[dev_name].put_nowait(device_buf)
            except queue.Full:
                log_print(
                    self.logger,
                    "warning",
                    f"[DUMMY RF] Rx queue full for {dev_name}; dropping chunk",
                )

        self._next_emit += self.chunk_duration
        sleep_for = self._next_emit - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._next_emit = time.monotonic()
