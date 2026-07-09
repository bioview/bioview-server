import contextlib
import queue
import time
from ctypes import byref, c_double

import numpy as np

from bioview_common import log_print, PausableWorker


class BiopacAcquisitionWorker(PausableWorker):
    """Poll mpdev for samples and emit (num_channels, num_samples) numpy chunks."""

    def __init__(
        self,
        mpdev_handler,
        channels,
        samp_rate: int,
        display_queue: queue.Queue,
        save_queue: queue.Queue = None,
        chunk_size: int = 50,
        logger=None,
    ):
        super().__init__(logger=logger)
        self.mpdev_handler = mpdev_handler
        self.channels = channels
        self.samp_rate = max(1, int(samp_rate))
        self.display_queue = display_queue
        self.save_queue = save_queue
        self.chunk_size = max(1, int(chunk_size))
        self.channel_count = len(channels)
        self._period_s = 1.0 / self.samp_rate
        self._buffer = (c_double * (self.channel_count + 1))()
        self._chunk = []
        self._next_poll = None

    def work(self):
        if self.mpdev_handler is None:
            time.sleep(0.05)
            return

        now = time.monotonic()
        if self._next_poll is None:
            self._next_poll = now

        if now < self._next_poll:
            time.sleep(min(self._next_poll - now, self._period_s))
            return

        try:
            if self.mpdev_handler.getMostRecentSample(byref(self._buffer)) != 1:
                self._next_poll = now + self._period_s
                return

            sample = [self._buffer[i] for i in range(self.channel_count)]
            self._chunk.append(sample)
            self._next_poll += self._period_s

            if len(self._chunk) < self.chunk_size:
                return

            data = np.asarray(self._chunk, dtype=np.float64).T
            self._chunk.clear()

            if self.save_queue is not None:
                with contextlib.suppress(queue.Full):
                    self.save_queue.put_nowait(data.copy())

            if self.display_queue is not None:
                try:
                    self.display_queue.put_nowait(
                        np.ascontiguousarray(data, dtype=np.float64)
                    )
                except queue.Full:
                    log_print(
                        self.logger,
                        "warning",
                        "[BIOPAC] Display queue full; dropping chunk",
                    )
        except Exception as e:
            log_print(self.logger, "error", f"[BIOPAC] Acquisition error: {e}")
            time.sleep(0.05)

    def cleanup(self):
        self._chunk.clear()
