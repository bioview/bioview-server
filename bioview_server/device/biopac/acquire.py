import queue
import time
from ctypes import byref, c_double, c_uint

import numpy as np
from bioview_common import PausableWorker, log_print

from .constants import describe_biopac_code
from .utils import daemon_last_error


MPSUCCESS = 1

_LAG_WARN_RATIO = 0.9
_LAG_WARN_INTERVAL_S = 5.0


class BiopacAcquisitionWorker(PausableWorker):
    """Read samples from mpdev and emit (num_channels, num_samples) chunks."""

    def __init__(
        self,
        mpdev_handler,
        channels,
        samp_rate: int,
        display_queue: queue.Queue,
        save_queue: queue.Queue = None,
        save_ds: int = 1,
        chunk_size: int = 50,
        use_stream: bool = None,
        logger=None,
    ):
        super().__init__(logger=logger)
        self.mpdev_handler = mpdev_handler
        self.channels = channels
        self.samp_rate = max(1, int(samp_rate))
        self.display_queue = display_queue
        self.save_queue = save_queue
        self.save_ds = max(1, int(save_ds))
        self.chunk_size = max(1, int(chunk_size))
        self.channel_count = len(channels)
        self._period_s = 1.0 / self.samp_rate

        self._receive = getattr(mpdev_handler, "receiveMPData", None)
        if use_stream is False:
            self._receive = None
        self._values_per_chunk = self.chunk_size * self.channel_count
        self._stream_buffer = (
            (c_double * (self._values_per_chunk * self.channel_count))()
            if self._receive is not None
            else None
        )

        self._buffer = (c_double * (self.channel_count + 1))()
        self._chunk = []
        self._next_poll = None
        self._samples_seen = 0
        self._save_samples_emitted = 0
        self._save_remainder = None
        self._rate_window_start = None
        self._last_lag_warning = 0.0

    def work(self):
        if self.mpdev_handler is None or self.channel_count == 0:
            time.sleep(0.05)
            return

        try:
            if self._receive is not None:
                self._work_stream()
            else:
                self._work_poll()
        except Exception as e:
            log_print(self.logger, "error", f"[BIOPAC] Acquisition error: {e}")
            time.sleep(0.05)

    def _work_stream(self):
        """One blocking bulk read; the MP unit sets the pace."""
        received = c_uint(0)
        retval = self._receive(
            byref(self._stream_buffer),
            c_uint(self._values_per_chunk),
            byref(received),
        )

        values = min(int(received.value), len(self._stream_buffer))
        samples = values // self.channel_count
        if samples <= 0:
            if retval != MPSUCCESS:
                self._report_stream_failure(retval)
                time.sleep(0.01)
            return

        flat = np.frombuffer(
            self._stream_buffer, dtype=np.float64, count=samples * self.channel_count
        )
        rows = flat.reshape(samples, self.channel_count).T
        self._emit(np.array(rows, dtype=np.float64, order="C"))

    def _work_poll(self):
        now = time.monotonic()
        if self._next_poll is None:
            self._next_poll = now
            self._rate_window_start = now

        if now < self._next_poll:
            time.sleep(min(self._next_poll - now, self._period_s))
            return

        if self.mpdev_handler.getMostRecentSample(byref(self._buffer)) != MPSUCCESS:
            self._next_poll = now + self._period_s
            return

        self._chunk.append([self._buffer[i] for i in range(self.channel_count)])
        self._next_poll += self._period_s
        if self._next_poll < now:
            self._next_poll = now + self._period_s

        self._samples_seen += 1
        self._check_lag(now)

        if len(self._chunk) < self.chunk_size:
            return

        data = np.ascontiguousarray(np.asarray(self._chunk, dtype=np.float64).T)
        self._chunk.clear()
        self._emit(data)

    def _report_stream_failure(self, retval):
        """Log a failing bulk read, rate-limited so a dead daemon cannot spam."""
        now = time.monotonic()
        if now - self._last_lag_warning < _LAG_WARN_INTERVAL_S:
            return
        self._last_lag_warning = now
        detail = daemon_last_error(self.mpdev_handler, self.logger)
        suffix = f" (daemon error {detail})" if detail is not None else ""
        log_print(
            self.logger,
            "warning",
            f"[BIOPAC] receiveMPData returned no data: "
            f"{describe_biopac_code(retval)}{suffix}",
        )

    def _check_lag(self, now: float):
        """Warn when the poll loop cannot keep up with the requested rate."""
        elapsed = now - self._rate_window_start
        if elapsed < _LAG_WARN_INTERVAL_S:
            return

        achieved = self._samples_seen / elapsed
        self._samples_seen = 0
        self._rate_window_start = now

        if achieved >= _LAG_WARN_RATIO * self.samp_rate:
            return
        if now - self._last_lag_warning < _LAG_WARN_INTERVAL_S:
            return
        self._last_lag_warning = now
        log_print(
            self.logger,
            "warning",
            f"[BIOPAC] Acquiring {achieved:.0f} samples/s of the requested "
            f"{self.samp_rate}; plots will scroll slower than real time. "
            "This build of mpdev.dll has no receiveMPData, so samples must be "
            "polled one at a time -- consider a lower sample rate.",
        )

    def _decimate_for_save(self, data: np.ndarray) -> np.ndarray:
        """Average ``save_ds`` acquired samples into one saved sample."""
        if self.save_ds <= 1:
            return data.copy()
        if self._save_remainder is not None and self._save_remainder.size:
            data = np.hstack([self._save_remainder, data])
        n_windows = data.shape[1] // self.save_ds
        usable = n_windows * self.save_ds
        self._save_remainder = data[:, usable:].copy()
        if n_windows <= 0:
            return np.empty((data.shape[0], 0), dtype=data.dtype)
        return (
            data[:, :usable].reshape(data.shape[0], n_windows, self.save_ds).mean(axis=2)
        )

    def _emit(self, data: np.ndarray):
        if self.save_queue is not None:
            save_data = self._decimate_for_save(data)
            if save_data.shape[1]:
                item = {
                    "data": np.ascontiguousarray(save_data),
                    "sample_idx": self._save_samples_emitted,
                    "t_wall": time.time(),
                }
                self._save_samples_emitted += save_data.shape[1]
                try:
                    self.save_queue.put_nowait(item)
                except queue.Full:
                    log_print(
                        self.logger,
                        "error",
                        "[BIOPAC] Save queue full; dropping chunk",
                    )

        if self.display_queue is not None:
            try:
                self.display_queue.put_nowait(data)
            except queue.Full:
                log_print(
                    self.logger,
                    "warning",
                    "[BIOPAC] Display queue full; dropping chunk",
                )

    def cleanup(self):
        self._chunk.clear()
        self._save_remainder = None
