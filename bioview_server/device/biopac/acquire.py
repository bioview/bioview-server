import contextlib
import queue
import time
from ctypes import byref, c_double, c_uint

import numpy as np
from bioview_common import PausableWorker, log_print

from .constants import describe_biopac_code
from .utils import daemon_last_error


#: mpdev's MPSUCCESS. Every mpdev entry point returns one of these codes.
MPSUCCESS = 1

#: How far behind real time the polling fallback may drift before it says so.
#: The plot's x-axis is drawn from the *nominal* sample rate, so a worker that
#: only manages half the requested rate makes the trace scroll at half speed --
#: silently, unless we measure it.
_LAG_WARN_RATIO = 0.9
_LAG_WARN_INTERVAL_S = 5.0


class BiopacAcquisitionWorker(PausableWorker):
    """Read samples from mpdev and emit (num_channels, num_samples) numpy chunks.

    Two strategies, picked at construction:

    * ``receiveMPData`` -- the buffered stream read. It blocks until the MP unit
      has produced the requested number of points and returns them *in order*,
      so the emitted timeline matches the hardware clock exactly and one DLL
      call covers a whole chunk.
    * ``getMostRecentSample`` -- a per-sample poll, used only when the DLL does
      not export ``receiveMPData``. This asks the device for whatever value it
      happens to be holding, so it neither paces itself nor guarantees distinct
      consecutive samples: the loop has to hit the sample rate on its own, and
      at 1 kHz a Python loop plus a driver round-trip per sample often cannot.
      Falling short does not look like dropped data -- it looks like a plot that
      scrolls slowly, because the samples that do arrive are stretched over the
      x-axis of the rate we asked for.
    """

    def __init__(
        self,
        mpdev_handler,
        channels,
        samp_rate: int,
        display_queue: queue.Queue,
        save_queue: queue.Queue = None,
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
        self.chunk_size = max(1, int(chunk_size))
        self.channel_count = len(channels)
        self._period_s = 1.0 / self.samp_rate

        # Bulk read, when the DLL offers it *and* the caller managed to start
        # the acquisition daemon that feeds it. receiveMPData returns nothing
        # without that daemon, so the flag is the caller's to set.
        self._receive = getattr(mpdev_handler, "receiveMPData", None)
        if use_stream is False:
            self._receive = None
        # mpdev counts *values*, not samples: the buffer comes back with the
        # enabled channels interleaved. The allocation carries a factor of
        # channel_count more room than the request needs, so that a DLL which
        # (contrary to the header) reads the count as samples-per-channel
        # overruns nothing.
        self._values_per_chunk = self.chunk_size * self.channel_count
        self._stream_buffer = (
            (c_double * (self._values_per_chunk * self.channel_count))()
            if self._receive is not None
            else None
        )

        # Per-sample poll state (fallback path only).
        self._buffer = (c_double * (self.channel_count + 1))()
        self._chunk = []
        self._next_poll = None
        self._samples_seen = 0
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
                # Acquisition stopped, or the unit has nothing for us. Back off
                # briefly so a persistent failure cannot spin this thread.
                self._report_stream_failure(retval)
                time.sleep(0.01)
            return

        flat = np.frombuffer(
            self._stream_buffer, dtype=np.float64, count=samples * self.channel_count
        )
        # Interleaved (sample-major) -> one row per channel, contiguous. np.array
        # copies unconditionally, which is required here: the source is the
        # ctypes buffer the next call overwrites. (ascontiguousarray would not
        # copy for a single channel, where the transpose is already C-order.)
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
        # A loop that has fallen behind would otherwise chase a deadline that is
        # already in the past for as long as it stays behind, polling flat out.
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
        detail = daemon_last_error(self.mpdev_handler)
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

    def _emit(self, data: np.ndarray):
        if self.save_queue is not None:
            with contextlib.suppress(queue.Full):
                self.save_queue.put_nowait(data.copy())

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
