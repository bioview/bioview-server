import contextlib
import queue
import time

import numpy as np
from bioview_common import PausableWorker, log_print


# How often an overflow burst may be reported. PortAudio raises the flag once
# per affected callback, so an underpowered machine would otherwise fill the log.
_OVERFLOW_WARN_INTERVAL_S = 5.0

#: Chunks held between PortAudio's callback thread and this worker. Deep enough
#: to ride out a scheduling hiccup on the consumer side, shallow enough that a
#: consumer which has stopped draining cannot grow it without limit.
CAPTURE_QUEUE_DEPTH = 64

#: How long an input may deliver nothing but digital silence before it is
#: reported. An input that opens and streams zeros is indistinguishable on the
#: plot from one that is simply quiet, and it is the single most common way a
#: session is recorded useless: a muted input, a jack in the wrong socket, or a
#: capture device Windows has set to zero gain.
_SILENCE_WARN_AFTER_S = 5.0

#: Below this peak amplitude a chunk counts as digital silence. Real inputs
#: carry a noise floor well above it; only a muted or disconnected one sits
#: this close to exact zero.
_SILENCE_FLOOR = 1e-6


class MicrophoneAcquisitionWorker(PausableWorker):
    """Drain PortAudio's capture callback and emit ``(channels, samples)`` chunks.

    Capture is a callback rather than blocking reads: PortAudio's blocking API
    is not implemented on every Windows host API (WDM-KS refuses it outright),
    and the callback path is the one that works everywhere. The callback itself
    runs on PortAudio's high-priority thread, so it does nothing but copy the
    frames into a bounded queue -- all conversion, gain and queue work happens
    here, where a slow consumer costs a dropped chunk rather than stalling the
    audio device.
    """

    def __init__(
        self,
        channels: int,
        samp_rate: int,
        display_queue: queue.Queue,
        save_queue: queue.Queue = None,
        gain: float = 1.0,
        logger=None,
    ):
        super().__init__(logger=logger)
        self.channel_count = max(1, int(channels))
        self.samp_rate = max(1, int(samp_rate))
        self.display_queue = display_queue
        self.save_queue = save_queue
        self.gain = float(gain)

        self.capture_queue = queue.Queue(maxsize=CAPTURE_QUEUE_DEPTH)
        self.overflows = 0
        self.dropped_captures = 0
        self._last_overflow_warning = 0.0

        # Silence tracking; see _SILENCE_WARN_AFTER_S. Reset by cleanup() so a
        # Stop/Start cycle gets a fresh verdict rather than inheriting the
        # previous run's.
        self._silent_since = None
        self._silence_reported = False

    # ------------------------------------------------- PortAudio callback

    def callback(self, indata, frames, time_info, status):
        """Handed to ``sd.InputStream``. Runs on PortAudio's own thread."""
        if status and status.input_overflow:
            self.overflows += 1

        # PortAudio reuses the buffer behind ``indata``, so this must copy.
        try:
            self.capture_queue.put_nowait(np.array(indata, dtype=np.float32))
        except queue.Full:
            self.dropped_captures += 1

    # ------------------------------------------------------------- worker

    def work(self):
        try:
            frames = self.capture_queue.get(timeout=0.1)
        except queue.Empty:
            return

        if self.overflows or self.dropped_captures:
            self._report_loss()

        if frames is None or len(frames) == 0:
            return

        # PortAudio hands back (frames, channels); the pipeline wants
        # (channels, samples), contiguous.
        data = np.ascontiguousarray(np.asarray(frames, dtype=np.float64).T)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] > self.channel_count:
            data = data[: self.channel_count, :]

        if self.gain != 1.0:
            data *= self.gain

        self._check_silence(data)
        self._emit(data)

    def _check_silence(self, data: np.ndarray):
        """Report an input that is delivering nothing but zeros.

        The stream is open and chunks are arriving on time, so nothing in the
        pipeline is wrong -- which is exactly why this is worth saying out loud.
        A muted or unconnected input plots a flat line, and a flat line is what
        a working-but-quiet input looks like too. Reported once per streaming
        run, and withdrawn as soon as any signal shows up.
        """
        if float(np.abs(data).max()) > _SILENCE_FLOOR:
            if self._silence_reported:
                log_print(
                    self.logger,
                    "info",
                    "[Microphone] Signal detected; the input is live",
                )
            self._silent_since = None
            self._silence_reported = False
            return

        now = time.monotonic()
        if self._silent_since is None:
            self._silent_since = now
            return
        if self._silence_reported or now - self._silent_since < _SILENCE_WARN_AFTER_S:
            return

        self._silence_reported = True
        log_print(
            self.logger,
            "warning",
            f"[Microphone] The input has delivered digital silence for "
            f"{_SILENCE_WARN_AFTER_S:.0f} s. The stream is healthy, so the "
            "device itself is muted, unplugged, or set to zero gain in the "
            "operating system's sound settings -- the plot will stay flat and "
            "the recording will be empty",
        )

    def _report_loss(self):
        """Log dropped input frames, rate-limited.

        Either counter means audio was lost before it reached the pipeline: the
        recording is short by that much and no longer sample-aligned with the
        rest of the session. That is a warning, not a debug note.
        """
        now = time.monotonic()
        if now - self._last_overflow_warning < _OVERFLOW_WARN_INTERVAL_S:
            return
        self._last_overflow_warning = now
        log_print(
            self.logger,
            "warning",
            f"[Microphone] Audio frames lost ({self.overflows} host overflows, "
            f"{self.dropped_captures} chunks dropped before conversion); the "
            "recording will be short by that much. Consider a larger blocksize "
            "or a lower sample rate",
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
                    "[Microphone] Display queue full; dropping chunk",
                )

    def cleanup(self):
        self._silent_since = None
        self._silence_reported = False
        with contextlib.suppress(Exception):
            while True:
                self.capture_queue.get_nowait()
        # get_nowait raises Empty, which the suppress above absorbs.
