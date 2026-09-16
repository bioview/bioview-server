"""A dead input and a quiet one look identical on the plot."""

import queue

import numpy as np
import pytest

from bioview_server.device.microphone import acquire
from bioview_server.device.microphone.acquire import MicrophoneAcquisitionWorker


class _Log:
    """Stands in for the backend logger; records (level, message)."""

    def __init__(self):
        self.records = []

    def _add(self, level):
        return lambda msg, *a, **k: self.records.append((level, str(msg)))

    def __getattr__(self, name):
        if name in ("debug", "info", "warning", "error"):
            return self._add(name)
        raise AttributeError(name)

    def messages(self, level=None):
        return [m for lvl, m in self.records if level is None or lvl == level]


@pytest.fixture
def worker():
    log = _Log()
    w = MicrophoneAcquisitionWorker(
        channels=1,
        samp_rate=16000,
        display_queue=queue.Queue(),
        gain=1.0,
        logger=log,
    )
    w.log = log
    return w


def _feed(worker, data, at):
    """Run one chunk through the silence check at monotonic time ``at``."""
    import time as time_mod

    original = time_mod.monotonic
    acquire.time.monotonic = lambda: at
    try:
        worker._check_silence(np.asarray(data, dtype=float))
    finally:
        acquire.time.monotonic = original


def test_silence_is_not_reported_immediately(worker):
    """A brief gap between words is not a broken microphone."""
    zeros = np.zeros(160)
    _feed(worker, zeros, at=100.0)
    _feed(worker, zeros, at=102.0)
    assert worker.log.messages("warning") == []


def test_sustained_silence_is_reported_once(worker):
    zeros = np.zeros(160)
    _feed(worker, zeros, at=100.0)
    _feed(worker, zeros, at=100.0 + acquire._SILENCE_WARN_AFTER_S + 0.1)
    warnings = worker.log.messages("warning")
    assert len(warnings) == 1
    assert "digital silence" in warnings[0]
    assert "muted" in warnings[0]

    _feed(worker, zeros, at=200.0)
    _feed(worker, zeros, at=300.0)
    assert len(worker.log.messages("warning")) == 1


def test_signal_withdraws_the_warning(worker):
    zeros = np.zeros(160)
    _feed(worker, zeros, at=100.0)
    _feed(worker, zeros, at=110.0)
    assert len(worker.log.messages("warning")) == 1

    _feed(worker, np.full(160, 0.2), at=111.0)
    assert "Signal detected" in worker.log.messages("info")[-1]

    _feed(worker, zeros, at=112.0)
    _feed(worker, zeros, at=112.0 + acquire._SILENCE_WARN_AFTER_S + 0.1)
    assert len(worker.log.messages("warning")) == 2


def test_a_live_input_is_never_warned_about(worker):
    for i in range(50):
        _feed(worker, np.sin(np.arange(160) / 4.0) * 0.1, at=100.0 + i)
    assert worker.log.messages("warning") == []


def test_a_noise_floor_counts_as_signal(worker):
    """Only a device sitting at exact zero is silent; a real input's noise"""
    rng = np.random.default_rng(0)
    for i in range(20):
        _feed(worker, rng.normal(0, 1e-4, 160), at=100.0 + i)
    assert worker.log.messages("warning") == []


def test_the_verdict_resets_between_runs(worker):
    zeros = np.zeros(160)
    _feed(worker, zeros, at=100.0)
    _feed(worker, zeros, at=110.0)
    assert len(worker.log.messages("warning")) == 1

    worker.cleanup()
    assert worker._silent_since is None
    assert worker._silence_reported is False


def test_the_check_runs_on_the_real_work_path(worker):
    """Guards the wiring, not the logic: a silence check nothing calls is"""
    worker.capture_queue.put(np.zeros((160, 1), dtype=np.float32))
    worker.work()
    assert worker._silent_since is not None
