"""How the BIOPAC worker reads samples out of mpdev.

The distinction that matters here is between the two mpdev transfer styles:
``receiveMPData`` (fed by the acquisition daemon) hands back a paced, ordered
stream, while ``getMostRecentSample`` is a per-sample poll that the caller has
to pace itself. Choosing the wrong one does not lose data outright -- it makes
the plot scroll slower than real time, which is much harder to spot.
"""

import ctypes
import multiprocessing as mp
import queue
import time

import numpy as np
import pytest

from bioview_server.device.biopac.acquire import BiopacAcquisitionWorker
from bioview_server.device.biopac.backend import BIOPACBackend


MPSUCCESS = 1


class FakeStreamingMpdev:
    """An mpdev.dll that supports the acquisition daemon."""

    def __init__(self, n_channels=2, fail=False):
        self.n_channels = n_channels
        self.fail = fail
        self.sample_index = 0
        self.requests = []
        self.daemon_started = False

    def startMPAcqDaemon(self):
        self.daemon_started = True
        return MPSUCCESS

    def receiveMPData(self, buff, num_points, num_received):
        n = num_points.value if hasattr(num_points, "value") else int(num_points)
        self.requests.append(n)
        received = ctypes.cast(num_received, ctypes.POINTER(ctypes.c_uint)).contents

        if self.fail:
            received.value = 0
            return 5  # MPNOTCON

        out = ctypes.cast(buff, ctypes.POINTER(ctypes.c_double))
        # Interleaved: channel c of sample s carries s * 100 + c, so a
        # de-interleaving bug shows up as an obviously wrong row.
        for i in range(n):
            sample, channel = divmod(i, self.n_channels)
            out[i] = (self.sample_index + sample) * 100 + channel
        self.sample_index += n // self.n_channels
        received.value = n
        return MPSUCCESS

    def getMostRecentSample(self, buff):  # pragma: no cover - must not be used
        raise AssertionError("polled a device that supports the daemon")


class FakePollingMpdev:
    """An older mpdev.dll: single-sample polling only."""

    def __init__(self, n_channels=2):
        self.n_channels = n_channels
        self.n = 0

    def getMostRecentSample(self, buff):
        out = ctypes.cast(buff, ctypes.POINTER(ctypes.c_double))
        for c in range(self.n_channels):
            out[c] = self.n * 100 + c
        self.n += 1
        return MPSUCCESS


@pytest.fixture
def display_queue():
    return queue.Queue(maxsize=32)


def _run(worker, seconds=1.0):
    worker.start()
    worker.resume()
    time.sleep(seconds)
    worker.stop()
    worker.join(timeout=2)


def test_the_stream_read_is_used_when_the_daemon_is_available(display_queue):
    dev = FakeStreamingMpdev()
    worker = BiopacAcquisitionWorker(
        mpdev_handler=dev,
        channels=[0, 1],
        samp_rate=1000,
        display_queue=display_queue,
        chunk_size=100,
        use_stream=True,
    )
    _run(worker, 0.5)

    assert dev.requests, "receiveMPData was never called"
    # 100 samples x 2 channels: mpdev counts values, not samples. Asking for 100
    # would deliver half a chunk per read and halve the effective sample rate.
    assert set(dev.requests) == {200}


def test_interleaved_values_are_split_back_into_one_row_per_channel(display_queue):
    worker = BiopacAcquisitionWorker(
        mpdev_handler=FakeStreamingMpdev(n_channels=2),
        channels=[0, 1],
        samp_rate=1000,
        display_queue=display_queue,
        chunk_size=10,
        use_stream=True,
    )
    _run(worker, 0.3)

    chunk = display_queue.get_nowait()
    assert chunk.shape == (2, 10)
    np.testing.assert_array_equal(chunk[0], np.arange(10) * 100)
    np.testing.assert_array_equal(chunk[1], np.arange(10) * 100 + 1)


def test_a_chunk_is_not_a_view_of_the_ctypes_buffer(display_queue):
    """Single channel is the case where the transpose is already C-contiguous."""
    worker = BiopacAcquisitionWorker(
        mpdev_handler=FakeStreamingMpdev(n_channels=1),
        channels=[0],
        samp_rate=1000,
        display_queue=display_queue,
        chunk_size=8,
        use_stream=True,
    )
    _run(worker, 0.3)

    first = display_queue.get_nowait()
    second = display_queue.get_nowait()
    # If the arrays aliased the reused ctypes buffer they would be identical.
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first[0], np.arange(8) * 100)


def test_use_stream_false_forces_polling_even_if_the_dll_could_stream(display_queue):
    dev = FakeStreamingMpdev()
    worker = BiopacAcquisitionWorker(
        mpdev_handler=dev,
        channels=[0, 1],
        samp_rate=200,
        display_queue=display_queue,
        chunk_size=5,
        use_stream=False,
    )
    # getMostRecentSample raises on this fake, so the worker logging an error
    # rather than crashing is the expected shape of "it tried to poll".
    _run(worker, 0.2)
    assert dev.requests == []


def test_polling_still_works_when_the_dll_has_no_daemon(display_queue):
    worker = BiopacAcquisitionWorker(
        mpdev_handler=FakePollingMpdev(n_channels=2),
        channels=[0, 1],
        samp_rate=200,
        display_queue=display_queue,
        chunk_size=4,
    )
    _run(worker, 0.5)

    chunk = display_queue.get_nowait()
    assert chunk.shape == (2, 4)
    np.testing.assert_array_equal(chunk[1] - chunk[0], np.ones(4))


def test_a_failing_stream_read_does_not_spin_or_emit(display_queue):
    dev = FakeStreamingMpdev(fail=True)
    worker = BiopacAcquisitionWorker(
        mpdev_handler=dev,
        channels=[0, 1],
        samp_rate=1000,
        display_queue=display_queue,
        chunk_size=100,
        use_stream=True,
    )
    _run(worker, 0.3)

    assert display_queue.empty()
    # A tight retry loop would run into the thousands over 0.3 s.
    assert len(dev.requests) < 60


def test_the_save_queue_gets_its_own_copy(display_queue):
    save_queue = queue.Queue(maxsize=32)
    worker = BiopacAcquisitionWorker(
        mpdev_handler=FakeStreamingMpdev(),
        channels=[0, 1],
        samp_rate=1000,
        display_queue=display_queue,
        save_queue=save_queue,
        chunk_size=10,
        use_stream=True,
    )
    _run(worker, 0.3)

    displayed = display_queue.get_nowait()
    saved = save_queue.get_nowait()
    np.testing.assert_array_equal(displayed, saved)
    assert displayed is not saved


# --- Backend wiring -------------------------------------------------------
#
# mpdev requires startMPAcqDaemon() *before* startAcquisition(), and forbids
# mixing the daemon with getMostRecentSample in one acquisition. So the choice
# of transfer style belongs to the backend, not the worker.

GROUP_CFG = {
    "device_name": "BIOPAC",
    "model": "MP36",
    "samp_rate": 1000,
    "disp_ds": 10,
    "channels": [1, 1, 0, 0],
    "hardware": {"BIOPAC_MP36": {"channels": [1, 1, 0, 0]}},
}


class _Worker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def is_alive(self):
        return False

    def start(self):
        pass

    def resume(self):
        pass

    def pause(self):
        pass


@pytest.fixture
def backend(monkeypatch):
    be = BIOPACBackend(
        group_id="BIOPAC", response_queue=mp.Queue(), group_config=dict(GROUP_CFG)
    )
    # run() installs the logger in the child process; these tests drive the
    # object directly.
    be.logger = None
    return be


def _spy_worker(monkeypatch, made):
    def factory(**kwargs):
        w = _Worker(**kwargs)
        made.append(w)
        return w

    monkeypatch.setattr(
        "bioview_server.device.biopac.backend.BiopacAcquisitionWorker", factory
    )


def test_the_daemon_is_started_before_acquisition(backend, monkeypatch):
    order = []
    monkeypatch.setattr(
        "bioview_server.device.biopac.backend.start_acquisition",
        lambda *_: order.append("acquire"),
    )
    monkeypatch.setattr(
        "bioview_server.device.biopac.backend.start_acq_daemon",
        lambda *_: (order.append("daemon"), True)[1],
    )
    made = []
    _spy_worker(monkeypatch, made)
    backend.mpdev_handler = object()

    assert backend._start_streaming() is True
    assert order == ["daemon", "acquire"]
    assert made[0].kwargs["use_stream"] is True


def test_a_daemon_that_will_not_start_falls_back_to_polling(backend, monkeypatch):
    monkeypatch.setattr(
        "bioview_server.device.biopac.backend.start_acquisition", lambda *_: True
    )

    def boom(*_):
        raise Exception("MPDRVERR")

    monkeypatch.setattr("bioview_server.device.biopac.backend.start_acq_daemon", boom)
    made = []
    _spy_worker(monkeypatch, made)
    backend.mpdev_handler = object()

    # Streaming still starts -- polling is degraded, not broken.
    assert backend._start_streaming() is True
    assert made[0].kwargs["use_stream"] is False


def test_chunk_size_is_capped_so_the_plot_cannot_lag_by_a_second(backend, monkeypatch):
    monkeypatch.setattr(
        "bioview_server.device.biopac.backend.start_acquisition", lambda *_: True
    )
    monkeypatch.setattr(
        "bioview_server.device.biopac.backend.start_acq_daemon", lambda *_: True
    )
    made = []
    _spy_worker(monkeypatch, made)
    backend.mpdev_handler = object()
    # disp_ds = 1 would otherwise ask for a full second of samples per read,
    # which the stream read blocks for.
    backend.disp_ds = 1

    backend._start_streaming()
    assert made[0].kwargs["chunk_size"] == 100  # 100 ms at 1 kHz
