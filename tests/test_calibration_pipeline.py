"""Calibration overlay: runtime enable survives Start, and reaches the display.

These cover the two failures that made an enabled calibration signal invisible
in the receive path:

1. ``_start_streaming`` replayed the config's start-up ``calibration.enabled``
   value, undoing a runtime enable on every Start.
2. The ``CalRef_*`` source was advertised to the client but excluded from the
   display payload, so its plot could never receive a row.
"""

import json
import logging
import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np
import pytest

from bioview_server.device.dummy.backend import DummyBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
RF_CFG_PATH = REPO_ROOT / "dummy_dpic_2x2_mimo_cfg.json"


def _rf_group_config(**overrides):
    cfg = json.loads(RF_CFG_PATH.read_text(encoding="utf-8"))["Dummy_DPIC_2x2"]
    cfg["dpic_balance"] = dict(cfg["dpic_balance"], auto_on_start=False)
    cfg["calibration"] = dict(cfg["calibration"], **overrides.pop("calibration", {}))
    cfg.update(overrides)
    return cfg


@pytest.fixture
def rf_backend():
    response_queue = mp.Queue()
    backend = DummyBackend(
        group_id="G", response_queue=response_queue, group_config=_rf_group_config()
    )
    backend.logger = logging.getLogger(__name__)
    backend._initialize()
    yield backend

    backend._stop_streaming()
    backend._disconnect()
    # Let the stopped workers fall out of their queue reads before the queues go
    # away, or they raise on a closed handle.
    for worker in (backend.display_worker, backend.save_worker):
        if worker is not None and worker.is_alive():
            worker.join(timeout=2)

    # mp.Queue feeder threads join at interpreter exit; without this the test
    # session hangs after the last test in this module.
    for q in (response_queue, backend.data_output_queue, backend.display_queue):
        if q is None:
            continue
        q.cancel_join_thread()
        q.close()


def test_runtime_calibration_enable_survives_start(rf_backend):
    # The user ticks "Calibration signal" before pressing Start.
    cal = dict(rf_backend.group_config["calibration"], enabled=True)
    rf_backend._queue_param_update({"calibration": cal})
    assert rf_backend._cal_enabled

    rf_backend._setup_display({})
    rf_backend._start_streaming()

    assert rf_backend._cal_enabled
    assert all(s.calibration_enabled() for s in rf_backend.schemes_by_device.values())


def test_calibration_reference_is_on_the_display_path(rf_backend):
    rf_backend._setup_display({})
    display_sources = rf_backend.display_worker.display_sources
    advertised = {s.label for s in rf_backend.get_data_sources()}

    assert "CalRef_Tx1" in advertised
    # Every advertised source must have a row in the emitted chunk, or its plot
    # stays blank forever.
    assert advertised == {s.label for s in display_sources}
    # Row i of the payload is display_sources[i], and ProcessWorker indexes its
    # output arrays by source.channel -- so the list has to be channel-ordered.
    channels = [s.channel for s in display_sources]
    assert channels == sorted(channels)
    assert channels == list(range(len(channels)))


def test_enabled_calibration_reaches_the_receive_chunk(rf_backend):
    cal = dict(rf_backend.group_config["calibration"], enabled=True)
    rf_backend._queue_param_update({"calibration": cal})
    rf_backend._setup_display({})
    rf_backend._start_streaming()

    sources = rf_backend.display_worker.display_sources
    cal_row = next(i for i, s in enumerate(sources) if s.label == "CalRef_Tx1")
    tx1rx1_row = next(i for i, s in enumerate(sources) if s.label == "Tx1Rx1")

    chunks = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and len(chunks) < 8:
        try:
            chunks.append(rf_backend.data_output_queue.get(timeout=1.0)["data"])
        except queue.Empty:
            continue

    assert chunks, "no display chunks were produced"
    stream = np.hstack(chunks)

    # The reference envelope is a real gated burst, so it must vary.
    assert np.ptp(stream[cal_row]) > 0.1
    # And the AM overlay must be visible on the demodulated Tx1 magnitude.
    assert np.ptp(stream[tx1rx1_row]) > 0.01


def test_calibration_stays_off_when_never_enabled(rf_backend):
    rf_backend._setup_display({})
    rf_backend._start_streaming()
    assert not rf_backend._cal_enabled
    assert not any(
        s.calibration_enabled() for s in rf_backend.schemes_by_device.values()
    )


def test_chunk_row_count_matches_the_advertised_source_count(rf_backend):
    """The client reshapes .bvr samples by ``header["num_sources"]``.

    That header is built from the advertised source list, so a payload with
    fewer rows than sources does not just blank a plot -- it makes every
    recording reshape wrong and scrambles the whole file.
    """
    rf_backend._setup_display({})
    rf_backend._start_streaming()

    num_sources = len(rf_backend.get_data_sources())
    deadline = time.monotonic() + 10
    payload = None
    while time.monotonic() < deadline and payload is None:
        try:
            payload = rf_backend.data_output_queue.get(timeout=1.0)
        except queue.Empty:
            continue

    assert payload is not None, "no display chunk arrived"
    assert len(payload["sources"]) == num_sources
    assert np.atleast_2d(payload["data"]).shape[0] == num_sources
