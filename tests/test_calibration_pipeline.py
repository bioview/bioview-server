"""Calibration overlay: runtime enable survives Start, and reaches the display."""

import json
import logging
import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np
import pytest
from fakes.backend import FakeBackend


DATA_DIR = Path(__file__).resolve().parent / "data"
RF_CFG_PATH = DATA_DIR / "fake_dpic_2x2_mimo_cfg.json"


def _rf_group_config(**overrides):
    cfg = json.loads(RF_CFG_PATH.read_text(encoding="utf-8"))["Fake_DPIC_2x2"]
    cfg["dpic_balance"] = dict(cfg["dpic_balance"], auto_on_start=False)
    cfg["calibration"] = dict(cfg["calibration"], **overrides.pop("calibration", {}))
    cfg.update(overrides)
    return cfg


@pytest.fixture
def rf_backend():
    response_queue = mp.Queue()
    backend = FakeBackend(
        group_id="G", response_queue=response_queue, group_config=_rf_group_config()
    )
    backend.logger = logging.getLogger(__name__)
    backend._initialize()
    yield backend

    backend._stop_streaming()
    backend._disconnect()
    for worker in (backend.display_worker, backend.save_worker):
        if worker is not None and worker.is_alive():
            worker.join(timeout=2)

    for q in (response_queue, backend.data_output_queue, backend.display_queue):
        if q is None:
            continue
        q.cancel_join_thread()
        q.close()


def test_runtime_calibration_enable_survives_start(rf_backend):
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
    assert advertised == {s.label for s in display_sources}
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

    assert np.ptp(stream[cal_row]) > 0.1
    assert np.ptp(stream[tx1rx1_row]) > 0.01


def test_calibration_stays_off_when_never_enabled(rf_backend):
    rf_backend._setup_display({})
    rf_backend._start_streaming()
    assert not rf_backend._cal_enabled
    assert not any(
        s.calibration_enabled() for s in rf_backend.schemes_by_device.values()
    )


def test_chunk_row_count_matches_the_advertised_source_count(rf_backend):
    """The client reshapes .bvr samples by ``header["num_sources"]``."""
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
