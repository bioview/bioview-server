"""Analog gains, and when they moved, belong in the recording."""

import json
import struct
import time

import numpy as np
import pytest
from bioview_common import BVR_TRAILER_MAGIC

from bioview_server.common.save import BvrWriter
from bioview_server.datatypes.backend import Backend
from bioview_server.device.usrp.backend import USRPBackend


def _read_trailer(path):
    blob = path.read_bytes()
    assert blob.endswith(BVR_TRAILER_MAGIC)
    end = len(blob) - len(BVR_TRAILER_MAGIC)
    (length,) = struct.unpack("!Q", blob[end - 8 : end])
    return json.loads(blob[end - 8 - length : end - 8])


@pytest.fixture
def writer(tmp_path):
    path = tmp_path / "session.bvr"
    w = BvrWriter(
        save_path=path,
        data_queue=None,
        devices=[{"device_id": "USRP1", "fs": 1000.0, "n_rows": 2}],
    )
    w.open(t0_unix=time.time())
    w.path = path
    return w


def test_a_gain_change_on_the_save_queue_lands_in_the_trailer(writer):
    """The metadata record rides the same queue as the samples."""
    assert (
        writer._handle_metadata(
            {
                "type": "param_change",
                "device_id": "USRP1",
                "param": "rx_gain",
                "value": [31.0, 30.0],
                "t_wall": writer.t0_unix + 2.5,
            }
        )
        is True
    )
    writer.cleanup()

    changes = _read_trailer(writer.path)["param_changes"]
    assert changes == [
        {
            "offset_us": 2_500_000,
            "device_id": "USRP1",
            "param": "rx_gain",
            "value": [31.0, 30.0],
        }
    ]


def test_a_sample_record_is_still_written_as_samples(writer):
    """The metadata check must not swallow ordinary chunks."""
    item = {
        "device_id": "USRP1",
        "data": np.zeros((2, 4)),
        "sample_idx": 0,
        "t_wall": writer.t0_unix,
    }
    assert writer._handle_metadata(item) is False
    writer._append(item)
    assert writer.records_written == 1


def test_an_unknown_metadata_type_is_dropped_rather_than_misread(writer):
    """Falling through to _append would log an error per item, forever."""
    assert (
        writer._handle_metadata({"type": "something_new", "device_id": "USRP1"}) is True
    )
    writer.cleanup()
    assert _read_trailer(writer.path)["param_changes"] == []


class _Queue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)

    def put(self, item):
        self.items.append(item)


class _GainBackend:
    """A USRPBackend cut down to the gain-writing path."""

    record_param_change = Backend.record_param_change
    _set_global_rx_gain = USRPBackend._set_global_rx_gain
    _set_global_tx_gain = USRPBackend._set_global_tx_gain
    _record_gain_baseline = USRPBackend._record_gain_baseline

    def __init__(self, enable_save=True):
        import threading

        self.group_id = "USRP1"
        self.logger = None
        self.enable_save = enable_save
        self.save_output_queue = _Queue()
        self._gain_lock = threading.Lock()

        self.hardware = {"dev": {"rx_channels": [0, 1], "tx_channels": [0, 1]}}
        self.group_config = {}
        self.rx_gains_global = [30.0, 30.0]
        self.tx_gains_global = [40.0, 40.0]
        self.global_rx_to_device = {0: ("dev", 0), 1: ("dev", 1)}
        self.global_tx_to_device = {0: ("dev", 0), 1: ("dev", 1)}
        self.rx_command_queue = {"dev": _Queue()}
        self.transmit_workers = {"dev": self}

    def set_global_tx_param(self, *_args):
        return None


def test_a_balancer_gain_step_is_recorded_with_the_whole_group_list():
    """The list, not the single channel: a gain trace has to be reconstructable"""
    backend = _GainBackend()

    backend._set_global_rx_gain(1, 33.0)
    backend._set_global_tx_gain(0, 41.0)

    recorded = backend.save_output_queue.items
    assert [(r["param"], r["value"]) for r in recorded] == [
        ("rx_gain", [30.0, 33.0]),
        ("tx_gain", [41.0, 40.0]),
    ]
    assert all(r["device_id"] == "USRP1" for r in recorded)
    assert all(r["type"] == "param_change" for r in recorded)
    assert all(r["t_wall"] > 0 for r in recorded)


def test_the_session_opens_with_the_gains_it_started_at():
    """The trace is a list of changes, which only reconstructs an absolute gain"""
    backend = _GainBackend()

    backend._record_gain_baseline()

    assert [(r["param"], r["value"]) for r in backend.save_output_queue.items] == [
        ("rx_gain", [30.0, 30.0]),
        ("tx_gain", [40.0, 40.0]),
    ]


def test_nothing_is_queued_when_the_session_is_not_recording():
    backend = _GainBackend(enable_save=False)

    backend._set_global_rx_gain(0, 35.0)
    backend._record_gain_baseline()

    assert backend.save_output_queue.items == []
    assert backend.rx_gains_global == [35.0, 30.0]


def test_a_full_recorder_queue_never_blocks_the_search_that_caused_it():
    class _Full(_Queue):
        def put_nowait(self, item):
            raise Exception("queue full")

    backend = _GainBackend()
    backend.save_output_queue = _Full()

    assert backend.record_param_change("rx_gain", [30.0, 30.0]) is False
