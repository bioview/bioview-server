"""Balance progress has to cross the process boundary to reach the UI."""

import multiprocessing as mp
import time

from bioview_common import DataSource, Response

from bioview_server.datatypes.backend import Backend
from bioview_server.server import Server


def _backend():
    return Backend(group_id="grp", response_queue=mp.Queue())


def _drained(be, timeout=5.0):
    """Drain until something arrives."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = be.drain_balance_progress()
        if value is not None:
            return value
    raise AssertionError("no progress arrived")


def test_progress_survives_the_queue():
    be = _backend()
    be.publish_balance_progress({"stage": "coarse phase", "point": 3})

    assert _drained(be) == {"stage": "coarse phase", "point": 3}


def test_only_the_newest_entry_is_reported():
    """A poll that lands after a burst must say where the search *is*."""
    be = _backend()
    for point in range(1, 6):
        be.publish_balance_progress({"stage": "coarse phase", "point": point})
    time.sleep(0.2)

    assert _drained(be)["point"] == 5
    assert be.drain_balance_progress() is None


def test_a_full_queue_never_blocks_the_search():
    """A UI that stops polling must not slow the balance down."""
    be = _backend()
    started = time.monotonic()
    for point in range(500):
        be.publish_balance_progress({"point": point})

    assert time.monotonic() - started < 5.0


class _FakeHandler:
    def __init__(self, progress):
        self._progress = progress

    def get_data_sources(self):
        return {DataSource(group_id="grp", channel=0, label="Tx1Rx1")}

    def drain_balance_progress(self):
        progress, self._progress = self._progress, None
        return progress


def _server_with(handler, monkeypatch):
    srv = Server(local_only=True, control_port=0, data_port=0)
    srv.device_group_handlers = {"grp": handler}
    srv._dpic_states = {
        "grp": {
            "pending": True,
            "ok": None,
            "message": "",
            "results": [],
            "device_id": "grp",
            "progress": None,
        }
    }
    srv._dpic_last_device = "grp"
    sent = {}

    def fake_send(sock, response, params=None, logger=None):
        sent["response"] = response
        sent["params"] = params or {}

    monkeypatch.setattr("bioview_server.server.send_response", fake_send)
    monkeypatch.setattr(Server, "client_control_conn", property(lambda self: None))
    return srv, sent


def test_device_status_carries_the_live_progress(monkeypatch):
    progress = {"stage": "fine phase", "point": 7, "phase_deg": 173.4}
    srv, sent = _server_with(_FakeHandler(progress), monkeypatch)

    srv._handle_get_device_status()

    assert sent["response"] is Response.SUCCESS
    assert sent["params"]["dpic_balance"]["progress"] == progress


def test_the_last_progress_is_repeated_when_nothing_new_arrived(monkeypatch):
    """The poll runs faster than a slow measurement; a gap is not a reset."""
    progress = {"stage": "fine phase", "point": 7}
    srv, sent = _server_with(_FakeHandler(progress), monkeypatch)

    srv._handle_get_device_status()
    srv._handle_get_device_status()

    assert sent["params"]["dpic_balance"]["progress"] == progress


def test_a_handler_without_progress_support_is_not_an_error(monkeypatch):
    class Plain:
        def get_data_sources(self):
            return set()

    srv, sent = _server_with(Plain(), monkeypatch)
    srv._handle_get_device_status()

    assert sent["response"] is Response.SUCCESS


def test_each_group_gets_its_own_live_progress(monkeypatch):
    """Two searches at once; neither group may show the other's values."""
    srv, sent = _server_with(_FakeHandler({"stage": "coarse", "point": 1}), monkeypatch)
    srv.device_group_handlers["grp2"] = _FakeHandler({"stage": "fine", "point": 9})
    srv._dpic_states["grp2"] = dict(srv._dpic_states["grp"], device_id="grp2")

    srv._handle_get_device_status()

    balances = sent["params"]["dpic_balances"]
    assert balances["grp"]["progress"] == {"stage": "coarse", "point": 1}
    assert balances["grp2"]["progress"] == {"stage": "fine", "point": 9}
