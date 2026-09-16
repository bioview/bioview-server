"""Balances on different device groups must run side by side."""

import threading

import pytest
from bioview_common import Response

from bioview_server.server import Server


class _BlockingHandler:
    """A handler whose balance blocks until the test releases it."""

    def __init__(self, group_id):
        self.group_id = group_id
        self.started = threading.Event()
        self.release = threading.Event()

    def run_dpic_balance(self):
        self.started.set()
        assert self.release.wait(timeout=10)
        return {"type": Response.SUCCESS, "message": f"{self.group_id} done"}

    def drain_balance_progress(self):
        return None

    def get_data_sources(self):
        return set()


@pytest.fixture
def server(monkeypatch):
    srv = Server(local_only=True, control_port=0, data_port=0)
    sent = []

    def fake_send(sock, response, params=None, logger=None):
        sent.append((response, params or {}))

    monkeypatch.setattr("bioview_server.server.send_response", fake_send)
    monkeypatch.setattr(Server, "client_control_conn", property(lambda self: None))
    srv.sent = sent
    return srv


def test_a_second_group_balances_while_the_first_is_running(server):
    one, two = _BlockingHandler("grp1"), _BlockingHandler("grp2")
    server.device_group_handlers = {"grp1": one, "grp2": two}

    server._run_dpic_balance({"id": "grp1"})
    assert one.started.wait(timeout=5)

    server._run_dpic_balance({"id": "grp2"})
    assert two.started.wait(timeout=5)
    assert server.sent[-1][0] is Response.SUCCESS

    states = server._dpic_states
    assert states["grp1"]["pending"] and states["grp2"]["pending"]

    one.release.set()
    two.release.set()
    for group in ("grp1", "grp2"):
        server._dpic_threads[group].join(timeout=10)
        assert server._dpic_states[group]["ok"] is True
        assert server._dpic_states[group]["message"] == f"{group} done"


def test_the_same_group_is_still_refused_a_second_balance(server):
    one = _BlockingHandler("grp1")
    server.device_group_handlers = {"grp1": one}

    server._run_dpic_balance({"id": "grp1"})
    assert one.started.wait(timeout=5)

    server._run_dpic_balance({"id": "grp1"})
    response, params = server.sent[-1]
    assert response is Response.ERROR
    assert "already running on grp1" in params["message"]

    one.release.set()
    server._dpic_threads["grp1"].join(timeout=10)


def test_status_reports_every_group_and_the_last_one_singly(server):
    one, two = _BlockingHandler("grp1"), _BlockingHandler("grp2")
    server.device_group_handlers = {"grp1": one, "grp2": two}
    server._run_dpic_balance({"id": "grp1"})
    server._run_dpic_balance({"id": "grp2"})
    assert one.started.wait(timeout=5) and two.started.wait(timeout=5)

    server.sent.clear()
    server._handle_get_device_status()
    _response, params = server.sent[-1]

    assert set(params["dpic_balances"]) == {"grp1", "grp2"}
    assert params["dpic_balance"]["device_id"] == "grp2"

    one.release.set()
    two.release.set()
    for group in ("grp1", "grp2"):
        server._dpic_threads[group].join(timeout=10)
