"""A recording must not outlive the clients that asked for one."""

import time

import pytest
from bioview_common import DeviceStatus

from bioview_server.server import Server


class FakeHandler:
    """Stands in for a backend process: records what it was asked to do."""

    def __init__(self, group_id="DEV"):
        self.group_id = group_id
        self.stopped = 0
        self.disconnected = 0
        self.shut_down = 0

    def stop_streaming(self):
        self.stopped += 1

    def disconnect(self):
        self.disconnected += 1

    def shutdown(self):
        self.shut_down += 1

    def get_data_sources(self):
        return set()


class FakeWriter:
    """Stands in for the BvrWriter thread."""

    def __init__(self):
        self.stopped = 0
        self.joined = 0

    def stop(self):
        self.stopped += 1

    def join(self, timeout=None):
        self.joined += 1

    def is_alive(self):
        return False


class FakeSession:
    def __init__(self, name="window"):
        self.active = True
        self.info = {"hostname": name}
        self.control_conn = None
        self.data_conn = None

    @property
    def name(self):
        return self.info["hostname"]

    def close(self):
        self.active = False


@pytest.fixture
def streaming_server():
    """A server mid-recording, with one client and one device."""
    srv = Server(control_port=0, data_port=0, allow="loopback")
    handler = FakeHandler()
    writer = FakeWriter()
    srv.device_group_handlers = {"DEV": handler}
    srv.device_group_states = {"DEV": DeviceStatus.STREAMING.value}
    srv.bvr_writer = writer
    srv._streaming_active = True
    return srv, handler, writer


def test_last_client_leaving_stops_the_stream_and_closes_the_file(streaming_server):
    srv, handler, writer = streaming_server
    session = FakeSession()
    srv.sessions.append(session)

    srv._end_session(session)

    assert handler.stopped == 1
    assert writer.stopped == 1, "the recorder was never told to finish"
    assert srv.bvr_writer is None, "the file was left open"
    assert srv._streaming_active is False


def test_a_second_client_still_connected_keeps_the_run_going(streaming_server):
    srv, handler, writer = streaming_server
    leaving, staying = FakeSession("closing"), FakeSession("watching")
    srv.sessions.extend([leaving, staying])

    srv._end_session(leaving)

    assert handler.stopped == 0, "one window closing stopped another's run"
    assert srv.bvr_writer is writer
    assert srv._streaming_active is True


def test_the_watchdog_catches_a_stream_no_session_ever_retired(streaming_server):
    """The path that does not go through _end_session at all."""
    srv, handler, writer = streaming_server
    assert not srv.sessions

    srv._check_abandoned_recording()

    assert handler.stopped == 1
    assert srv.bvr_writer is None


def test_the_watchdog_closes_a_file_left_open_without_a_running_stream(
    streaming_server,
):
    srv, handler, writer = streaming_server
    srv._streaming_active = False

    srv._check_abandoned_recording()

    assert writer.stopped == 1
    assert srv.bvr_writer is None


def test_the_watchdog_is_quiet_when_there_is_nothing_to_close():
    srv = Server(control_port=0, data_port=0, allow="loopback")
    srv._check_abandoned_recording()
    assert srv.bvr_writer is None


def test_halting_twice_finalizes_the_file_once(streaming_server):
    srv, handler, writer = streaming_server

    srv._halt_orphaned_streaming("first")
    srv._halt_orphaned_streaming("second")

    assert handler.stopped == 1
    assert writer.stopped == 1


def test_server_stop_reaps_the_backend_processes(streaming_server):
    """Not housekeeping: an unreaped non-daemon child blocks interpreter exit."""
    srv, handler, writer = streaming_server

    srv.stop()

    assert handler.stopped == 1
    assert handler.shut_down == 1, "the backend subprocess was never reaped"
    assert writer.stopped == 1
    assert srv.bvr_writer is None
    assert srv.device_group_handlers == {}


def test_backend_processes_are_daemonic():
    """The backstop for every path that misses an orderly shutdown."""
    import multiprocessing as mp

    from fakes.backend import FakeBackend

    backend = FakeBackend(group_id="DEV", response_queue=mp.Queue())
    assert backend.daemon is True


def test_disconnect_devices_reaps_the_backend_too(monkeypatch):
    srv = Server(control_port=0, data_port=0, allow="loopback")
    handler = FakeHandler()
    srv.device_group_handlers = {"DEV": handler}
    srv.device_group_states = {"DEV": DeviceStatus.CONNECTED.value}

    sent = []
    monkeypatch.setattr(
        "bioview_server.server.send_response",
        lambda *a, **k: sent.append(k.get("params")),
    )

    srv._disconnect_devices()

    assert handler.disconnected == 1
    assert handler.shut_down == 1
    assert srv.device_group_handlers["DEV"] is None
    assert srv.device_group_states["DEV"] == DeviceStatus.DISCONNECTED.value


def test_shutdown_request_from_this_machine_retires_the_server(monkeypatch):
    srv = Server(control_port=0, data_port=0, allow="loopback")
    srv.running = True
    sent = []
    monkeypatch.setattr(
        "bioview_server.server.send_response",
        lambda *a, **k: sent.append(k.get("response")),
    )

    class Conn:
        closed = False

        def close(self):
            Conn.closed = True

    srv._handle_shutdown_request(Conn(), "127.0.0.1")

    assert srv.running is False
    assert Conn.closed


def test_shutdown_request_from_elsewhere_is_refused(monkeypatch):
    """Retiring the rig's server is not something the network may ask for."""
    srv = Server(control_port=0, data_port=0, allow="any")
    srv.running = True
    monkeypatch.setattr("bioview_server.server.send_response", lambda *a, **k: None)

    class Conn:
        def close(self):
            pass

    srv._handle_shutdown_request(Conn(), "8.8.8.8")

    assert srv.running is True


def test_stopping_a_server_that_never_streamed_is_harmless():
    srv = Server(control_port=0, data_port=0, allow="loopback")
    started = time.monotonic()
    srv.stop()
    assert time.monotonic() - started < 5
