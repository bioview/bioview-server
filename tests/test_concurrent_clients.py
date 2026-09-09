"""Several BioView windows share the one server on the machine.

The Monitor and the Configurator both need a server, and only one may run per
machine, so the server has to serve them at the same time: each client gets its
own command thread, replies go back on the connection the command arrived on,
and acquired data is fanned out to every client.
"""
import contextlib
import socket
import threading
import time

import numpy as np
from bioview_common import (
    Command,
    Response,
    parse_and_validate_response,
    send_command,
)


def test_a_second_client_connects_while_the_first_is_connected(server, clients):
    # Before, the accept loop blocked inside the first client's session, so a
    # second window's connection sat unanswered until it timed out.
    assert len(server.sessions) == 2


def test_each_client_gets_its_own_replies(clients):
    for client in clients:
        resp_type, payload = client.command(Command.GET_DEVICE_STATUS)
        assert resp_type == Response.SUCCESS.name
        assert "device_status" in payload


def test_acquired_data_reaches_every_connected_client(server, clients):
    chunk = np.arange(6, dtype=np.float32).reshape(2, 3)
    server.data_queue.put({"data": chunk, "sources": [{"name": "a"}, {"name": "b"}]})

    for client in clients:
        data, sources = client.recv_data_chunk(timeout=5.0)
        np.testing.assert_array_equal(data, chunk)
        assert [s["name"] for s in sources] == ["a", "b"]


def test_one_client_leaving_does_not_disturb_the_others(server, clients):
    leaving, staying = clients
    leaving.close()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and len(server.sessions) > 1:
        time.sleep(0.05)
    assert len(server.sessions) == 1

    resp_type, _ = staying.command(Command.GET_DEVICE_STATUS)
    assert resp_type == Response.SUCCESS.name

    chunk = np.arange(3, dtype=np.float32)
    server.data_queue.put({"data": chunk, "sources": [{"name": "a"}]})
    data, _ = staying.recv_data_chunk(timeout=5.0)
    np.testing.assert_array_equal(data, chunk)


def test_discovery_reports_how_many_clients_are_connected(server, clients):
    control_port, _ = server._test_ports
    with socket.create_connection(("127.0.0.1", control_port), timeout=5) as sock:
        raw = send_command(sock, Command.DISCOVER_SERVERS)
        resp_type, payload = parse_and_validate_response(raw)

    assert resp_type == Response.SUCCESS.name
    assert payload["clients"] == 2


def test_a_server_started_by_a_window_retires_once_every_client_has_gone(idle_server):
    # A shared server outlives the window that spawned it, so it cleans itself
    # up rather than being killed while another window is still using it.
    srv, client, thread = idle_server(1.0)
    client.connect_and_authenticate()

    # Still serving a client, so it stays up well past the idle timeout.
    time.sleep(2.0)
    assert srv.running

    client.close()
    thread.join(timeout=10)
    assert not srv.running, "server should have retired after its last client left"


def test_a_server_without_the_idle_flag_stays_up(server, clients):
    for client in clients:
        client.close()
    time.sleep(1.5)
    assert server.running


def _probe_loop(control_port, command, stop, params=None):
    """Hammer the control port with one command until told to stop."""

    def _run():
        while not stop.is_set():
            with (
                contextlib.suppress(OSError),
                socket.create_connection(
                    ("127.0.0.1", control_port), timeout=0.5
                ) as sock,
            ):
                send_command(sock, command, params=params)
            time.sleep(0.1)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _claim(token="window-1", heartbeat=0.5, **extra):
    return {"window": token, "heartbeat": heartbeat, **extra}


def test_the_idle_check_does_not_hang_off_accept_timing_out(idle_server):
    """Traffic that is not a claim must not postpone the idle shutdown.

    The check used to run only when accept() timed out, so anything keeping the
    accept loop busy -- here, connections the server rejects -- would hold an
    abandoned server open indefinitely.
    """
    srv, _client, thread = idle_server(1.0)

    stop = threading.Event()
    prober = _probe_loop(srv.control_port, Command.GET_DEVICE_STATUS, stop)
    try:
        thread.join(timeout=10)
        assert not srv.running, "server stayed up while being probed"
    finally:
        stop.set()
        prober.join(timeout=2)


def test_a_window_holds_its_server_open_without_ever_connecting(idle_server):
    """A claiming window is a window that intends to connect.

    It may not have authenticated yet -- the Monitor builds its client only
    once its configuration dialog has been answered, and that can take as long
    as the user likes. So a window's heartbeat holds the server open, and the
    server only retires once nothing claims it any more.
    """
    srv, _client, thread = idle_server(1.0)

    stop = threading.Event()
    prober = _probe_loop(
        srv.control_port, Command.DISCOVER_SERVERS, stop, params=_claim()
    )
    try:
        time.sleep(3.0)
        assert srv.running, "a server a window still wanted was retired"
    finally:
        stop.set()
        prober.join(timeout=2)

    # ...and it is not immortal: once the last window stops calling, so is it.
    thread.join(timeout=10)
    assert not srv.running, "server stayed up after its last window went quiet"


def test_an_anonymous_probe_answers_but_claims_nothing(idle_server):
    """A subnet scan sweeps every host on the network. Answering one must not
    be enough to keep an abandoned server alive."""
    srv, _client, thread = idle_server(1.0)

    stop = threading.Event()
    prober = _probe_loop(srv.control_port, Command.DISCOVER_SERVERS, stop)
    try:
        thread.join(timeout=10)
        assert not srv.running, "an unclaimed server was held open by a bare probe"
    finally:
        stop.set()
        prober.join(timeout=2)


def test_a_window_that_says_goodbye_is_forgotten_at_once(idle_server):
    """Closing a window must not leave its claim to time out: the next window
    to close would then see a phantom and decline to shut the server down."""
    srv, _client, _thread = idle_server(0)

    with socket.create_connection(("127.0.0.1", srv.control_port), timeout=5) as sock:
        raw = send_command(sock, Command.DISCOVER_SERVERS, params=_claim(heartbeat=60))
        _, payload = parse_and_validate_response(raw)
    assert payload["windows"] == 1

    with socket.create_connection(("127.0.0.1", srv.control_port), timeout=5) as sock:
        raw = send_command(
            sock, Command.DISCOVER_SERVERS, params=_claim(heartbeat=60, leaving=True)
        )
        _, payload = parse_and_validate_response(raw)
    assert payload["windows"] == 0, "the reply counts everyone but the leaver"


def test_windows_are_counted_separately_not_collapsed(idle_server):
    """Two windows, two claims: the second to close is the one that may kill."""
    srv, _client, _thread = idle_server(0)

    for token in ("window-a", "window-b"):
        with socket.create_connection(
            ("127.0.0.1", srv.control_port), timeout=5
        ) as sock:
            raw = send_command(
                sock, Command.DISCOVER_SERVERS, params=_claim(token, heartbeat=60)
            )
            _, payload = parse_and_validate_response(raw)

    assert payload["windows"] == 2
