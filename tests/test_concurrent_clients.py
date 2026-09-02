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


def test_discovery_probes_do_not_keep_an_idle_server_alive(idle_server):
    """The idle check must not hang off accept() timing out.

    A GUI hunting for a server probes localhost once a second, and a LAN scan
    probes every host; either would keep accept() busy and, if the check only
    ran when accept() timed out, hold an abandoned server open indefinitely.
    """
    srv, client, thread = idle_server(1.0)
    control_port = srv.control_port

    stop = threading.Event()

    def keep_probing():
        while not stop.is_set():
            with (
                contextlib.suppress(OSError),
                socket.create_connection(
                    ("127.0.0.1", control_port), timeout=0.5
                ) as sock,
            ):
                send_command(sock, Command.DISCOVER_SERVERS)
            time.sleep(0.1)

    prober = threading.Thread(target=keep_probing, daemon=True)
    prober.start()
    try:
        thread.join(timeout=10)
        assert not srv.running, "server stayed up while being probed"
    finally:
        stop.set()
        prober.join(timeout=2)
