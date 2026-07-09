"""Shared fixtures for the BioView server end-to-end tests.

These boot a real :class:`~bioview_server.server.Server` in a background thread on
ephemeral ports and provide a thin, headless test client (raw sockets speaking
the BioView protocol via bioview-common) so the full connect / authenticate /
discover / initialize / stream path can be exercised against the always-available
dummy device backend -- no PyQt and no hardware required.
"""
import contextlib
import socket
import struct
import threading
import time

import numpy as np
import pytest

from bioview_common import (
    Command,
    DEVICE_OP_POLL_INTERVAL,
    DISCOVER_TIMEOUT,
    INIT_TIMEOUT_DEFAULT,
    Response,
    get_challenge_response,
    parse_and_validate_response,
    recv_message,
    send_command,
)
from bioview_server.server import Server


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class RawTestClient:
    """Minimal headless client: opens the control + data sockets, authenticates,
    and exposes command / data-chunk helpers used by the tests."""

    def __init__(self, control_port: int, data_port: int):
        self.control_port = control_port
        self.data_port = data_port
        self.control_sock = None
        self.data_sock = None

    def connect_and_authenticate(self, timeout: float = 5.0):
        self.control_sock = socket.create_connection(
            ("127.0.0.1", self.control_port), timeout=timeout
        )
        self.control_sock.settimeout(timeout)

        raw = send_command(self.control_sock, Command.CONNECT_SERVER, {"client_info": {}})
        resp_type, payload = parse_and_validate_response(raw)
        assert resp_type == Response.SERVER_CHALLENGE.name, resp_type

        token = get_challenge_response(payload["challenge"])
        raw = send_command(self.control_sock, Command.AUTHENTICATE_CLIENT, {"token": token})
        resp_type, _ = parse_and_validate_response(raw)
        assert resp_type == Response.AUTHENTICATION_SUCCESS.name, resp_type

        # The server accepts the data connection immediately after auth succeeds.
        self.data_sock = socket.create_connection(
            ("127.0.0.1", self.data_port), timeout=timeout
        )
        self.data_sock.settimeout(timeout)

    def command(self, command: Command, params=None):
        raw = send_command(self.control_sock, command, params)
        return parse_and_validate_response(raw)

    def wait_for_device_op(
        self,
        initial_resp_type,
        initial_payload,
        timeout: float = None,
    ):
        """Wait for an async discover/init operation to finish via polling."""
        if initial_resp_type != Response.DEVICE_CONNECTING.name:
            return initial_resp_type, initial_payload

        deadline = time.monotonic() + (timeout or INIT_TIMEOUT_DEFAULT)
        while time.monotonic() < deadline:
            resp_type, payload = self.command(Command.GET_DEVICE_STATUS)
            assert resp_type == Response.SUCCESS.name, payload
            if not payload.get("pending", False) and payload.get("device_status"):
                return Response.SUCCESS.name, payload
            time.sleep(DEVICE_OP_POLL_INTERVAL)

        pytest.fail("device operation did not complete in time")

    def device_command(self, command: Command, params=None, timeout: float = None):
        """Send discover/init and block until the async operation completes."""
        resp_type, payload = self.command(command, params)
        return self.wait_for_device_op(resp_type, payload, timeout=timeout)

    def recv_data_chunk(self, timeout: float = 5.0):
        """Receive one streamed numpy chunk from the data socket and return
        (data, sources)."""
        self.data_sock.settimeout(timeout)
        length_bytes = self._recv_exactly(self.data_sock, 4)
        (frame_len,) = struct.unpack("!I", length_bytes)
        body = self._recv_exactly(self.data_sock, frame_len)

        (header_len,) = struct.unpack("!I", body[:4])
        import json

        header = json.loads(body[4 : 4 + header_len].decode("utf-8"))
        raw = body[4 + header_len :]
        data = np.frombuffer(raw, dtype=np.dtype(header["dtype"])).reshape(header["shape"])
        return data, header.get("sources")

    @staticmethod
    def _recv_exactly(sock, n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed while receiving")
            buf += chunk
        return buf

    def close(self):
        for s in (self.data_sock, self.control_sock):
            if s is not None:
                with contextlib.suppress(Exception):
                    s.close()


@pytest.fixture
def server():
    """A running local-only server on ephemeral ports."""
    control_port = _free_port()
    data_port = _free_port()
    srv = Server(local_only=True, control_port=control_port, data_port=data_port)

    thread = threading.Thread(target=srv.start, daemon=True)
    thread.start()

    # Wait until the control port is accepting connections.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", control_port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.05)
    else:
        srv.stop()
        pytest.fail("server did not start listening in time")

    srv._test_ports = (control_port, data_port)
    yield srv

    # Tear down any device backend subprocesses spawned during the test. These
    # are non-daemon mp.Process workers whose run loop only exits on SHUTDOWN, so
    # they must be stopped explicitly or the interpreter hangs joining them.
    for handler in list(getattr(srv, "device_group_handlers", {}).values()):
        if handler is None:
            continue
        with contextlib.suppress(Exception):
            handler.disconnect()
        with contextlib.suppress(Exception):
            handler.shutdown()

    with contextlib.suppress(Exception):
        srv.stop()
    thread.join(timeout=5)


@pytest.fixture
def client(server):
    """An authenticated headless client connected to the running server."""
    control_port, data_port = server._test_ports
    c = RawTestClient(control_port, data_port)
    c.connect_and_authenticate()
    yield c
    c.close()
