"""A balance must not block anything that has to keep running while it does."""

import multiprocessing as mp
import threading
import time

from bioview_common import IPCCommand, Response
from bioview_common.signal_schemes.dpic import DpicBalancer, DpicChannel

from bioview_server.datatypes.backend import Backend


class _SlowBalanceBackend(Backend):
    """A backend whose balance blocks until it is released or aborted."""

    def __init__(self):
        super().__init__(
            group_id="grp", response_queue=mp.Queue(), data_output_queue=mp.Queue()
        )
        self.started = threading.Event()
        self.release = threading.Event()

    def _run_dpic_balance(self):
        self.started.set()
        while not self.release.is_set() and not self.balance_aborted():
            time.sleep(0.01)
        return {
            "ok": not self.balance_aborted(),
            "message": "aborted" if self.balance_aborted() else "done",
            "results": [],
        }

    def _stop_streaming(self):
        return True


def _drain_reply(backend, request_id, seen, timeout=5.0):
    """Read one request's reply, keeping any others in ``seen`` for later."""
    if request_id in seen:
        return seen.pop(request_id)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        response = backend.response_queue.get(timeout=remaining)
        if response.get("request_id") == request_id:
            return response
        seen[response.get("request_id")] = response
    raise AssertionError(f"no reply for request {request_id}")


def test_stop_streaming_is_answered_while_a_balance_runs():
    be = _SlowBalanceBackend()
    be._handle_command(
        {"command": IPCCommand.RUN_DPIC_BALANCE, "args": {}, "request_id": 1}
    )
    assert be.started.wait(timeout=5)

    be._handle_command(
        {"command": IPCCommand.STOP_STREAMING, "args": {}, "request_id": 2}
    )
    seen = {}
    stop_reply = _drain_reply(be, 2, seen)
    assert stop_reply["type"] == Response.SUCCESS

    balance_reply = _drain_reply(be, 1, seen)
    assert balance_reply["type"] == Response.ERROR
    assert balance_reply["message"] == "aborted"


def test_a_second_balance_is_refused_while_one_is_running():
    be = _SlowBalanceBackend()
    be._handle_command(
        {"command": IPCCommand.RUN_DPIC_BALANCE, "args": {}, "request_id": 1}
    )
    assert be.started.wait(timeout=5)

    be._handle_command(
        {"command": IPCCommand.RUN_DPIC_BALANCE, "args": {}, "request_id": 2}
    )
    seen = {}
    refusal = _drain_reply(be, 2, seen)
    assert refusal["type"] == Response.ERROR
    assert "already running" in refusal["message"]

    be.release.set()
    assert _drain_reply(be, 1, seen)["type"] == Response.SUCCESS


def test_a_reply_read_by_the_wrong_waiter_is_not_lost():
    """Two threads wait on one response queue; neither may eat the other's reply."""
    be = Backend(group_id="grp", response_queue=mp.Queue())
    results = {}

    def _call(name, command):
        results[name] = be._request(command, timeout=15)

    stop = threading.Thread(target=_call, args=("stop", IPCCommand.STOP_STREAMING))
    stop.start()
    stop_cmd = be.command_queue.get(timeout=5)

    balance = threading.Thread(
        target=_call, args=("balance", IPCCommand.RUN_DPIC_BALANCE)
    )
    balance.start()
    balance_cmd = be.command_queue.get(timeout=5)

    be.response_queue.put(
        {
            "request_id": balance_cmd["request_id"],
            "type": Response.SUCCESS,
            "for": "balance",
        }
    )
    be.response_queue.put(
        {"request_id": stop_cmd["request_id"], "type": Response.SUCCESS, "for": "stop"}
    )

    stop.join(timeout=15)
    balance.join(timeout=15)

    assert results["stop"]["for"] == "stop"
    assert results["balance"]["for"] == "balance"


def test_balancer_stops_at_the_next_point_when_aborted():
    aborted = {"value": False}
    visited = []

    channel = DpicChannel(
        inject_tx=1,
        measure_tx=0,
        measure_rx=0,
        set_phase=lambda v: visited.append(("phase", v)),
        set_amplitude=lambda v: visited.append(("amp", v)),
        read_metric=lambda: 1.0,
    )
    balancer = DpicBalancer(should_abort=lambda: aborted["value"])

    aborted["value"] = True
    result = balancer.balance(channel)
    assert result.converged is False
    assert result.best_amplitude == channel.start_amplitude
