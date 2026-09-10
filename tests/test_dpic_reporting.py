"""A DPIC balance that does not run must never reach the client as a success.

Every early-out in ``_run_dpic_balance`` used to return None, which the IPC
layer turned into ``Response.SUCCESS``, which the server turned into "DPIC
balance complete". The UI reported a finished balance in about 2 ms while
nothing had been driven at all.
"""

import multiprocessing as mp

import pytest
from bioview_common import Response
from bioview_common.signal_schemes.dpic import DpicBalancer, DpicChannel
from fakes.backend import FakeBackend

from bioview_server.common import balance_outcome


def _backend(group_config=None):
    return FakeBackend(
        group_id="grp",
        response_queue=mp.Queue(),
        data_output_queue=mp.Queue(),
        group_config=group_config or {},
    )


def test_balance_without_pairs_reports_a_reason():
    outcome = _backend()._run_dpic_balance()
    assert outcome["ok"] is False
    assert outcome["message"]
    assert outcome["results"] == []


def test_balance_without_a_running_worker_reports_a_reason():
    be = _backend(
        {
            "hardware": {
                "A": {"tx_channels": [0, 1], "rx_channels": [0, 1]},
                "B": {"tx_channels": [0], "rx_channels": [0]},
            },
            "channel_map": {
                "layout": "hybrid_mimo",
                "mimo": {"tx_global": [0, 1], "rx_global": [0, 1]},
                "dpic": [{"inject_tx": 2, "measure_tx": 0, "measure_rx": 0}],
            },
        }
    )
    outcome = be._run_dpic_balance()
    assert outcome["ok"] is False
    assert "start streaming" in outcome["message"].lower()


def test_base_backend_default_is_a_failure_not_a_none():
    from bioview_server.datatypes.backend import Backend

    outcome = Backend._run_dpic_balance(type("X", (), {"group_id": "grp"})())
    assert outcome["ok"] is False
    assert "grp" in outcome["message"]


def _silent_channel():
    return DpicChannel(
        inject_tx=1,
        measure_tx=0,
        measure_rx=0,
        set_phase=lambda v: None,
        set_amplitude=lambda v: None,
        read_metric=lambda: None,
        start_phase_deg=30.0,
        start_amplitude=0.7,
    )


def test_a_silent_measurement_path_is_reported_not_swallowed():
    result = DpicBalancer().balance(_silent_channel())
    assert not result.converged
    assert "no metric" in result.message

    outcome = balance_outcome(None, [result])
    assert outcome["ok"] is False
    assert "no metric" in outcome["message"]


def test_an_expired_budget_says_so_rather_than_looking_instant():
    """The 2 ms balance: every sweep broke on its first point, silently."""
    ch = DpicChannel(
        inject_tx=1,
        measure_tx=0,
        measure_rx=0,
        set_phase=lambda v: None,
        set_amplitude=lambda v: None,
        read_metric=lambda: 1.0,
    )
    # Negative budget: the deadline is already past when the search starts.
    result = DpicBalancer(time_budget_s=-1.0).balance(ch)

    assert not result.converged
    assert "budget expired" in result.message
    # The stages are still reported, showing zero points visited.
    assert [s.name for s in result.stages] == [
        "coarse phase",
        "coarse amplitude",
        "fine phase",
        "fine amplitude",
    ]
    assert all(s.visited == 0 for s in result.stages)
    assert balance_outcome(None, [result])["ok"] is False


def test_a_truncated_sweep_is_flagged():
    """A budget that runs out mid-search keeps its best point but says so."""
    seen = {"n": 0}

    def read_metric():
        seen["n"] += 1
        return 1.0 / seen["n"]

    ch = DpicChannel(
        inject_tx=1,
        measure_tx=0,
        measure_rx=0,
        set_phase=lambda v: None,
        set_amplitude=lambda v: None,
        read_metric=read_metric,
        # A fixed dwell, ignoring the duration the balancer asks for: the
        # point of this test is the budget, not the VI's timings.
        wait_settle=lambda _s: __import__("time").sleep(0.004),
    )
    result = DpicBalancer(time_budget_s=0.05).balance(ch)

    assert result.converged
    assert result.truncated
    assert result.num_measurements < 241
    assert balance_outcome(None, [result])["ok"] is True


def test_stage_trace_covers_every_sweep_on_a_completed_search():
    ch = DpicChannel(
        inject_tx=1,
        measure_tx=0,
        measure_rx=0,
        set_phase=lambda v: None,
        set_amplitude=lambda v: None,
        read_metric=lambda: 1.0,
    )
    result = DpicBalancer().balance(ch)
    assert result.converged
    assert not result.truncated
    planned = [s.planned for s in result.stages]
    assert planned == [60, 20, 60, 100]
    assert all(s.measured == s.planned for s in result.stages)
    assert result.num_measurements == 241


@pytest.mark.parametrize("ok", [True, False])
def test_ipc_reply_type_follows_the_outcome(ok):
    """The IPC reply carries ERROR for a balance that did not happen."""
    import threading

    from bioview_common import IPCCommand

    from bioview_server.datatypes.backend import Backend

    replies = []
    answered = threading.Event()

    class Stub(Backend):
        def __init__(self):
            # Skip Backend.__init__: only the command dispatch is under test.
            self.group_id = "grp"
            self.logger = None
            self._init_local_state()

        def _reply(self, request_id, payload):
            replies.append(payload)
            answered.set()

        def _run_dpic_balance(self):
            return {"ok": ok, "message": "why", "results": [{"x": 1}]}

    stub = Stub()
    stub._handle_command(
        {"command": IPCCommand.RUN_DPIC_BALANCE, "request_id": 1, "args": {}}
    )
    # The balance is answered from its own thread now, so the command loop can
    # keep serving Stop while it runs.
    assert answered.wait(timeout=5)

    assert replies[0]["type"] is (Response.SUCCESS if ok else Response.ERROR)
    assert replies[0]["message"] == "why"
    assert replies[0]["result"] == [{"x": 1}]
