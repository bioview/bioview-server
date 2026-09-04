"""Streaming with more than one device group in the session.

Backends used to share a single response queue. A reply carries no sender, so
with two devices in one session each could consume the other's answer: the
parent then waited out its timeout on a reply that had already been taken, and
the bare ``queue.Empty`` surfaced to the user as "Failed to start streaming: "
with nothing after the colon.

Each backend now has its own queue and every reply is matched by request id.
"""

import multiprocessing as mp

import pytest
from bioview_common import Command, DummyConfiguration, IPCCommand, Response

from bioview_server.datatypes import Backend


NUM_CHANNELS = 2


def _dummy(samp_rate: int) -> dict:
    return DummyConfiguration.from_dict(
        {
            "type": "DUMMY",
            "samp_rate": samp_rate,
            "num_channels": NUM_CHANNELS,
            "signal_freq": 1.0,
            "amplitude": 1.0,
            "noise_std": 0.0,
            "chunk_duration": 0.05,
        }
    ).to_dict()


TWO_GROUPS = {"DeviceA": _dummy(500), "DeviceB": _dummy(1000)}


# --------------------------------------------------------------------------
# The IPC contract
# --------------------------------------------------------------------------


class _Backend(Backend):
    """A backend object driven from the parent side only (never started)."""

    def populate_data_sources(self):
        self.data_sources = set()


def test_each_backend_gets_its_own_response_queue():
    a, b = _Backend("A"), _Backend("B")
    assert a.response_queue is not b.response_queue


def test_a_reply_meant_for_another_request_is_not_consumed():
    """A late answer to a timed-out request must not be handed to the next one."""
    backend = _Backend("A")

    # Stand-in for a reply to request 1 that arrived after request 1 gave up.
    backend.response_queue.put(
        {"type": Response.SUCCESS, "result": True, "request_id": 1}
    )
    backend.response_queue.put(
        {"type": Response.SUCCESS, "result": "mine", "request_id": 2}
    )
    backend._request_id = 1

    response = backend._request(IPCCommand.STOP_STREAMING, timeout=5)
    assert response["result"] == "mine"


def test_a_timeout_reports_which_device_and_which_command():
    """The old failure was an empty message, which told the user nothing."""
    backend = _Backend("BIOPAC")

    with pytest.raises(Exception) as excinfo:
        backend._request(IPCCommand.START_STREAMING, timeout=0.2)

    message = str(excinfo.value)
    assert message, "a timeout must never surface as an empty message"
    assert "BIOPAC" in message
    assert "START_STREAMING" in message


def test_a_backend_error_is_raised_with_its_message():
    backend = _Backend("USRP")
    backend.response_queue.put(
        {"type": Response.ERROR, "message": "mpdev refused the handle", "request_id": 1}
    )

    with pytest.raises(Exception, match="mpdev refused the handle"):
        backend.start_streaming({})


def test_an_error_with_no_message_still_says_something():
    backend = _Backend("USRP")
    backend.response_queue.put({"type": Response.ERROR, "message": "", "request_id": 1})

    with pytest.raises(Exception) as excinfo:
        backend.start_streaming({})
    assert "USRP" in str(excinfo.value)


def test_the_child_echoes_the_request_id_back():
    backend = _Backend("A")
    backend.logger = None
    backend.response_queue = mp.Queue()
    backend._streaming = mp.Event()
    backend._start_streaming = lambda: True
    backend._setup_display = lambda cfg: None

    backend._handle_command(
        {"command": IPCCommand.START_STREAMING, "args": {}, "request_id": 77}
    )
    assert backend.response_queue.get(timeout=5)["request_id"] == 77


# --------------------------------------------------------------------------
# End to end, through the real server
# --------------------------------------------------------------------------


def test_two_device_groups_stream_together(client):
    resp_type, payload = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": TWO_GROUPS}
    )
    assert resp_type == Response.SUCCESS.name, payload
    assert set(payload["device_status"]) == {"DeviceA", "DeviceB"}

    resp_type, payload = client.command(
        Command.START_STREAMING,
        {"Experiment": {"type": "EXPERIMENT"}, **TWO_GROUPS},
    )
    assert resp_type == Response.SUCCESS.name, payload

    # Both devices must actually reach the wire, not just one of them.
    seen = set()
    for _ in range(40):
        _, sources = client.recv_data_chunk(timeout=5.0)
        for src in sources or []:
            seen.add(src.get("group_id"))
        if {"DeviceA", "DeviceB"} <= seen:
            break

    assert {"DeviceA", "DeviceB"} <= seen, f"only saw {seen}"

    resp_type, _ = client.command(Command.STOP_STREAMING)
    assert resp_type == Response.SUCCESS.name


def test_sources_from_two_devices_stay_distinguishable(client):
    """Channel labels repeat across devices; the group id is what separates them."""
    resp_type, payload = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": TWO_GROUPS}
    )
    assert resp_type == Response.SUCCESS.name, payload

    sources = payload.get("data_sources", [])
    assert len(sources) == 2 * NUM_CHANNELS

    keys = {(s["group_id"], s["channel"]) for s in sources}
    assert len(keys) == len(sources), "sources must be unique per (group, channel)"
    assert {s["group_id"] for s in sources} == {"DeviceA", "DeviceB"}


def test_a_dead_child_is_reported_as_a_crash_not_a_timeout():
    """A native crash inside a driver leaves no traceback and no reply.

    Waiting out the full timeout and then blaming the device for being slow
    ("did not answer START_STREAMING within 90s") sends the reader looking at
    the wrong thing entirely.
    """

    class _DeadChild(_Backend):
        """A started process that has since exited. ``pid``, ``is_alive`` and
        ``exitcode`` are read-only on ``mp.Process``, so they are overridden."""

        pid = 4242
        exitcode = -1073741819  # 0xC0000005, a Windows access violation

        def is_alive(self):
            return False

    backend = _DeadChild("USRP")

    with pytest.raises(Exception) as excinfo:
        backend._request(IPCCommand.START_STREAMING, timeout=30)

    message = str(excinfo.value)
    assert "USRP" in message
    assert "exited" in message
    assert "START_STREAMING" in message
    assert "-1073741819" in message


def test_an_unstarted_backend_still_times_out_normally():
    """``is_alive()`` is False before ``start()`` too; that is not a crash."""
    backend = _Backend("BIOPAC")
    assert backend.pid is None

    with pytest.raises(Exception, match="did not answer"):
        backend._request(IPCCommand.START_STREAMING, timeout=0.2)
