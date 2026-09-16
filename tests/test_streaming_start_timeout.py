"""A device that fails to stream must be reported in seconds, not minutes."""

import time

import pytest
from bioview_common import DeviceError, IPCCommand, Response

from bioview_server.datatypes.backend import (
    CONNECT_TIMEOUT,
    START_STREAMING_TIMEOUT,
    STOP_STREAMING_TIMEOUT,
    Backend,
)


class _SilentBackend(Backend):
    """A backend whose child process never answers. Never started, so `pid` is"""

    def _initialize(self):  # pragma: no cover
        raise AssertionError("child code must not run in-process")


def test_start_streaming_gives_up_within_a_few_seconds():
    backend = _SilentBackend("USRP")

    started = time.monotonic()
    with pytest.raises(DeviceError) as excinfo:
        backend.start_streaming({})
    elapsed = time.monotonic() - started

    assert "START_STREAMING" in str(excinfo.value)
    assert "USRP" in str(excinfo.value)
    assert START_STREAMING_TIMEOUT <= elapsed < START_STREAMING_TIMEOUT + 2


def test_start_is_far_stricter_than_opening_a_device():
    assert START_STREAMING_TIMEOUT <= 10
    assert START_STREAMING_TIMEOUT < STOP_STREAMING_TIMEOUT < CONNECT_TIMEOUT


def test_a_late_reply_still_reaches_the_caller():
    """The short timeout must not turn into a dropped answer."""
    backend = _SilentBackend("BIOPAC")
    backend.response_queue.put(
        {"type": Response.SUCCESS, "result": True, "request_id": 1}
    )

    response = backend.start_streaming({})

    assert response["type"] is Response.SUCCESS
    sent = backend.command_queue.get(timeout=5)
    assert sent["command"] is IPCCommand.START_STREAMING
    assert sent["request_id"] == 1
