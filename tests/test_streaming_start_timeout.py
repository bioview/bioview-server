"""A device that fails to stream must be reported in seconds, not minutes.

Starting a stream is near-instant once the device is open: every worker thread
already exists and is only resumed. A backend that has not answered in several
seconds is wedged, not slow. The timeout used to be 90 s, and because the server
starts devices in sequence and refuses a partially started session, one wedged
USRP meant a minute and a half of silence followed by a healthy BIOPAC being
stopped too.
"""

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
    """A backend whose child process never answers. Never started, so `pid` is
    None and the liveness check stays out of the way."""

    def _initialize(self):  # pragma: no cover - never reached
        raise AssertionError("child code must not run in-process")


def test_start_streaming_gives_up_within_a_few_seconds():
    backend = _SilentBackend("USRP")

    started = time.monotonic()
    with pytest.raises(DeviceError) as excinfo:
        backend.start_streaming({})
    elapsed = time.monotonic() - started

    assert "START_STREAMING" in str(excinfo.value)
    assert "USRP" in str(excinfo.value)
    # Bounded above by the timeout plus the 0.25 s response poll, and below it
    # so the test fails if the wait is silently skipped.
    assert START_STREAMING_TIMEOUT <= elapsed < START_STREAMING_TIMEOUT + 2


def test_start_is_far_stricter_than_opening_a_device():
    # Opening a radio is genuinely slow (USB enumeration, FPGA and clock
    # bring-up); resuming its already-running workers is not.
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
    # The command still went to the child, exactly once.
    sent = backend.command_queue.get(timeout=5)
    assert sent["command"] is IPCCommand.START_STREAMING
    assert sent["request_id"] == 1
