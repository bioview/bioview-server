"""No worker may be resumed before every worker thread has been created.

``Thread.start()`` waits, without a timeout, for the new thread to be scheduled
and set its started event. The USRP bring-up used to interleave start and resume
-- transmit up and running, receive up and running, *then* create the process
worker -- so the last thread was created only once two threads were already
spinning inside tight UHD calls, and it never got the GIL to finish starting.

The backend then never answered START_STREAMING. After 90 s the server gave up,
and because a partially started session is refused outright it also stopped
every other device: a healthy BIOPAC in the same session plotted nothing, which
is how this was first reported.
"""

import pytest

from bioview_server.device.usrp.backend import USRPBackend


class _FakeWorker:
    """Records bring-up events on a shared log, in order."""

    def __init__(self, name, log):
        self.name = name
        self.log = log
        self._alive = False

    def is_alive(self):
        return self._alive

    def start(self):
        self._alive = True
        self.log.append(("start", self.name))

    def resume(self):
        self.log.append(("resume", self.name))


def _backend():
    """A USRPBackend with the bring-up state filled in and no radio behind it."""
    backend = object.__new__(USRPBackend)
    backend.logger = None
    log = []

    backend.transmit_workers = {"MyB210_3": _FakeWorker("tx", log)}
    backend.receive_workers = {"MyB210_3": _FakeWorker("rx", log)}
    backend.process_worker = _FakeWorker("process", log)
    backend.save_worker = _FakeWorker("save", log)
    backend.display_worker = _FakeWorker("display", log)

    backend.tx_command_queue = {}
    backend._cal_enabled = False
    return backend, log


@pytest.fixture(autouse=True)
def _no_filling_delay(monkeypatch):
    monkeypatch.setattr("bioview_server.device.usrp.backend.FILLING_TIME", 0)


def test_every_thread_is_created_before_any_of_them_runs():
    backend, log = _backend()

    assert backend._start_streaming() is True

    starts = [i for i, (event, _) in enumerate(log) if event == "start"]
    resumes = [i for i, (event, _) in enumerate(log) if event == "resume"]

    assert starts and resumes
    assert max(starts) < min(resumes), (
        "a thread was created after another had already been resumed; "
        f"order was {log}"
    )


def test_all_five_workers_are_brought_up():
    backend, log = _backend()
    backend._start_streaming()

    started = {name for event, name in log if event == "start"}
    resumed = {name for event, name in log if event == "resume"}
    assert started == {"tx", "rx", "process", "save", "display"}
    assert resumed == started


def test_the_radio_is_resumed_before_the_consumers():
    """The process worker drains the Rx queues; DPIC has no metrics until then."""
    backend, log = _backend()
    backend._start_streaming()

    order = [name for event, name in log if event == "resume"]
    assert order.index("process") < order.index("display")
    assert order.index("tx") < order.index("process")


def test_a_worker_that_is_already_alive_is_not_restarted():
    """Start/Stop pauses threads rather than tearing them down."""
    backend, log = _backend()
    backend._start_streaming()
    log.clear()

    backend._start_streaming()

    assert not [event for event, _ in log if event == "start"]
    assert len([event for event, _ in log if event == "resume"]) == 5


def test_optional_workers_may_be_absent():
    """Saving is off by default; the display worker is built at Start."""
    backend, log = _backend()
    backend.save_worker = None
    backend.display_worker = None

    assert backend._start_streaming() is True
    assert {name for event, name in log if event == "start"} == {"tx", "rx", "process"}
