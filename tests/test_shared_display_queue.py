"""The output queue is shared by every backend in a session.

``Server`` creates one ``data_queue`` and hands the same object to every device
handler, and one server thread drains it. A backend that drained that queue on
its own Start therefore threw away whatever the *other* devices had already
queued -- data loss that only appears once two devices stream together.
"""

import multiprocessing as mp
import queue as _queue

from bioview_common import DATA_OUTPUT_QUEUE_DEPTH, DataSource

from bioview_server.datatypes import Backend


class _Backend(Backend):
    """Driven from the parent side only (never started)."""

    # Set by run() in the child; these objects are never started.
    logger = None

    def populate_data_sources(self):
        self.data_sources = set()


def _drain_all(q, limit=50):
    out = []
    for _ in range(limit):
        try:
            out.append(q.get(timeout=0.5))
        except _queue.Empty:
            break
    return out


def test_one_device_starting_does_not_discard_another_devices_chunks():
    shared = mp.Queue(maxsize=DATA_OUTPUT_QUEUE_DEPTH)

    biopac = _Backend("BIOPAC", data_output_queue=shared)
    usrp = _Backend("USRP", data_output_queue=shared)

    # BIOPAC is already streaming and has queued chunks for the server thread.
    for i in range(3):
        shared.put({"data": i, "sources": []})

    # USRP now starts. This must not touch BIOPAC's queued data.
    usrp._setup_display({})

    assert [item["data"] for item in _drain_all(shared)] == [0, 1, 2]
    assert biopac.data_output_queue is shared


def test_the_display_worker_is_reused_across_a_restart():
    """Replacing it leaks the old thread onto the same input queue."""
    backend = _Backend("USRP", data_output_queue=mp.Queue(maxsize=4))
    backend.data_sources = {DataSource("USRP", 0, "Tx1Rx1")}

    backend._setup_display({})
    first = backend.display_worker
    assert first is not None

    backend.data_sources = {
        DataSource("USRP", 0, "Tx1Rx1"),
        DataSource("USRP", 1, "Tx1Rx2"),
    }
    backend._setup_display({})

    assert backend.display_worker is first
    assert len(first.display_sources) == 2


def test_display_sources_defaults_to_the_advertised_sources():
    backend = _Backend("BIOPAC")
    backend.data_sources = {DataSource("BIOPAC", 0, "Ch1")}
    assert backend._display_sources() == [DataSource("BIOPAC", 0, "Ch1")]
