"""BIOPAC display plumbing and live channel changes.

Three bugs lived here at once, and they all had the same visible symptom --
nothing on the plots:

* ``_start_streaming`` never started the shared DisplayWorker, so the
  acquisition worker filled the display queue and nothing drained it into the
  client's output queue. USRP and dummy have always started it.
* Every sample is forwarded to the display, but the sources advertised the
  default 200 Hz display rate rather than the sample rate, so the plot window
  was sized for the wrong rate.
* mpdev returns one value per *enabled* channel; the worker was told to read one
  per channel in the mask, appending uninitialized doubles to each chunk
  whenever fewer than four channels were on.

Plus the reason a channel change never reached the plot-source selector: the
parameter update is applied in the backend's child process, while
``get_data_sources()`` is answered by the parent.
"""

import multiprocessing as mp

import pytest
from bioview_common import DeviceStatus

from bioview_server.device.biopac.backend import BIOPACBackend


GROUP_CFG = {
    "device_name": "BIOPAC",
    "model": "MP36",
    "samp_rate": 500,
    "channels": [1, 1, 0, 0],
    "hardware": {"BIOPAC_MP36": {"channels": [1, 1, 0, 0]}},
}


class _Worker:
    """Stand-in for a PausableWorker, recording what the backend did to it."""

    def __init__(self):
        self.started = False
        self.resumed = False
        self.paused = False
        self.sources = None

    def is_alive(self):
        return self.started

    def start(self):
        self.started = True

    def resume(self):
        self.resumed = True

    def pause(self):
        self.paused = True

    def set_display_sources(self, sources):
        self.sources = list(sources)


@pytest.fixture
def backend():
    be = BIOPACBackend(
        group_id="BIOPAC",
        response_queue=mp.Queue(),
        group_config=dict(GROUP_CFG),
    )
    # run() installs this in the child process; these tests drive the object
    # directly, so give it one up front.
    be.logger = None
    return be


def _labels(backend):
    return sorted(src.label for src in backend.get_data_sources())


def test_only_enabled_channels_are_advertised(backend):
    assert _labels(backend) == ["Ch1", "Ch2"]


def test_sources_report_the_real_display_rate(backend):
    # Every acquired sample reaches the display, so the display rate is the
    # sample rate -- not DataSource's 200 Hz default.
    assert {src.get_disp_freq() for src in backend.get_data_sources()} == {500.0}


def test_acquisition_reads_one_value_per_enabled_channel(backend):
    assert backend._acquired_channel_indices() == [0, 1]

    backend._queue_param_update({"channels": [1, 1, 1, 1]})
    assert backend._acquired_channel_indices() == [0, 1, 2, 3]


def test_start_streaming_runs_the_display_worker(backend, monkeypatch):
    monkeypatch.setattr(
        "bioview_server.device.biopac.backend.start_acquisition", lambda *_: True
    )
    monkeypatch.setattr(
        "bioview_server.device.biopac.backend.BiopacAcquisitionWorker",
        lambda **kwargs: _Worker(),
    )

    backend.mpdev_handler = object()
    backend.display_worker = _Worker()
    backend.save_worker = _Worker()

    assert backend._start_streaming() is True
    assert backend.display_worker.started and backend.display_worker.resumed
    assert backend.save_worker.started and backend.save_worker.resumed
    assert backend.status == DeviceStatus.STREAMING


def test_stop_streaming_pauses_the_display_worker(backend):
    backend.display_worker = _Worker()
    backend.save_worker = _Worker()

    assert backend._stop_streaming() is True
    assert backend.display_worker.paused
    assert backend.save_worker.paused


def test_changing_channels_relabels_the_display_rows(backend):
    backend.display_worker = _Worker()

    backend._queue_param_update({"channels": [1, 0, 1, 0]})

    assert _labels(backend) == ["Ch1", "Ch2"]
    # Two rows, and the worker knows which sources they now describe.
    assert len(backend.display_worker.sources) == 2
    assert backend._acquired_channel_indices() == [0, 2]


def test_channel_change_is_mirrored_on_the_parent_side(backend):
    """queue_param_update() is answered by the child; get_data_sources() by the
    parent. Without the mirror the server keeps advertising the old channels."""
    backend.queue_param_update(channels=[1, 1, 1, 1])
    assert _labels(backend) == ["Ch1", "Ch2", "Ch3", "Ch4"]

    backend.queue_param_update(channels=[0, 0, 0, 1])
    assert _labels(backend) == ["Ch1"]


def test_parent_side_sample_rate_change_updates_the_display_rate(backend):
    backend.queue_param_update(samp_rate=2000)
    assert {src.get_disp_freq() for src in backend.get_data_sources()} == {2000.0}
