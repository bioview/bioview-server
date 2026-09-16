"""The IF band-pass is a live setting, not a start-up constant."""

import numpy as np
import pytest
from bioview_common import DataSource

from bioview_server.device.usrp.backend import USRPBackend
from bioview_server.device.usrp.process import ProcessWorker


SAMP_RATE = 1e6


def _source(channel, tx_idx=0, rx_idx=0):
    source = DataSource(group_id="grp", channel=channel, label=f"Tx{tx_idx}Rx{rx_idx}")
    source.tx_idx = tx_idx
    source.rx_idx = rx_idx
    source.component = "amplitude"
    return source


def _worker(**kwargs):
    return ProcessWorker(
        data_sources=[_source(0)],
        cal_ref_sources=[],
        samp_rate=SAMP_RATE,
        channel_ifs=[100e3, 110e3],
        if_filter_bw=[5e3, 5e3],
        rx_queues={},
        rx_device_order=["dev"],
        schemes_by_device={},
        global_tx_to_device={0: ("dev", 0), 1: ("dev", 1)},
        **kwargs,
    )


def _response_at(sos, freq):
    """Steady-state gain of a filter at one frequency."""
    from scipy import signal

    _w, h = signal.sosfreqz(sos, worN=[2 * np.pi * freq / SAMP_RATE])
    return float(abs(h[0]))


def test_the_shipped_default_is_the_second_order_elliptic_it_always_was():
    worker = _worker()
    assert worker.if_filter_type == "ellip"
    assert worker.if_filter_order == 2


def test_widening_the_band_passes_a_tone_the_old_one_attenuated():
    worker = _worker()
    probe = 125e3
    narrow = _response_at(worker.if_filts[0], probe)
    assert narrow < 0.2

    assert worker.set_filter_params(bandwidths=60e3) is True
    assert _response_at(worker.if_filts[0], probe) > 0.9


def test_a_scalar_bandwidth_applies_to_every_channel():
    worker = _worker()
    worker.set_filter_params(bandwidths=12e3)
    assert worker.if_filter_bw == [12e3, 12e3]


def test_a_shorter_list_leaves_the_channels_it_does_not_mention_alone():
    """Narrowing an unmentioned channel to the last value given would silently"""
    worker = _worker()
    worker.set_filter_params(bandwidths=[8e3])
    assert worker.if_filter_bw == [8e3, 5e3]


def test_the_response_type_and_order_are_both_applied():
    worker = _worker()
    assert worker.set_filter_params(ftype="butter", order=4) is True
    assert (worker.if_filter_type, worker.if_filter_order) == ("butter", 4)
    assert worker.if_filts[0].shape[0] == 4
    assert _worker().if_filts[0].shape[0] == 2


def test_setting_the_same_values_again_changes_nothing():
    worker = _worker()
    before = worker.if_filts
    assert worker.set_filter_params(ftype="ellip", order=2) is False
    assert worker.if_filts is before


def test_filter_state_is_dropped_so_the_old_passband_does_not_leak_through():
    worker = _worker()
    source = worker.mimo_sources[0]
    source.filter_state = np.ones((1, 2))

    worker.set_filter_params(bandwidths=9e3)
    assert source.filter_state is None


def test_a_band_wider_than_the_spectrum_is_clamped_rather_than_refused():
    """The passband cannot extend past Nyquist, so it stops there."""
    worker = _worker()
    assert worker.set_filter_params(bandwidths=10 * SAMP_RATE) is True
    assert _response_at(worker.if_filts[0], 100e3) > 0.5


def test_a_design_that_fails_leaves_every_running_filter_untouched(monkeypatch):
    """Half-updated filters would demodulate channels through different shapes."""
    worker = _worker()
    before = worker.if_filts

    def _explode(*_args, **_kwargs):
        raise ValueError("filter design failed")

    monkeypatch.setattr(worker, "_load_filter", _explode)

    with pytest.raises(ValueError):
        worker.set_filter_params(bandwidths=9e3, ftype="butter", order=6)

    assert worker.if_filts is before
    assert (worker.if_filter_bw, worker.if_filter_type, worker.if_filter_order) == (
        [5e3, 5e3],
        "ellip",
        2,
    )


class _Backend:
    """A USRPBackend with only what the filter-parameter path touches."""

    def __init__(self, worker):
        self.logger = None
        self.process_worker = worker
        self.group_config = {}
        self.channel_ifs = [100e3, 110e3]
        self.if_filter_bw = [5e3, 5e3]
        self.hardware = {"dev": {"tx_channels": [0, 1], "if_filter_bw": [5e3, 5e3]}}

    _apply_filter_param = USRPBackend._apply_filter_param
    _queue_param_update = USRPBackend._queue_param_update


def test_a_filter_parameter_reaches_the_worker_and_the_config():
    worker = _worker()
    backend = _Backend(worker)

    backend._apply_filter_param("if_filter_type", "butter")
    backend._apply_filter_param("if_filter_order", 3)
    backend._apply_filter_param("if_filter_bw", 15e3)

    assert worker.if_filter_type == "butter"
    assert worker.if_filter_order == 3
    assert worker.if_filter_bw == [15e3, 15e3]
    assert backend.group_config["if_filter_type"] == "butter"
    assert backend.if_filter_bw == [15e3, 15e3]


def test_a_live_width_survives_a_channel_map_reload():
    """populate_data_sources() reads the widths out of ``hardware``."""
    worker = _worker()
    backend = _Backend(worker)

    backend._apply_filter_param("if_filter_bw", 15e3)

    assert backend.hardware["dev"]["if_filter_bw"] == [15e3, 15e3]


def test_a_rejected_filter_parameter_does_not_reach_the_config(monkeypatch):
    """The panel must never describe a filter that is not running."""
    worker = _worker()
    backend = _Backend(worker)

    def _refuse(**_kwargs):
        raise ValueError("nope")

    monkeypatch.setattr(worker, "set_filter_params", _refuse)
    backend._apply_filter_param("if_filter_bw", 25e3)

    assert "if_filter_bw" not in backend.group_config
    assert worker.if_filter_bw == [5e3, 5e3]


def test_before_connect_the_setting_is_kept_for_the_worker_that_is_not_built_yet():
    backend = _Backend(worker=None)
    backend.process_worker = None

    backend._apply_filter_param("if_filter_bw", 7e3)

    assert backend.group_config["if_filter_bw"] == 7e3
    assert backend.if_filter_bw == [7e3, 7e3]


def test_filter_params_are_not_forwarded_to_the_radio_command_queues():
    """Nothing in a transmit or receive worker reads these."""
    worker = _worker()
    backend = _Backend(worker)
    backend.tx_command_queue = {"dev": []}
    backend.rx_command_queue = {"dev": []}
    backend.hardware = {}

    backend._queue_param_update({"if_filter_order": 3})

    assert backend.tx_command_queue["dev"] == []
    assert backend.rx_command_queue["dev"] == []
    assert worker.if_filter_order == 3
