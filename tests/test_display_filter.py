"""The display filter shapes the plots and nothing else.

It exists so the operator can see roughly what preprocessing will do to a live
trace. That makes "the recording is untouched" the property worth pinning, and
the fact that its cutoffs are in the *display* rate's Hz rather than anywhere
near the IF.
"""

import numpy as np
import pytest
from bioview_common import DataSource
from bioview_common.signal_schemes import CwScheme

from bioview_server.device.usrp.process import ProcessWorker


SAMP_RATE = 1e6
IF_HZ = 100e3
SAVE_DS = 100
DISPLAY_DS = 1
DISPLAY_RATE = SAMP_RATE / (SAVE_DS * DISPLAY_DS)  # 10 kHz


def _worker(**kwargs):
    source = DataSource(group_id="g", channel=0, label="Tx1Rx1")
    source.tx_idx = source.rx_idx = 0
    scheme = CwScheme(SAMP_RATE, [IF_HZ], [1.0], [0.0])
    return ProcessWorker(
        data_sources={source},
        cal_ref_sources=[],
        samp_rate=SAMP_RATE,
        channel_ifs=[IF_HZ],
        if_filter_bw=[5e3],
        rx_queues={"d": None},
        rx_device_order=["d"],
        schemes_by_device={"d": scheme},
        global_tx_to_device={0: ("d", 0)},
        save_ds=SAVE_DS,
        display_ds=DISPLAY_DS,
        **kwargs,
    )


def _tone(n, freq, rate=DISPLAY_RATE):
    t = np.arange(n) / rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32).reshape(1, n)


def _settled_gain(worker, freq, n=20_000):
    """Output amplitude at `freq`, measured past the filter's transient."""
    payload = _tone(n, freq)
    out = worker._filter_display(payload)
    return float(np.abs(out[0, n // 2 :]).max())


def test_off_is_a_pass_through():
    worker = _worker(disp_filter_btype="off")
    payload = _tone(1000, 5.0)
    assert worker._filter_display(payload) is payload


@pytest.mark.parametrize(
    ("btype", "passes", "blocks"),
    [
        ("low", 2.0, 300.0),
        ("high", 300.0, 2.0),
    ],
)
def test_a_single_cutoff_keeps_the_right_side(btype, passes, blocks):
    worker = _worker(
        disp_filter_btype=btype,
        disp_filter_low=40.0,
        disp_filter_high=40.0,
        disp_filter_order=4,
    )
    assert _settled_gain(worker, passes) > 0.7

    worker = _worker(
        disp_filter_btype=btype,
        disp_filter_low=40.0,
        disp_filter_high=40.0,
        disp_filter_order=4,
    )
    assert _settled_gain(worker, blocks) < 0.1


def test_band_pass_keeps_only_what_is_between_the_cutoffs():
    for freq, expected in ((0.2, False), (20.0, True), (500.0, False)):
        worker = _worker(
            disp_filter_btype="band",
            disp_filter_low=2.0,
            disp_filter_high=100.0,
            disp_filter_order=4,
        )
        gain = _settled_gain(worker, freq)
        assert (gain > 0.7) is expected, f"{freq} Hz -> {gain:.3f}"


def test_the_cutoffs_are_in_the_display_rate_s_own_hz():
    """40 Hz means 40 Hz of plotted signal, not 40 Hz at the 1 MSps front end."""
    worker = _worker(disp_filter_btype="low", disp_filter_high=40.0)
    assert worker.get_display_rate() == pytest.approx(DISPLAY_RATE)


def test_the_saved_stream_is_untouched():
    """The whole point: a display filter must never reach the recording."""
    n = 40_000
    t = np.arange(n) / SAMP_RATE
    buffer = (
        (0.5 * np.exp(1j * 2 * np.pi * IF_HZ * t)).astype(np.complex64).reshape(1, n)
    )

    plain = _worker(disp_filter_btype="off")
    filtered = _worker(
        disp_filter_btype="high", disp_filter_low=200.0, disp_filter_order=4
    )

    plain_save, _ = plain._assemble_outputs(buffer, plain._process_mimo_chunk(buffer))
    filt_save, _ = filtered._assemble_outputs(
        buffer, filtered._process_mimo_chunk(buffer)
    )
    np.testing.assert_array_equal(plain_save, filt_save)


def test_state_carries_between_chunks():
    """A per-chunk filter would ring at every chunk boundary."""
    worker = _worker(disp_filter_btype="low", disp_filter_high=50.0, disp_filter_order=4)
    whole = _tone(4000, 5.0)
    one_pass = worker._filter_display(whole)

    worker = _worker(disp_filter_btype="low", disp_filter_high=50.0, disp_filter_order=4)
    chunked = np.hstack(
        [worker._filter_display(whole[:, i : i + 400]) for i in range(0, 4000, 400)]
    )
    np.testing.assert_allclose(chunked, one_pass, rtol=1e-4, atol=1e-5)


def test_a_live_change_takes_effect_and_reports_it():
    worker = _worker(disp_filter_btype="off")
    assert worker.set_display_filter(btype="low", high=25.0) is True
    assert worker.set_display_filter(btype="low", high=25.0) is False
    assert _settled_gain(worker, 500.0) < 0.1


def test_a_refused_setting_leaves_the_running_filter_alone():
    worker = _worker(
        disp_filter_btype="band", disp_filter_low=2.0, disp_filter_high=100.0
    )
    with pytest.raises(ValueError):
        worker.set_display_filter(low=500.0)  # above the high cutoff

    assert worker.disp_filter_low == 2.0
    assert worker.disp_filter_high == 100.0
    assert _settled_gain(worker, 20.0) > 0.7


def test_a_new_channel_map_resets_the_filter_state():
    """Row count changes, so the held state no longer matches the payload."""
    worker = _worker(disp_filter_btype="low", disp_filter_high=50.0)
    worker._filter_display(_tone(400, 5.0))
    assert worker._disp_zi is not None

    source = DataSource(group_id="g", channel=0, label="Tx1Rx1")
    source.tx_idx = source.rx_idx = 0
    worker.set_sources({source}, [], [IF_HZ], [5e3])
    assert worker._disp_zi is None


class _Backend:
    """A USRPBackend with only what the parameter path touches."""

    def __init__(self, worker):
        self.logger = None
        self.process_worker = worker
        self.group_config = {}
        self.channel_ifs = [IF_HZ]
        self.if_filter_bw = [5e3]
        self.hardware = {"dev": {"tx_channels": [0], "if_filter_bw": [5e3]}}

    from bioview_server.device.usrp.backend import USRPBackend

    _apply_display_filter_param = USRPBackend._apply_display_filter_param
    _apply_filter_param = USRPBackend._apply_filter_param
    _queue_param_update = USRPBackend._queue_param_update


def test_the_panel_s_settings_reach_the_worker_and_the_config():
    worker = _worker(disp_filter_btype="off")
    backend = _Backend(worker)

    backend._queue_param_update(
        {
            "disp_filter_btype": "band",
            "disp_filter_low": 1.0,
            "disp_filter_high": 30.0,
            "disp_filter_order": 3,
        }
    )

    assert worker.disp_filter_btype == "band"
    assert worker.disp_filter_low == 1.0
    assert worker.disp_filter_high == 30.0
    assert worker.disp_filter_order == 3
    assert backend.group_config["disp_filter_btype"] == "band"
    assert _settled_gain(worker, 10.0) > 0.7


def test_a_rejected_setting_does_not_reach_the_config():
    """The panel must never describe a filter that is not running."""
    worker = _worker(
        disp_filter_btype="band", disp_filter_low=2.0, disp_filter_high=100.0
    )
    backend = _Backend(worker)

    backend._apply_display_filter_param("disp_filter_low", 500.0)

    assert "disp_filter_low" not in backend.group_config
    assert worker.disp_filter_low == 2.0


def test_the_display_filter_is_not_confused_with_the_if_filter():
    """Two separate settings; tuning one must not disturb the other."""
    worker = _worker(disp_filter_btype="low", disp_filter_high=40.0)
    backend = _Backend(worker)

    backend._queue_param_update({"if_filter_bw": 15e3})

    assert worker.disp_filter_btype == "low"
    assert worker.disp_filter_high == 40.0
    assert worker.if_filter_bw == [15e3]
