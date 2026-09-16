"""The advertised display rate must match the rate actually emitted."""

import queue

import numpy as np
import pytest
from bioview_common import DataSource
from bioview_common.signal_schemes import CwScheme

from bioview_server.device.usrp.process import ProcessWorker


SAMP_RATE = 1e6
IF_HZ = 100e3


def _worker(save_ds, display_ds, display_queue=None):
    source = DataSource(group_id="g", channel=0, label="Tx1Rx1")
    source.tx_idx = 0
    source.rx_idx = 0
    return ProcessWorker(
        data_sources={source},
        cal_ref_sources=[],
        samp_rate=SAMP_RATE,
        channel_ifs=[IF_HZ],
        if_filter_bw=[5e3],
        rx_queues={"d": None},
        rx_device_order=["d"],
        schemes_by_device={"d": CwScheme(SAMP_RATE, [IF_HZ], [1.0], [30.0])},
        global_tx_to_device={0: ("d", 0)},
        save_ds=save_ds,
        display_ds=display_ds,
        display_queue=display_queue,
    )


@pytest.mark.parametrize("save_ds,display_ds", [(1, 1), (10, 1), (100, 10), (50, 4)])
def test_emitted_display_length_matches_the_advertised_rate(save_ds, display_ds):
    """One second of Rx samples must yield disp_freq display points."""
    disp_q = queue.Queue(maxsize=8)
    worker = _worker(save_ds, display_ds, display_queue=disp_q)

    n_raw = int(SAMP_RATE)
    buffer = np.zeros((1, n_raw), dtype=np.complex64)
    results = worker._process_mimo_chunk(buffer)
    _, display_data = worker._assemble_outputs(buffer, results)
    emitted = worker._decimate_display(display_data)

    expected_disp_freq = SAMP_RATE / (save_ds * display_ds)
    assert emitted.shape[1] == pytest.approx(expected_disp_freq, rel=1e-9)


def test_display_decimation_averages_rather_than_drops():
    worker = _worker(save_ds=1, display_ds=4)
    payload = np.arange(8, dtype=np.float32).reshape(1, 8)
    out = worker._decimate_display(payload)
    assert out.tolist() == [[1.5, 5.5]]


def test_a_chunk_shorter_than_one_display_window_still_emits_a_point():
    worker = _worker(save_ds=1, display_ds=100)
    payload = np.ones((2, 10), dtype=np.float32)
    out = worker._decimate_display(payload)
    assert out.shape == (2, 1)


def test_display_ds_of_one_is_a_passthrough():
    worker = _worker(save_ds=1, display_ds=1)
    payload = np.arange(6, dtype=np.float32).reshape(1, 6)
    assert worker._decimate_display(payload) is payload


def test_fake_rf_backend_advertises_the_post_decimation_rate():
    """Every source a backend advertises carries the rate it really emits."""
    import multiprocessing as mp

    from fakes.backend import FakeBackend

    group_config = {
        "samp_rate": 1e6,
        "save_ds": 100,
        "disp_ds": 10,
        "hardware": {
            "SimA": {
                "tx_channels": [0, 1],
                "rx_channels": [0, 1],
                "if_freq": [100e3, 110e3],
            }
        },
        "channel_map": {"layout": "full_nxn", "dpic": []},
        "calibration": {"enabled": False, "record_reference": True},
    }
    backend = FakeBackend(
        group_id="Sim",
        group_config=group_config,
        response_queue=mp.Queue(),
    )

    expected = 1e6 / (100 * 10)
    assert backend.get_display_frequency() == expected
    sources = list(backend.mimo_sources) + list(backend.cal_ref_sources)
    assert sources
    for source in sources:
        assert source.get_disp_freq() == expected
        assert DataSource.from_dict(source.to_dict()).get_disp_freq() == expected


def test_fake_non_rf_backend_advertises_the_sample_rate():
    """The sine path forwards every sample, so disp_freq is the sample rate."""
    import multiprocessing as mp

    from fakes.backend import FakeBackend

    backend = FakeBackend(
        group_id="Sim",
        group_config={"samp_rate": 500, "num_channels": 2},
        response_queue=mp.Queue(),
    )
    assert backend.get_display_frequency() == 500.0
    for source in backend.data_sources:
        assert source.get_disp_freq() == 500.0
