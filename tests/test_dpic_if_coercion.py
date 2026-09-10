"""The inject Tx is driven at the measure Tx's IF, not at its own.

The Rx band-passes around the measure Tx's IF, so an injection anywhere else is
rejected by that filter and no phase/amplitude can cancel the direct path (see
ENGINEERING_NOTES 3.5, defect 11). A mismatch used to be logged as an error and
the balance ran regardless, sweeping 241 points against a tone the receive
chain could not see. The frequency follows from which pair is being cancelled,
so balance coerces it.
"""

import copy
import multiprocessing as mp

import pytest
from fakes.backend import FakeBackend

from bioview_server.device.usrp.backend import USRPBackend


GROUP_CONFIG = {
    "hardware": {
        "A": {
            "tx_channels": [0, 1],
            "rx_channels": [0, 1],
            # Tx1 at 100 kHz, Tx2 at 110 kHz -- the shipped two-channel default.
            "if_freq": [100e3, 110e3],
        }
    },
    "channel_map": {
        "layout": "full_nxn",
        # Inject from Tx2, measure on Tx1: the balance has to work at 100 kHz.
        "dpic": [{"inject_tx": 1, "measure_tx": 0, "measure_rx": 0}],
    },
}


def _fake():
    return FakeBackend(
        group_id="grp",
        response_queue=mp.Queue(),
        data_output_queue=mp.Queue(),
        group_config=copy.deepcopy(GROUP_CONFIG),
    )


def _usrp():
    return USRPBackend(
        group_id="grp",
        samp_rate=1e6,
        devices={},
        group_config=copy.deepcopy(GROUP_CONFIG),
        response_queue=mp.Queue(),
        data_output_queue=mp.Queue(),
    )


@pytest.mark.parametrize("make_backend", [_fake, _usrp], ids=["fake", "usrp"])
def test_inject_tx_is_moved_onto_the_measure_tx_if(make_backend):
    be = make_backend()
    assert be.channel_ifs == [100e3, 110e3]

    be._coerce_dpic_inject_frequencies()

    # Both Tx now sit at the measure Tx's IF.
    assert be.channel_ifs == [100e3, 100e3]
    # And the config agrees, so the settings panel does not still show 110 kHz
    # for a radio that is being driven at 100 kHz.
    assert be.hardware["A"]["if_freq"] == [100e3, 100e3]
    assert be.registry.tx_if_freq == [100e3, 100e3]


@pytest.mark.parametrize("make_backend", [_fake, _usrp], ids=["fake", "usrp"])
def test_matching_ifs_are_left_alone(make_backend):
    be = make_backend()
    be.channel_ifs[1] = 100e3
    be._coerce_dpic_inject_frequencies()
    assert be.channel_ifs == [100e3, 100e3]


def test_usrp_coercion_reaches_the_transmit_workers():
    """The waveform generators must be retuned, not just the bookkeeping."""
    import queue

    be = _usrp()
    be.tx_command_queue = {"A": queue.Queue()}
    be._coerce_dpic_inject_frequencies()

    updates = []
    while not be.tx_command_queue["A"].empty():
        updates.append(be.tx_command_queue["A"].get_nowait())

    assert {"param": "if_freq", "value": [100e3, 100e3]} in updates


def test_process_worker_retune_moves_the_band_pass():
    """A retuned channel needs a new band-pass, or it rejects its own tone."""
    from bioview_server.device.usrp.process import ProcessWorker

    worker = ProcessWorker(
        data_sources=[],
        cal_ref_sources=[],
        samp_rate=1e6,
        channel_ifs=[100e3, 110e3],
        if_filter_bw=[5e3, 5e3],
        rx_queues={},
        rx_device_order=["A"],
        schemes_by_device={},
        global_tx_to_device={},
    )
    before = worker.if_filts[1]

    worker.set_channel_if(1, 100e3)

    assert worker.channel_ifs[1] == 100e3
    assert worker.if_filts[1] is not before
    # The two channels are on one frequency now, so their filters match.
    assert (worker.if_filts[1] == worker.if_filts[0]).all()
