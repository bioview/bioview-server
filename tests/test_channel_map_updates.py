"""A channel-map edit must reach the backend that acts on it.

DPIC pairs are *specified* in the channel map, and the channel map is edited in
the settings panel -- so `UPDATE_RUNNING_PARAMETER` with `channel_map` is the
path a pair actually arrives by. It used to fall through `_queue_param_update`
untouched: the value never reached `group_config`, `populate_data_sources()`
was never re-run, and `dpic_pairs` kept whatever the config file had at connect
time. Adding a loop in the UI and pressing Balance reported "No DPIC pairs are
configured for this device group".
"""

import copy
import multiprocessing as mp
import queue

import pytest

from bioview_server.device.dummy.backend import DummyBackend
from bioview_server.device.usrp.backend import USRPBackend


BASE_CONFIG = {
    "hardware": {
        "A": {
            "tx_channels": [0, 1],
            "rx_channels": [0, 1],
            "if_freq": [100e3, 110e3],
            "if_filter_bw": 5000,
        }
    },
    "channel_map": {"layout": "full_nxn", "dpic": []},
}

WITH_PAIR = {
    "layout": "full_nxn",
    "dpic": [{"inject_tx": 1, "measure_tx": 0, "measure_rx": 0}],
}


def _dummy():
    return DummyBackend(
        group_id="grp",
        response_queue=mp.Queue(),
        data_output_queue=mp.Queue(),
        group_config=copy.deepcopy(BASE_CONFIG),
    )


def _usrp():
    return USRPBackend(
        group_id="grp",
        samp_rate=1e6,
        devices={},
        group_config=copy.deepcopy(BASE_CONFIG),
        response_queue=mp.Queue(),
        data_output_queue=mp.Queue(),
    )


@pytest.mark.parametrize("make_backend", [_dummy, _usrp], ids=["dummy", "usrp"])
def test_a_pair_added_in_the_ui_reaches_dpic_pairs(make_backend):
    be = make_backend()
    assert be.dpic_pairs == []

    be._queue_param_update({"channel_map": copy.deepcopy(WITH_PAIR)})

    assert [(p.inject_tx, p.measure_tx, p.target_rx) for p in be.dpic_pairs] == [
        (1, 0, 0)
    ]
    assert be.group_config["channel_map"]["dpic"]


@pytest.mark.parametrize("make_backend", [_dummy, _usrp], ids=["dummy", "usrp"])
def test_the_measurement_grid_follows_the_new_map(make_backend):
    be = make_backend()
    assert sorted(s.label for s in be.mimo_sources) == [
        "Tx1Rx1",
        "Tx1Rx2",
        "Tx2Rx1",
        "Tx2Rx2",
    ]

    be._queue_param_update({"channel_map": copy.deepcopy(WITH_PAIR)})

    # Tx2 now radiates the cancellation tone and its Rx has nothing to receive,
    # so both halves of that channel are retired from the grid.
    assert sorted(s.label for s in be.mimo_sources) == ["Tx1Rx1"]


@pytest.mark.parametrize("make_backend", [_dummy, _usrp], ids=["dummy", "usrp"])
def test_the_parent_side_mirror_sees_it_too(make_backend):
    """`get_data_sources()` and the balance dispatch are answered by the parent."""
    be = make_backend()
    be._apply_param_update_local({"channel_map": copy.deepcopy(WITH_PAIR)})
    assert len(be.dpic_pairs) == 1
    # The cal-ref row rides along; the measurement grid is what the map changed.
    assert sorted(s.label for s in be.get_data_sources()) == ["CalRef_Tx1", "Tx1Rx1"]


def test_transmit_schemes_survive_the_rebuild():
    """The live transmit workers hold these objects; replacing them detaches them."""
    be = _usrp()
    before = dict(be.schemes_by_device)

    be._queue_param_update({"channel_map": copy.deepcopy(WITH_PAIR)})

    assert be.schemes_by_device == before
    for name, scheme in before.items():
        assert be.schemes_by_device[name] is scheme


def test_a_map_change_is_refused_while_streaming():
    """It changes the emitted row count, which would desync a live recording."""
    be = _usrp()
    be._streaming.set()

    assert be._reload_channel_map(copy.deepcopy(WITH_PAIR)) is False
    assert be.dpic_pairs == []


def test_the_processing_worker_adopts_the_new_rows():
    from bioview_server.device.usrp.process import ProcessWorker

    be = _usrp()
    be.process_worker = ProcessWorker(
        data_sources=be.mimo_sources,
        cal_ref_sources=be.cal_ref_sources,
        samp_rate=1e6,
        channel_ifs=be.channel_ifs,
        if_filter_bw=be.if_filter_bw,
        rx_queues={},
        rx_device_order=["A"],
        schemes_by_device=be.schemes_by_device,
        global_tx_to_device=be.global_tx_to_device,
    )
    be.process_worker.latest_metrics[(0, 0)] = (1.0, 5)

    be._queue_param_update({"channel_map": copy.deepcopy(WITH_PAIR)})

    assert sorted(s.label for s in be.process_worker.mimo_sources) == ["Tx1Rx1"]
    # Metrics are keyed by (tx, rx) meanings that just changed.
    assert be.process_worker.latest_metrics == {}


def test_unrelated_parameters_still_reach_the_worker_queues():
    """The channel-map branch must not swallow everything else in the same call."""
    be = _usrp()
    be.tx_command_queue = {"A": queue.Queue()}
    be.rx_command_queue = {"A": queue.Queue()}

    be._queue_param_update(
        {"channel_map": copy.deepcopy(WITH_PAIR), "calibration.enabled": True}
    )

    forwarded = []
    while not be.tx_command_queue["A"].empty():
        forwarded.append(be.tx_command_queue["A"].get_nowait())
    assert {"param": "calibration.enabled", "value": True} in forwarded
    # ...and the map itself is not forwarded: nothing down there reads it.
    assert all(item["param"] != "channel_map" for item in forwarded)


def test_a_map_edit_republishes_the_servers_data_sources(monkeypatch):
    """The plot-source selector follows the map without a reconnect.

    The reply to UPDATE_RUNNING_PARAMETER carries the new source list, which is
    what drives `data_sources_changed` -> `populate_plot_grid_sources` on the
    client. That only works if the *parent* handler rebuilt its sources, since
    `get_data_sources()` is answered out of the parent process.
    """
    from bioview_common import Response

    from bioview_server.server import Server

    handler = _usrp()
    srv = Server(local_only=True, control_port=0, data_port=0)
    srv.device_group_handlers = {"grp": handler}
    srv.config = None

    sent = {}

    def fake_send(sock, response, params=None, logger=None):
        sent["response"] = response
        sent["params"] = params or {}

    monkeypatch.setattr("bioview_server.server.send_response", fake_send)
    monkeypatch.setattr(Server, "client_control_conn", property(lambda self: None))

    srv._update_running_parameter(
        {"id": "grp", "config": {"channel_map": copy.deepcopy(WITH_PAIR)}}
    )

    assert sent["response"] is Response.SUCCESS
    labels = sorted(src["label"] for src in sent["params"]["data_sources"])
    assert labels == ["CalRef_Tx1", "Tx1Rx1"]
