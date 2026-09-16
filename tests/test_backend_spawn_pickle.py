"""A backend must survive the pickle that ``spawn`` puts it through."""

import copy
import multiprocessing as mp

import pytest
from fakes.backend import FakeBackend

from bioview_server.device.usrp.backend import USRPBackend


USRP_CONFIG = {
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


def _fake():
    return FakeBackend(
        group_id="grp",
        response_queue=mp.Queue(),
        data_output_queue=mp.Queue(),
        group_config={},
    )


def _usrp():
    return USRPBackend(
        group_id="grp",
        samp_rate=1e6,
        devices={},
        group_config=copy.deepcopy(USRP_CONFIG),
        response_queue=mp.Queue(),
        data_output_queue=mp.Queue(),
    )


MAKERS = [_fake, _usrp]
IDS = ["fake", "usrp"]


@pytest.mark.parametrize("make_backend", MAKERS, ids=IDS)
def test_no_thread_primitive_is_shipped_to_the_child(make_backend):
    import threading

    backend = make_backend()
    state = backend.__getstate__()

    unpicklable = (
        type(threading.Lock()),
        type(threading.RLock()),
        threading.Event,
        threading.Condition,
        threading.Thread,
    )
    offenders = [k for k, v in state.items() if isinstance(v, unpicklable)]
    assert not offenders, (
        f"{type(backend).__name__} would ship {offenders} to the child; "
        "create them in _init_local_state and list them in _LOCAL_STATE_KEYS"
    )


@pytest.mark.parametrize("make_backend", MAKERS, ids=IDS)
def test_local_state_is_rebuilt_in_the_child(make_backend):
    backend = make_backend()
    keys = type(backend)._local_state_keys()
    assert "_reply_lock" in keys

    revived = type(backend).__new__(type(backend))
    revived.__setstate__(backend.__getstate__())

    for key in keys:
        assert hasattr(revived, key), key


def test_a_subclass_tuple_extends_the_base_list_rather_than_hiding_it():
    assert "_gain_lock" in USRPBackend._local_state_keys()
