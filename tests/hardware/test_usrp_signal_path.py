"""Signal-path check against an attached USRP.

The lifecycle smoke test proves the process survives Start/Stop. This proves the
radio is actually doing something: that the transmitted IF tone comes back on
the receive chain, and that switching the calibration overlay on at runtime
changes the received magnitude the way it should.

Run with:  pytest tests/hardware --hardware
"""

import contextlib
import multiprocessing as mp
import queue
import time

import numpy as np
import pytest
from bioview_common import Configuration, DeviceType

from bioview_server.device import AVAILABLE_BACKENDS, get_device_handler


SAMP_RATE = 1_000_000
IF_FREQ = 100e3
MODULATION_DEPTH = 0.2
COLLECT_SECONDS = 3.0


def _attached_usrps():
    backend = AVAILABLE_BACKENDS.get(DeviceType.USRP.value)
    if backend is None:
        return []
    found = backend.discover_devices() or {}
    return [d for d in found.values() if isinstance(d, dict)]


def _group_config(device):
    return {
        "device_type": DeviceType.USRP.value,
        "samp_rate": SAMP_RATE,
        "carrier_freq": 1e9,
        "signal_scheme": "cw",
        "save_ds": 100,
        "hardware": {
            device["name"]: {
                "serial": device.get("serial"),
                "tx_channels": [0],
                "rx_channels": [0],
                "if_freq": [IF_FREQ],
                "tx_gain": [40],
                "rx_gain": [40],
                "tx_amplitude": [0.5],
                "tx_phase": [0.0],
                "if_filter_bw": 5000,
            }
        },
        "channel_map": {"layout": "full_nxn", "dpic": []},
        "calibration": {
            "enabled": False,
            "shape": "triangle",
            "num_pulses": 5,
            "packet_spacing_s": 0.5,
            "envelope_freq_hz": 10.0,
            "modulation_depth": MODULATION_DEPTH,
            "inject_channels": [0],
            "record_reference": True,
        },
    }


def _collect(data_queue, seconds):
    """Drain display chunks for ``seconds``; returns (stacked_rows, sources)."""
    chunks = []
    sources = None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            payload = data_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if isinstance(payload, dict):
            chunks.append(np.atleast_2d(payload["data"]))
            sources = payload.get("sources") or sources
        else:
            chunks.append(np.atleast_2d(payload))
    return (np.hstack(chunks) if chunks else np.empty((0, 0))), sources


@pytest.fixture
def streaming_usrp():
    devices = _attached_usrps()
    if not devices:
        pytest.skip("no USRP devices attached")

    device = devices[0]
    group_id = device["name"]
    config = Configuration.from_dict({group_id: _group_config(device)})

    response_queue = mp.Queue()
    data_queue = mp.Queue(maxsize=64)
    handler = get_device_handler(
        group_id,
        config.devices[group_id],
        response_queue,
        data_queue,
        discovered_devices={group_id: device},
    )
    assert handler is not None

    handler.start()
    assert handler.initialize() is not None, "USRP did not answer initialize()"
    handler.start_streaming(
        {"save_config": {"enable_save": False}, "display_config": {}}
    )

    yield handler, data_queue

    with contextlib.suppress(Exception):
        handler.stop_streaming()
    with contextlib.suppress(Exception):
        handler.disconnect()
    with contextlib.suppress(Exception):
        handler.shutdown()
    handler.join(timeout=10)


def _row_for(sources, label):
    assert sources, "server sent no per-chunk source list"
    for idx, src in enumerate(sources):
        if src.get("label") == label:
            return idx
    pytest.fail(f"no row labelled {label} in {[s.get('label') for s in sources]}")


def test_transmitted_tone_is_received(streaming_usrp):
    """Tx0's IF tone must appear on Rx0 well above the noise floor.

    A near-zero magnitude here means the Tx/Rx pair is not actually coupled:
    check antennas/cabling, the Tx and Rx gains, and that both are tuned to the
    same carrier before looking any further up the stack.
    """
    _handler, data_queue = streaming_usrp
    data, sources = _collect(data_queue, COLLECT_SECONDS)

    assert data.size, "no data chunks arrived from the USRP backend"
    row = _row_for(sources, "Tx1Rx1")
    level = float(np.mean(np.abs(data[row])))
    assert level > 1e-3, f"Tx1Rx1 magnitude {level:.2e} -- no tone reaching the Rx"


def test_calibration_overlay_modulates_the_received_magnitude(streaming_usrp):
    """Turning calibration on at runtime must show up in the received data."""
    handler, data_queue = streaming_usrp

    baseline, sources = _collect(data_queue, COLLECT_SECONDS)
    assert baseline.size, "no baseline data"
    row = _row_for(sources, "Tx1Rx1")
    quiet_ptp = float(np.ptp(baseline[row]))

    handler.queue_param_update(**{"calibration.enabled": True})
    time.sleep(1.0)
    _drained, _ = _collect(data_queue, 1.0)  # discard the transition

    modulated, sources = _collect(data_queue, COLLECT_SECONDS)
    assert modulated.size, "no data after enabling calibration"
    row = _row_for(sources, "Tx1Rx1")
    cal_ptp = float(np.ptp(modulated[row]))

    assert cal_ptp > quiet_ptp * 2, (
        f"peak-to-peak magnitude barely moved when calibration was enabled "
        f"({quiet_ptp:.4f} -> {cal_ptp:.4f}); the burst is not reaching the Rx"
    )


def test_calibration_reference_row_is_streamed(streaming_usrp):
    """The CalRef_* source must carry the gated envelope, not stay flat."""
    handler, data_queue = streaming_usrp
    handler.queue_param_update(**{"calibration.enabled": True})
    time.sleep(1.0)

    data, sources = _collect(data_queue, COLLECT_SECONDS)
    assert data.size, "no data chunks arrived"
    row = _row_for(sources, "CalRef_Tx1")
    assert float(np.ptp(data[row])) > 0.1, "calibration reference row is flat"
