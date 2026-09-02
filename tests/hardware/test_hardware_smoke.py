"""End-to-end smoke test against physically attached devices.

For every backend that reports devices: discover, initialize, stream briefly,
stop, disconnect -- asserting the process survives each step and leaves nothing
behind. The point is not signal quality (that is the dummy-backend suite's job)
but that the lifecycle is crash-free and terminates cleanly on real hardware.

Run with:  pytest tests/hardware --hardware
"""

import contextlib
import multiprocessing as mp
import time

import pytest
from bioview_common import Command, Response

from bioview_server.device import AVAILABLE_BACKENDS, get_device_handler


STREAM_SECONDS = 2.0


def _attached(backend_type):
    backend = AVAILABLE_BACKENDS[backend_type]
    found = backend.discover_devices()
    entries = list(found.values()) if isinstance(found, dict) else (found or [])
    return [e for e in entries if isinstance(e, dict)]


@pytest.fixture(params=sorted(AVAILABLE_BACKENDS))
def backend_type(request):
    return request.param


def test_discovery_completes_without_crashing(backend_type):
    """Discovery must return promptly and never raise, attached or not."""
    start = time.monotonic()
    devices = _attached(backend_type)
    elapsed = time.monotonic() - start

    assert elapsed < 30.0, f"{backend_type} discovery took {elapsed:.1f}s"
    for device in devices:
        assert device.get("name"), device
        assert device.get("device_type") == backend_type or "device_type" in device


def test_discovery_is_repeatable(backend_type):
    """Back-to-back discovery must be stable; caches must not lose devices."""
    first = {d.get("serial") or d.get("name") for d in _attached(backend_type)}
    second = {d.get("serial") or d.get("name") for d in _attached(backend_type)}
    assert first == second, f"{backend_type} discovery is not repeatable"


def _minimal_group_config(backend_type, device):
    """Smallest config that will bring one device up for a short stream."""
    from bioview_common import DeviceType

    if backend_type == DeviceType.USRP.value:
        return {
            "device_type": backend_type,
            "samp_rate": 1_000_000,
            "carrier_freq": 1e9,
            "signal_scheme": "cw",
            "hardware": {
                device["name"]: {
                    "serial": device.get("serial"),
                    "tx_channels": [0],
                    "rx_channels": [0],
                    "if_freq": [100e3],
                    "tx_gain": [40],
                    "rx_gain": [40],
                    "tx_amplitude": [0.3],
                    "tx_phase": [0.0],
                }
            },
        }
    return {"device_type": backend_type, "samp_rate": 1000}


def test_full_lifecycle_on_attached_devices(backend_type):
    """Discover -> init -> stream -> stop -> disconnect, per attached device."""
    from bioview_common import Configuration

    devices = _attached(backend_type)
    if not devices:
        pytest.skip(f"no {backend_type} devices attached")

    device = devices[0]
    group_id = device["name"]
    group_cfg = _minimal_group_config(backend_type, device)

    config = Configuration.from_dict({group_id: group_cfg})
    device_cfg = config.devices[group_id]

    response_queue = mp.Queue()
    data_queue = mp.Queue(maxsize=32)
    handler = get_device_handler(
        group_id,
        device_cfg,
        response_queue,
        data_queue,
        discovered_devices={device["name"]: device},
    )
    assert handler is not None, f"no handler built for {backend_type}"

    try:
        handler.start()
        result = handler.initialize()
        assert result is not None, "device did not answer initialize()"

        handler.start_streaming(
            {"save_config": {"enable_save": False}, "display_config": {}}
        )

        chunks = 0
        deadline = time.monotonic() + STREAM_SECONDS
        while time.monotonic() < deadline:
            try:
                data_queue.get(timeout=0.5)
                chunks += 1
            except Exception:
                pass
        assert chunks > 0, f"{backend_type} produced no data in {STREAM_SECONDS}s"

        handler.stop_streaming()
        handler.disconnect()
    finally:
        with contextlib.suppress(Exception):
            handler.shutdown()
        handler.join(timeout=10)
        assert (
            not handler.is_alive()
        ), f"{backend_type} backend process did not exit; it would be orphaned"


def test_server_lists_attached_hardware(client):
    """The Configurator's listing must show real devices, not just the dummy."""
    resp_type, payload = client.command(Command.LIST_DEVICES)
    assert resp_type == Response.DEVICE_LIST.name, payload

    real = [d for d in payload["devices"] if d.get("device_type") != "dummy"]
    if not real:
        pytest.skip("no non-virtual devices attached")

    for device in real:
        assert device.get("serial"), device
        assert device.get("name"), device
