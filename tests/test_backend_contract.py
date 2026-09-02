"""Backend contract, and graceful behaviour when hardware is absent.

This is the half of the hardware story that runs everywhere, including CI with
no radios plugged in. It asserts that a missing driver or a missing device is
reported, not crashed on: every backend module answers the same questions, and
the server can enumerate and stream with whatever happens to be present.
"""

import types

import pytest
from bioview_common import DEVICE_OP_COMMAND_TIMEOUT, Command, Response

from bioview_server.device import AVAILABLE_BACKENDS


#: Every entry in AVAILABLE_BACKENDS must provide these.
REQUIRED_BACKEND_ATTRS = (
    "discover_devices",
    "set_device_config",
)


def test_dummy_backend_is_always_available():
    """Hardware-free streaming must work on any machine."""
    assert "dummy" in AVAILABLE_BACKENDS


@pytest.mark.parametrize("backend_type", sorted(AVAILABLE_BACKENDS))
def test_backend_exposes_the_full_contract(backend_type):
    backend = AVAILABLE_BACKENDS[backend_type]
    assert isinstance(backend, types.ModuleType)
    for attr in REQUIRED_BACKEND_ATTRS:
        assert callable(getattr(backend, attr, None)), f"{backend_type}.{attr}"
    assert isinstance(getattr(backend, "EDITABLE_PROPERTIES", None), dict)


@pytest.mark.parametrize("backend_type", sorted(AVAILABLE_BACKENDS))
def test_discovery_returns_a_sane_shape_with_or_without_hardware(backend_type):
    """Discovery must return a container, never raise, when nothing is attached."""
    backend = AVAILABLE_BACKENDS[backend_type]
    found = backend.discover_devices()
    assert isinstance(found, (dict, list)), type(found)

    entries = list(found.values()) if isinstance(found, dict) else found
    for entry in entries:
        assert isinstance(entry, dict), entry
        assert entry.get("name"), entry


def test_registered_backends_have_their_driver_present():
    """A backend is only listed if it can actually be used.

    ``usrp/__init__`` resolves its heavy attributes lazily, so importing it
    succeeds with no UHD installed. Registration must therefore prove the driver
    is really there, or the server advertises a backend that fails on first use.
    """
    for backend_type, backend in AVAILABLE_BACKENDS.items():
        # Touching the lazy attribute is what would raise on a missing driver.
        assert callable(backend.discover_devices), backend_type


def test_server_lists_devices_without_a_loaded_configuration(client):
    """LIST_DEVICES is what the Configurator calls before any config exists."""
    resp_type, payload = client.command(
        Command.LIST_DEVICES,
        {"include_virtual": True},
        timeout=DEVICE_OP_COMMAND_TIMEOUT,
    )
    assert resp_type == Response.DEVICE_LIST.name, payload

    devices = payload.get("devices")
    backends = payload.get("backends")
    assert isinstance(devices, list)
    assert isinstance(backends, dict)

    # The virtual device is always available, so with virtual devices asked for
    # the listing is never empty.
    assert any(d.get("device_type") == "dummy" for d in devices), devices

    for backend_type, info in backends.items():
        assert "editable_properties" in info, backend_type
        assert "available" in info, backend_type


def test_listed_devices_carry_their_editability(client):
    _resp, payload = client.command(
        Command.LIST_DEVICES,
        {"include_virtual": True},
        timeout=DEVICE_OP_COMMAND_TIMEOUT,
    )
    backends = payload["backends"]
    for device in payload["devices"]:
        schema = backends[device["device_type"]]["editable_properties"]
        assert device["editable"] == bool(schema), device


def test_set_device_config_rejects_an_unknown_backend(client):
    resp_type, payload = client.command(
        Command.SET_DEVICE_CONFIG,
        {"device_info": {"device_type": "not_a_backend"}, "config": {}},
    )
    assert resp_type == Response.ERROR.name
    assert "not_a_backend" in payload.get("message", "")


def test_set_device_config_rejects_a_backend_without_editable_properties(client):
    resp_type, payload = client.command(
        Command.SET_DEVICE_CONFIG,
        {
            "device_info": {"device_type": "dummy", "name": "DummyVirtual"},
            "config": {"device_name": "whatever"},
        },
    )
    assert resp_type == Response.ERROR.name
    assert payload.get("message")


def test_one_failing_backend_does_not_hide_the_others(client, monkeypatch):
    """A backend that raises during discovery is reported, not fatal."""
    import bioview_server.server as server_mod

    class Exploding:
        EDITABLE_PROPERTIES = {}

        @staticmethod
        def discover_devices():
            raise RuntimeError("driver on fire")

    patched = dict(server_mod.AVAILABLE_BACKENDS)
    patched["exploding"] = Exploding
    monkeypatch.setattr(server_mod, "AVAILABLE_BACKENDS", patched)

    resp_type, payload = client.command(
        Command.LIST_DEVICES,
        {"include_virtual": True},
        timeout=DEVICE_OP_COMMAND_TIMEOUT,
    )
    assert resp_type == Response.DEVICE_LIST.name, payload

    # The dummy backend still reports its device...
    assert any(d.get("device_type") == "dummy" for d in payload["devices"])
    # ...and the broken one is flagged rather than swallowed.
    assert payload["backends"]["exploding"]["available"] is False
    assert "driver on fire" in payload["backends"]["exploding"]["error"]
