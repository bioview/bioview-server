"""Backend contract, and graceful behaviour when hardware is absent."""

import types

import pytest
from bioview_common import DEVICE_OP_COMMAND_TIMEOUT, Command, Response

from bioview_server.device import AVAILABLE_BACKENDS


REQUIRED_BACKEND_ATTRS = (
    "discover_devices",
    "set_device_config",
)


def test_fake_backend_is_registered():
    """Hardware-free streaming must work on any machine."""
    assert "fake" in AVAILABLE_BACKENDS


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
    assert isinstance(found, dict | list), type(found)

    entries = list(found.values()) if isinstance(found, dict) else found
    for entry in entries:
        assert isinstance(entry, dict), entry
        assert entry.get("name"), entry


def test_registered_backends_have_their_driver_present():
    """A backend is only listed if it can actually be used."""
    for backend_type, backend in AVAILABLE_BACKENDS.items():
        assert callable(backend.discover_devices), backend_type


def test_server_lists_devices_without_a_loaded_configuration(client):
    """LIST_DEVICES is what the Configurator calls before any config exists."""
    resp_type, payload = client.command(
        Command.LIST_DEVICES,
        timeout=DEVICE_OP_COMMAND_TIMEOUT,
    )
    assert resp_type == Response.DEVICE_LIST.name, payload

    devices = payload.get("devices")
    backends = payload.get("backends")
    assert isinstance(devices, list)
    assert isinstance(backends, dict)

    assert any(d.get("device_type") == "fake" for d in devices), devices

    for backend_type, info in backends.items():
        assert "editable_properties" in info, backend_type
        assert "available" in info, backend_type


def test_listed_devices_carry_their_editability(client):
    _resp, payload = client.command(
        Command.LIST_DEVICES,
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
            "device_info": {"device_type": "fake", "name": "FakeDevice"},
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
        timeout=DEVICE_OP_COMMAND_TIMEOUT,
    )
    assert resp_type == Response.DEVICE_LIST.name, payload

    assert any(d.get("device_type") == "fake" for d in payload["devices"])
    assert payload["backends"]["exploding"]["available"] is False
    assert "driver on fire" in payload["backends"]["exploding"]["error"]
