"""Configurator support: device enumeration and editable properties."""

import pytest
from bioview_common import Command, Response

from bioview_server.device import AVAILABLE_BACKENDS, usrp


def test_configurator_protocol_commands_exist():
    assert Command.LIST_DEVICES.name in [c.name for c in Command]
    assert Command.SET_DEVICE_CONFIG.name in [c.name for c in Command]
    assert Response.DEVICE_LIST.name in [r.name for r in Response]
    assert Response.DEVICE_CONFIG_UPDATED.name in [r.name for r in Response]


def test_every_backend_declares_an_editable_schema():
    """The Configurator gates its Edit button on this, so it must always exist."""
    assert AVAILABLE_BACKENDS, "no backends loaded"
    for backend_type, backend in AVAILABLE_BACKENDS.items():
        schema = getattr(backend, "EDITABLE_PROPERTIES", None)
        assert isinstance(schema, dict), backend_type
        assert callable(getattr(backend, "set_device_config", None)), backend_type
        for prop, spec in schema.items():
            assert "type" in spec, (backend_type, prop)
            assert "display_name" in spec, (backend_type, prop)


def test_backend_without_editable_properties_refuses_edits():
    dummy = AVAILABLE_BACKENDS.get("dummy")
    assert dummy is not None
    assert {} == dummy.EDITABLE_PROPERTIES
    ok, message = dummy.set_device_config({"name": "DummyVirtual"}, {"x": 1})
    assert not ok
    assert message


@pytest.fixture
def alias_cache(tmp_path, monkeypatch):
    """Redirect the name/serial caches into a temp dir.

    Patching ``naming`` rather than ``utils`` is the point of the split: device
    naming is filesystem work and must be usable, and testable, with no UHD
    installed.
    """
    import bioview_server.device.usrp.naming as naming

    def fake_cache_file(name):
        return str(tmp_path / f"{name}.json")

    monkeypatch.setattr(naming, "get_cache_file", fake_cache_file)
    return tmp_path


def test_usrp_declares_device_name_as_editable():
    assert "device_name" in usrp.EDITABLE_PROPERTIES
    spec = usrp.EDITABLE_PROPERTIES["device_name"]
    assert spec["type"] == "text"
    assert spec["required"] is True


def test_usrp_rename_stores_an_alias_keyed_on_serial(alias_cache):
    from bioview_server.device.usrp.naming import get_device_aliases

    ok, message = usrp.set_device_config(
        {"name": "MyB210", "serial": "31ABCDE", "device_type": "usrp"},
        {"device_name": "LeftArmRadio"},
    )
    assert ok, message
    assert get_device_aliases() == {"31ABCDE": "LeftArmRadio"}


def test_usrp_rename_also_updates_the_serial_lookup(alias_cache):
    from bioview_server.device.usrp.naming import get_usrp_address

    usrp.set_device_config(
        {"name": "MyB210", "serial": "31ABCDE", "device_type": "usrp"},
        {"device_name": "LeftArmRadio"},
    )
    # resolve_device_serial consults this, so the new name must resolve.
    assert get_usrp_address("LeftArmRadio") == "31ABCDE"


def test_usrp_rename_rejects_bad_names(alias_cache):
    device = {"name": "MyB210", "serial": "31ABCDE", "device_type": "usrp"}

    ok, msg = usrp.set_device_config(device, {"device_name": "   "})
    assert not ok and "empty" in msg.lower()

    ok, msg = usrp.set_device_config(device, {"device_name": "x" * 65})
    assert not ok and "64" in msg

    ok, _msg = usrp.set_device_config(device, {"device_name": "Radio"})
    assert ok
    other = {"name": "Other", "serial": "31FFFFF", "device_type": "usrp"}
    ok, msg = usrp.set_device_config(other, {"device_name": "Radio"})
    assert not ok and "already uses" in msg


def test_usrp_rename_requires_a_serial(alias_cache):
    ok, msg = usrp.set_device_config(
        {"name": "MyB210", "device_type": "usrp"}, {"device_name": "Radio"}
    )
    assert not ok and "serial" in msg.lower()


def test_unknown_device_name_resolves_to_none(alias_cache):
    """get_usrp_address used to raise KeyError for an unseen device."""
    from bioview_server.device.usrp.naming import get_usrp_address

    usrp.set_device_config(
        {"name": "MyB210", "serial": "31ABCDE", "device_type": "usrp"},
        {"device_name": "Known"},
    )
    assert get_usrp_address("NeverSeen") is None
