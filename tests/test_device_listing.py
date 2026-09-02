"""What the Configurator is shown when it enumerates attached hardware.

Two things went wrong here in the field: a BIOPAC unit that was physically
connected never appeared (the backend had silently failed to load, and nothing
said so), and simulated devices were listed alongside the real ones.
"""

from bioview_common import DEVICE_OP_COMMAND_TIMEOUT, Command, Response

import bioview_server.server as server_mod


def _list(client, **params):
    resp_type, payload = client.command(
        Command.LIST_DEVICES, params or None, timeout=DEVICE_OP_COMMAND_TIMEOUT
    )
    assert resp_type == Response.DEVICE_LIST.name, payload
    return payload


def test_virtual_devices_are_hidden_from_the_configurator(client):
    payload = _list(client)
    assert not [d for d in payload["devices"] if d.get("device_type") == "dummy"], (
        "the Configurator lists attached hardware; a simulated device among the "
        "real ones is misleading"
    )
    assert "dummy" not in payload["backends"]


def test_virtual_devices_can_still_be_asked_for(client):
    payload = _list(client, include_virtual=True)
    assert any(d.get("device_type") == "dummy" for d in payload["devices"])


def test_a_backend_that_failed_to_load_is_reported_with_its_reason(client, monkeypatch):
    """A missing driver or Python dependency used to be invisible.

    The reason was printed at import time, and a server spawned by the GUI has
    its stdout detached, so the hardware just never turned up with no
    explanation anywhere the user could see.
    """
    # A name no real backend uses, so the assertion holds whatever hardware
    # support happens to be installed on the machine running the tests.
    monkeypatch.setattr(
        server_mod, "UNAVAILABLE_BACKENDS", {"absent": "No module named 'wmi'"}
    )

    payload = _list(client)
    absent = payload["backends"]["absent"]
    assert absent["available"] is False
    assert "wmi" in absent["error"]


def test_an_unavailable_backend_does_not_stop_the_others_being_listed(
    client, monkeypatch
):
    monkeypatch.setattr(server_mod, "UNAVAILABLE_BACKENDS", {"absent": "no driver"})

    payload = _list(client, include_virtual=True)
    assert any(d.get("device_type") == "dummy" for d in payload["devices"])
    assert payload["backends"]["absent"]["available"] is False
    # Everything that did load is still listed alongside it.
    assert any(info.get("available") for info in payload["backends"].values())


def test_an_unavailable_virtual_backend_is_not_reported_to_the_configurator(
    client, monkeypatch
):
    monkeypatch.setattr(server_mod, "UNAVAILABLE_BACKENDS", {"dummy": "boom"})
    assert "dummy" not in _list(client)["backends"]
