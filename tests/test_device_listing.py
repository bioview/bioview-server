"""What the Configurator is shown when it enumerates attached hardware."""

from bioview_common import DEVICE_OP_COMMAND_TIMEOUT, Command, Response

import bioview_server.server as server_mod


def _list(client, **params):
    resp_type, payload = client.command(
        Command.LIST_DEVICES, params or None, timeout=DEVICE_OP_COMMAND_TIMEOUT
    )
    assert resp_type == Response.DEVICE_LIST.name, payload
    return payload


def test_every_registered_backend_is_listed(client):
    payload = _list(client)
    assert any(d.get("device_type") == "fake" for d in payload["devices"])
    assert payload["backends"]["fake"]["available"] is True


def test_a_backend_that_failed_to_load_is_reported_with_its_reason(client, monkeypatch):
    """A missing driver or Python dependency used to be invisible."""
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

    payload = _list(client)
    assert any(d.get("device_type") == "fake" for d in payload["devices"])
    assert payload["backends"]["absent"]["available"] is False
    assert any(info.get("available") for info in payload["backends"].values())
