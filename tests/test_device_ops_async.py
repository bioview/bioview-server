"""Tests for async device discover/init and GET_DEVICE_STATUS polling."""

from bioview_common import Command, Response

from test_e2e_dummy_streaming import DUMMY_DEVICE_GROUPS


def test_initialize_returns_device_connecting_then_polls(client):
    resp_type, payload = client.command(
        Command.INITIALIZE_DEVICES, {"device_groups": DUMMY_DEVICE_GROUPS}
    )
    assert resp_type == Response.DEVICE_CONNECTING.name, payload
    assert payload.get("pending") is True
    assert payload["device_status"]["DummyDevice"] == "Connecting"

    final_type, final_payload = client.wait_for_device_op(resp_type, payload)
    assert final_type == Response.SUCCESS.name, final_payload
    assert final_payload["device_status"]["DummyDevice"] == "Connected"
    assert len(final_payload.get("data_sources", [])) > 0


def test_get_device_status_while_idle(client):
    resp_type, payload = client.command(Command.GET_DEVICE_STATUS)
    assert resp_type == Response.SUCCESS.name, payload
    assert payload.get("pending") is False
