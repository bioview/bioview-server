"""End-to-end server tests driven by the fake backend in tests/fakes."""

from bioview_common import Command, Response
from fakes import FakeConfiguration


NUM_CHANNELS = 4

FAKE_DEVICE_GROUPS = {
    "FakeDevice": FakeConfiguration.from_dict(
        {
            "type": "FAKE",
            "samp_rate": 500,
            "num_channels": NUM_CHANNELS,
            "signal_freq": 1.0,
            "amplitude": 1.0,
            "noise_std": 0.0,
            "chunk_duration": 0.05,
        }
    ).to_dict()
}


def test_connection_and_auth(client):
    assert client.control_sock is not None
    assert client.data_sock is not None


def test_discover_fake_device(client):
    resp_type, payload = client.device_command(
        Command.DISCOVER_DEVICES, {"device_groups": FAKE_DEVICE_GROUPS}
    )
    assert resp_type == Response.SUCCESS.name, payload
    assert "device_status" in payload
    assert "FakeDevice" in payload["device_status"]
    assert payload["device_status"]["FakeDevice"] == "Available"


def test_initialize_fake_device(client):
    resp_type, payload = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": FAKE_DEVICE_GROUPS}
    )
    assert resp_type == Response.SUCCESS.name, payload
    assert payload["device_status"]["FakeDevice"]
    sources = payload.get("data_sources", [])
    assert len(sources) == NUM_CHANNELS


def test_stream_fake_device_end_to_end(client):
    resp_type, _ = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": FAKE_DEVICE_GROUPS}
    )
    assert resp_type == Response.SUCCESS.name

    resp_type, payload = client.command(
        Command.START_STREAMING,
        {"Experiment": {"type": "EXPERIMENT"}, **FAKE_DEVICE_GROUPS},
    )
    assert resp_type == Response.SUCCESS.name, payload

    data, sources = client.recv_data_chunk(timeout=5.0)
    assert data.shape[0] == NUM_CHANNELS
    assert data.shape[1] > 0

    data2, _ = client.recv_data_chunk(timeout=5.0)
    assert data2.shape[0] == NUM_CHANNELS

    resp_type, _ = client.command(Command.STOP_STREAMING)
    assert resp_type == Response.SUCCESS.name


def test_streaming_requires_initialized_devices(client):
    resp_type, payload = client.command(
        Command.START_STREAMING, {"Experiment": {"type": "EXPERIMENT"}}
    )
    assert resp_type == Response.ERROR.name, payload
