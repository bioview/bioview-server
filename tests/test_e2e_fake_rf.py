"""E2E tests for the fake backend in RF-simulation mode (MIMO / DPIC config)."""

import json
from pathlib import Path

from bioview_common import Command, Response
from fakes import FakeConfiguration


# Kept beside the tests: the server is checked out on its own in CI,
# so a path above the repo root does not exist there.
DATA_DIR = Path(__file__).resolve().parent / "data"
RF_CFG_PATH = DATA_DIR / "fake_dpic_2x2_mimo_cfg.json"


def _rf_device_groups():
    data = json.loads(RF_CFG_PATH.read_text(encoding="utf-8"))
    return {
        key: FakeConfiguration.from_dict(value).to_dict()
        for key, value in data.items()
        if key.lower() != "experiment"
    }


RF_DEVICE_GROUPS = _rf_device_groups()


def test_discover_rf_fake_device(client):
    resp_type, payload = client.device_command(
        Command.DISCOVER_DEVICES, {"device_groups": RF_DEVICE_GROUPS}
    )
    assert resp_type == Response.SUCCESS.name, payload
    assert payload["device_status"]["Fake_DPIC_2x2"] == "Available"


def test_initialize_rf_fake_device(client):
    resp_type, payload = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": RF_DEVICE_GROUPS}
    )
    assert resp_type in (Response.SUCCESS.name, Response.WARNING.name), payload
    assert payload["device_status"]["Fake_DPIC_2x2"] == "Connected"
    sources = payload.get("data_sources", [])
    labels = {s["label"] for s in sources}
    assert "Tx1Rx1" in labels
    assert len(sources) >= 4
