"""E2E tests for RF-simulation dummy device (MIMO / DPIC config)."""

import json
from pathlib import Path

from bioview_common import Command, DummyConfiguration, Response

REPO_ROOT = Path(__file__).resolve().parents[2]
RF_CFG_PATH = REPO_ROOT / "dummy_dpic_2x2_mimo_cfg.json"


def _rf_device_groups():
    data = json.loads(RF_CFG_PATH.read_text(encoding="utf-8"))
    return {
        key: DummyConfiguration.from_dict(value).to_dict()
        for key, value in data.items()
        if key.lower() != "experiment"
    }


RF_DEVICE_GROUPS = _rf_device_groups()


def test_discover_rf_dummy_device(client):
    resp_type, payload = client.device_command(
        Command.DISCOVER_DEVICES, {"device_groups": RF_DEVICE_GROUPS}
    )
    assert resp_type == Response.SUCCESS.name, payload
    assert payload["device_status"]["Dummy_DPIC_2x2"] == "Available"


def test_initialize_rf_dummy_device(client):
    resp_type, payload = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": RF_DEVICE_GROUPS}
    )
    assert resp_type in (Response.SUCCESS.name, Response.WARNING.name), payload
    assert payload["device_status"]["Dummy_DPIC_2x2"] == "Connected"
    sources = payload.get("data_sources", [])
    labels = {s["label"] for s in sources}
    assert "Tx1Rx1" in labels
    assert len(sources) >= 4
