from bioview_common import Response

from bioview_server import device
from bioview_server.server import Server


class DummyBackendModule:
    def __init__(self, devices):
        self._devices = devices

    def discover_devices(self):
        return self._devices


def test_discover_devices_full(monkeypatch):
    s = Server()

    # Mock AVAILABLE_BACKENDS with two backends
    dummy = {
        "biopac": DummyBackendModule([{"device_id": "C"}, {"device_id": "D"}]),
        "usrp": DummyBackendModule([{"device_id": "A"}, {"device_id": "B"}]),
    }
    monkeypatch.setattr(device, "AVAILABLE_BACKENDS", dummy)

    resp = s.handle_discover_devices()
    assert resp["type"] == Response.SUCCESS.value
    payload = resp["payload"]
    assert "devices" in payload
    assert "biopac" in payload["devices"]


def test_discover_devices_matching(monkeypatch):
    s = Server()

    dummy = {
        "biopac": DummyBackendModule([{"device_id": "C"}, {"device_id": "D"}]),
        "usrp": DummyBackendModule([{"device_id": "A"}, {"device_id": "B"}]),
    }
    monkeypatch.setattr(device, "AVAILABLE_BACKENDS", dummy)

    # Request groups: group1 wants devices C and E; group2 wants A
    requested = {
        "group1": {"C": {}, "E": {}},
        "group2": {"A": {}},
    }

    resp = s.handle_discover_devices(requested=requested)
    assert resp["type"] == Response.SUCCESS.value
    devices = resp["payload"]["devices"]

    assert devices["group1"]["C"] is True
    assert devices["group1"]["E"] is False
    assert devices["group2"]["A"] is True
