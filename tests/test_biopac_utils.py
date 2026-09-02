import types

import pytest


# The BIOPAC backend is Windows-only and imports the `wmi` package at module
# load; skip the whole module cleanly on platforms where it is unavailable.
utils = pytest.importorskip(
    "bioview_server.device.biopac.utils",
    reason="BIOPAC backend (wmi) is only available on Windows",
)


class DummyDevice:
    def __init__(self, device_id, name, manufacturer, present=True):
        self.DeviceID = device_id
        self.Name = name
        self.Description = name
        self.Manufacturer = manufacturer
        self.Service = "mpdev"
        self.Status = "OK"
        self.Present = present


def test_discover_devices_filters_by_vid_and_name(monkeypatch):
    # Create dummy devices with different VID/Manufacturer
    d1 = DummyDevice("USB\\VID_097E&PID_0001", "BIOPAC Device", "BIOPAC Corp")
    d2 = DummyDevice("USB\\VID_1234&PID_5678", "Other Device", "Other Co")
    d3 = DummyDevice("USB\\VID_097E&PID_ABCD", "Unknown", "SomeVendor")

    class DummyWMI:
        def Win32_PnPEntity(self):
            return [d1, d2, d3]

    monkeypatch.setattr(utils, "wmi", types.SimpleNamespace(WMI=lambda: DummyWMI()))

    found = utils.discover_devices()
    assert isinstance(found, dict)
    ids = [d["device_id"] for d in found.values()]
    assert any("VID_097E" in i for i in ids)
    assert any(
        "BIOPAC" in d["manufacturer"].upper() or "BIOPAC" in d["name"].upper()
        for d in found.values()
    )


def test_discover_devices_initializes_pythoncom_when_available(monkeypatch):
    d1 = DummyDevice("USB\\VID_DEAD&PID_BEEF", "NoBioPac", "Nope")

    class DummyWMI:
        def Win32_PnPEntity(self):
            return [d1]

    monkeypatch.setattr(utils, "wmi", types.SimpleNamespace(WMI=lambda: DummyWMI()))

    class DummyPyCom:
        initialized = False

        @staticmethod
        def CoInitialize():
            DummyPyCom.initialized = True

        @staticmethod
        def CoUninitialize():
            DummyPyCom.initialized = False

    monkeypatch.setattr(utils, "_com_module", lambda: DummyPyCom)

    found = utils.discover_devices()
    assert isinstance(found, dict)
    assert not DummyPyCom.initialized, "COM should be uninitialized again afterwards"


def test_discover_devices_handles_missing_pythoncom(monkeypatch):
    d1 = DummyDevice("USB\\VID_097E&PID_0002", "BIOPAC Device", "BIOPAC")

    class DummyWMI:
        def Win32_PnPEntity(self):
            return [d1]

    monkeypatch.setattr(utils, "wmi", types.SimpleNamespace(WMI=lambda: DummyWMI()))
    # pywin32 absent: discovery must still run rather than fail outright
    monkeypatch.setattr(utils, "_com_module", lambda: None)

    found = utils.discover_devices()
    assert len(found) == 1
