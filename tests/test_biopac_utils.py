import types
import pytest

from bioview_server.device.biopac import utils


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
    # Should find d1 and d3 because of BIOPAC vid or name
    ids = [d["device_id"] for d in found]
    assert any("VID_097E" in i for i in ids)
    assert any("BIOPAC" in d["manufacturer"].upper() or "BIOPAC" in d["name"].upper() for d in found)


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

    monkeypatch.setattr(utils, "importlib", __import__("importlib"))
    # Ensure importlib.import_module returns our dummy pythoncom
    monkeypatch.setattr(utils.importlib, "import_module", lambda name: DummyPyCom if name == "pythoncom" else __import__(name))

    found = utils.discover_devices()
    # Should run and uninitialize pythoncom without error
    assert isinstance(found, list)


def test_discover_devices_handles_missing_pythoncom(monkeypatch):
    d1 = DummyDevice("USB\\VID_097E&PID_0002", "BIOPAC Device", "BIOPAC")

    class DummyWMI:
        def Win32_PnPEntity(self):
            return [d1]

    monkeypatch.setattr(utils, "wmi", types.SimpleNamespace(WMI=lambda: DummyWMI()))
    # Make import_module raise to simulate pythoncom absence
    monkeypatch.setattr(utils, "importlib", __import__("importlib"))
    monkeypatch.setattr(utils.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("no pythoncom")))

    found = utils.discover_devices()
    assert len(found) == 1
