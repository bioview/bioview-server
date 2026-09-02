"""BIOPAC discovery: it must find an attached unit, and say why when it cannot."""
import sys

import pytest


wmi = pytest.importorskip("wmi", reason="BIOPAC discovery needs the Windows wmi package")

if sys.platform != "win32":
    pytest.skip("BIOPAC is Windows-only", allow_module_level=True)

# Imported after the skips above: on a non-Windows machine importing the
# discovery module at all would fail on `import wmi`.
from bioview_server.device.biopac import utils as biopac_utils  # noqa: E402
from bioview_server.device.biopac.constants import BIOPAC_VENDOR_ID  # noqa: E402


def test_the_vendor_id_matches_what_windows_reports_for_biopac_hardware():
    # BIOPAC Systems' USB vendor id, as it appears in a Windows DeviceID
    # (USB\VID_097E&PID_....). Discovery keys off this, so a wrong value here
    # means attached hardware is never recognised.
    assert BIOPAC_VENDOR_ID == 0x097E


def test_discovery_reports_a_failure_instead_of_swallowing_it(monkeypatch, caplog):
    """A `return` inside the `finally` block used to discard the exception, so a
    genuine WMI or COM failure was indistinguishable from "nothing attached"."""

    def explode():
        raise RuntimeError("WMI is unhappy")

    monkeypatch.setattr(biopac_utils.wmi, "WMI", explode)

    with caplog.at_level("ERROR"):
        found = biopac_utils._discover_devices_list()

    assert found == []
    assert "WMI is unhappy" in caplog.text


def test_discovery_returns_a_usable_mapping():
    devices = biopac_utils.discover_devices()
    assert isinstance(devices, dict)
    for key, info in devices.items():
        assert key and isinstance(key, str)
        assert info.get("name")
        assert info.get("device_id")


@pytest.mark.hardware
def test_an_attached_biopac_unit_is_found():
    """Only meaningful with a BIOPAC unit plugged in; skipped otherwise."""
    devices = biopac_utils.discover_devices()
    if not devices:
        pytest.skip("no BIOPAC hardware attached to this machine")
    assert any("biopac" in info["name"].lower() for info in devices.values())


class TestIdentifyingDetails:
    """The Configurator needs something real to show under a device's name."""

    def test_a_synthesised_windows_instance_id_is_not_a_serial_number(self):
        # An MP36 supplies no USB serial, so Windows makes an id up from the
        # port path. Showing that to the user as "S/N" would be a fiction.
        assert (
            biopac_utils._usb_serial_from_device_id(
                r"USB\VID_097E&PID_0036\5&2887CE2C&0&1"
            )
            is None
        )

    def test_a_real_serial_is_read_from_the_instance_path(self):
        assert (
            biopac_utils._usb_serial_from_device_id(r"USB\VID_097E&PID_00A0\MP160-01234")
            == "MP160-01234"
        )

    @pytest.mark.parametrize("device_id", [None, "", "USB\\VID_097E&PID_0036\\"])
    def test_a_missing_or_empty_instance_path_yields_no_serial(self, device_id):
        assert biopac_utils._usb_serial_from_device_id(device_id) is None

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("BIOPAC MP36 USB Data Acquisition Unit", "MP36"),
            ("BIOPAC MP160 System", "MP160"),
            ("biopac mp150", "MP150"),
            ("Some Other Device", None),
        ],
    )
    def test_the_model_is_read_from_the_device_name(self, name, expected):
        assert biopac_utils._model_from_name(name) == expected

    def test_the_model_falls_back_to_the_description(self):
        assert biopac_utils._model_from_name(None, "BIOPAC MP36 Unit") == "MP36"

    def test_discovered_devices_carry_the_details_the_configurator_shows(self):
        for info in biopac_utils.discover_devices().values():
            assert "serial" in info, "the Configurator reads this key"
            assert "model" in info
            assert "status" in info
