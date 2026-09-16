import contextlib
import ctypes
import json
import logging
import os
import threading
from ctypes import byref, c_double, c_int
from pathlib import Path


try:
    import wmi
except ImportError:  # pragma: no cover
    wmi = None
from bioview_common import get_cache_file, log_print

from .constants import (
    BIOPAC_CONNECTION_CODES,
    BIOPAC_VENDOR_ID,
    describe_biopac_code,
)


_HVCI_SERVICE_ID = 2

_HVCI_QUERY_TIMEOUT = 3.0

_hvci_state = None


def _query_memory_integrity():
    """Ask Windows which security services are configured and running."""
    import wmi as _wmi

    pythoncom = _com_module()
    coinit = False
    if pythoncom is not None:
        with contextlib.suppress(Exception):
            pythoncom.CoInitializeEx(0x2)
            coinit = True
    try:
        for entry in _wmi.WMI(
            namespace=r"root\Microsoft\Windows\DeviceGuard"
        ).Win32_DeviceGuard():
            running = set(entry.SecurityServicesRunning or [])
            configured = set(entry.SecurityServicesConfigured or [])
            return _HVCI_SERVICE_ID in running, _HVCI_SERVICE_ID in configured
    finally:
        if coinit:
            with contextlib.suppress(Exception):
                pythoncom.CoUninitialize()

    return False, False


def memory_integrity_state():
    """Whether Windows Memory Integrity is running, and whether it is configured."""
    global _hvci_state
    if _hvci_state is not None:
        return _hvci_state

    if os.name != "nt":
        _hvci_state = (False, False)
        return _hvci_state

    result = {}

    def _run():
        with contextlib.suppress(Exception):
            result["state"] = _query_memory_integrity()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=_HVCI_QUERY_TIMEOUT)

    if "state" in result:
        _hvci_state = result["state"]
        return _hvci_state

    configured = _memory_integrity_configured_in_registry()
    _hvci_state = (configured, configured)
    return _hvci_state


def _memory_integrity_configured_in_registry() -> bool:
    """The configured Memory Integrity setting, read straight from the policy key."""
    try:
        import winreg
    except Exception:
        return False

    key_path = (
        r"SYSTEM\CurrentControlSet\Control\DeviceGuard\Scenarios"
        r"\HypervisorEnforcedCodeIntegrity"
    )
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            enabled, _ = winreg.QueryValueEx(key, "Enabled")
        return bool(enabled)
    except Exception:
        return False


def driver_failure_hint() -> str:
    """What this machine's configuration adds to a driver that would not start."""
    running, configured = memory_integrity_state()
    if not running:
        return ""

    if configured:
        return " Memory Integrity is enabled on this machine."

    return (
        " Memory Integrity is switched off but still running until this "
        "machine is restarted."
    )


def _usb_serial_from_device_id(device_id):
    """The unit's own serial number from a Windows instance path, if it has one."""
    if not device_id:
        return None
    tail = str(device_id).rsplit("\\", 1)[-1].strip()
    if not tail or "&" in tail:
        return None
    return tail


def _model_from_name(*candidates):
    """The BIOPAC model (MP36, MP150, ...) named in any of these strings."""
    from bioview_common.datatypes.configuration.biopac import MODEL_CODE_MAPPING

    for text in candidates:
        upper = (text or "").upper()
        for model in sorted(MODEL_CODE_MAPPING, key=len, reverse=True):
            if model in upper:
                return model
    return None


def _com_module():
    """The pythoncom module, or None when pywin32 is unavailable."""
    try:
        import pythoncom
    except Exception:
        return None
    return pythoncom


def discover_devices():
    """Discover BIOPAC USB devices; returns {hardware_key: device_info}."""
    discovered_devices = {}
    discovered_list = _discover_devices_list()
    for index, device_info in enumerate(discovered_list):
        key = _discovery_key(device_info, index)
        discovered_devices[key] = device_info
    return discovered_devices


def _discovery_key(device_info: dict, index: int) -> str:
    name = (device_info.get("name") or "").strip()
    if name:
        key = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        key = key.strip("_") or f"BIOPAC_{index}"
        return key
    return f"BIOPAC_{index}"


def _discover_devices_list():
    discovered_devices = []
    coinit = False
    pythoncom = None
    try:
        pythoncom = _com_module()
        if pythoncom is not None:
            try:
                pythoncom.CoInitializeEx(0x2)
                coinit = True
            except Exception:
                pythoncom = None

        c = wmi.WMI()
        for device in c.Win32_PnPEntity():
            if device.DeviceID and "USB" in device.DeviceID:
                vid = pid = None
                if "VID_" in device.DeviceID and "PID_" in device.DeviceID:
                    with contextlib.suppress(Exception):
                        vid_start = device.DeviceID.find("VID_") + 4
                        vid = device.DeviceID[vid_start : vid_start + 4]
                        pid_start = device.DeviceID.find("PID_") + 4
                        pid = device.DeviceID[pid_start : pid_start + 4]

                name = device.Name or "Unknown"
                device_info = {
                    "device_id": device.DeviceID,
                    "name": name,
                    "description": device.Description or "Unknown",
                    "manufacturer": device.Manufacturer or "Unknown",
                    "service": device.Service or "None",
                    "status": device.Status or "Unknown",
                    "present": device.Present,
                    "vid": vid,
                    "pid": pid,
                    "serial": _usb_serial_from_device_id(device.DeviceID),
                    "model": _model_from_name(name, device.Description),
                }

                vid_int = None
                if vid is not None:
                    try:
                        vid_int = int(vid, 16)
                    except Exception:
                        vid_int = None

                if (
                    vid_int == BIOPAC_VENDOR_ID
                    or "biopac" in (device_info["manufacturer"] or "").lower()
                    or "biopac" in (device_info["name"] or "").lower()
                ):
                    discovered_devices.append(device_info)
    except Exception as e:
        logging.getLogger(__name__).error("Unable to discover BIOPAC devices: %s", e)

    finally:
        if coinit and pythoncom is not None:
            with contextlib.suppress(Exception):
                pythoncom.CoUninitialize()

    return discovered_devices


def build_hardware_dict_from_group(group_config: dict, group_id: str) -> dict:
    hardware = group_config.get("hardware")
    if hardware:
        return dict(hardware)

    entry = {
        k: v
        for k, v in group_config.items()
        if k
        in {
            "channels",
            "model",
            "connection_type",
            "port",
            "samp_rate",
            "labels",
            "device_code",
        }
    }
    device_name = group_config.get("device_name") or group_id
    return {device_name: entry}


def resolve_biopac_hardware_entry(
    hardware: dict, discovered_devices: dict | None = None
) -> dict:
    """Pick the first hardware entry (BIOPAC groups are typically single-unit)."""
    if not hardware:
        return {}
    if len(hardware) == 1:
        return dict(next(iter(hardware.values())))
    if discovered_devices:
        for key, entry in hardware.items():
            if key in discovered_devices:
                return dict(entry)
    return dict(next(iter(hardware.values())))


def load_mpdev_dll(custom_loc: str = None):
    dll = None
    try:
        dll = ctypes.CDLL("mpdev.dll")
        print("mpdev.dll found!")
        return dll
    except FileNotFoundError:
        print("mpdev.dll is not located in $PATH")

    if custom_loc is not None:
        print(f"Searching for mpdev.dll in {custom_loc}")
        dll_locs = Path(custom_loc).glob("**/mpdev.dll")
        for loc in dll_locs:
            dll = ctypes.CDLL(loc)
            print("mpdev.dll found!")
            return dll

    dll_path = get_mpdev_path()
    if dll_path is not None:
        print("mpdev.dll found!")
        return ctypes.CDLL(dll_path)

    print("Searching for mpdev.dll in OS folders")
    sys_dir = Path(os.path.abspath(os.sep))
    dll_locs = sys_dir.glob("Program Files*/BIOPAC*/**/x64/mpdev.dll")

    for loc in dll_locs:
        update_mpdev_path(loc)
        dll = ctypes.CDLL(loc)
        print("mpdev.dll found!")
        return dll

    return None


def connect_biopac_device(
    mpdev_handler,
    device_code: int = 103,
    connection_code: int = 10,
    port: str = "auto",
):
    port_bytes = port.encode("utf-8") if isinstance(port, str) else port
    result_code = mpdev_handler.connectMPDev(
        c_int(device_code), c_int(connection_code), port_bytes
    )
    if BIOPAC_CONNECTION_CODES.get(result_code, None) != "MPSUCCESS":
        raise Exception(f"BIOPAC connection failed: {describe_biopac_code(result_code)}")


def configure_biopac_device(mpdev_handler, channels, sample_rate):
    result_code = mpdev_handler.setAcqChannels(byref(channels))
    if BIOPAC_CONNECTION_CODES.get(result_code, None) != "MPSUCCESS":
        raise Exception(
            f"BIOPAC channel configuration failed: {describe_biopac_code(result_code)}"
        )

    result_code = mpdev_handler.setSampleRate(c_double(1000.0 / sample_rate))
    if BIOPAC_CONNECTION_CODES.get(result_code, None) != "MPSUCCESS":
        raise Exception(
            "BIOPAC sample rate configuration failed: "
            f"{describe_biopac_code(result_code)}"
        )


def start_acq_daemon(mpdev_handler) -> bool:
    """Start mpdev's acquisition daemon, which backs ``receiveMPData``."""
    start = getattr(mpdev_handler, "startMPAcqDaemon", None)
    if start is None:
        return False

    result_code = start()
    if BIOPAC_CONNECTION_CODES.get(result_code, None) != "MPSUCCESS":
        raise Exception(
            f"BIOPAC acquisition daemon failed to start: "
            f"{describe_biopac_code(result_code)}"
        )
    return True


def daemon_last_error(mpdev_handler, logger=None):
    """mpdev's daemon-specific error code, or None if the DLL cannot report one."""
    last_error = getattr(mpdev_handler, "getMPDaemonLastError", None)
    if last_error is None:
        return None
    try:
        return last_error()
    except Exception as e:
        log_print(logger, "debug", f"getMPDaemonLastError failed: {e}")
        return None


def start_acquisition(mpdev_handler):
    result_code = mpdev_handler.startAcquisition()
    if BIOPAC_CONNECTION_CODES.get(result_code, None) != "MPSUCCESS":
        raise Exception(
            f"BIOPAC Acquisition Start Failed with Error Code: {result_code}"
        )


def stop_acquisition(mpdev_handler):
    result_code = mpdev_handler.stopAcquisition()
    if BIOPAC_CONNECTION_CODES.get(result_code, None) != "MPSUCCESS":
        raise Exception(
            f"BIOPAC Acquisition Stopping Failed with Error Code: {result_code}"
        )


def disconnect_biopac_device(mpdev_handler):
    result_code = mpdev_handler.disconnectMPDev()
    if BIOPAC_CONNECTION_CODES.get(result_code, None) not in ("MPSUCCESS", None):
        raise Exception(f"BIOPAC Disconnect Failed with Error Code: {result_code}")


def get_mpdev_path(logger=None):
    """Cached mpdev.dll path, or None when there is not one to use."""
    cache_file = get_cache_file("mpdev_path")
    try:
        with open(cache_file) as fobj:
            return json.load(fobj)
    except FileNotFoundError:
        log_print(logger, "debug", "mpdev path is not cached yet")
        return None
    except (OSError, json.JSONDecodeError) as e:
        log_print(
            logger,
            "warning",
            f"mpdev path cache at {cache_file} is unreadable ({e}); "
            "searching for the DLL again.",
        )
        return None


def update_mpdev_path(dll_path):
    cache_file = get_cache_file("mpdev_path")

    try:
        with open(cache_file, "w") as fobj:
            json.dump(str(dll_path), fobj)
    except Exception as e:
        print(f"Error updating cache: {e}")
