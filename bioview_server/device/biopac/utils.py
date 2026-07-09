import contextlib
import ctypes
import json
import os 
import logging
import importlib
from pathlib import Path
from ctypes import byref, c_double, c_int
from pathlib import Path

import wmi

from bioview_common import get_cache_file

from .constants import BIOPAC_CONNECTION_CODES, BIOPAC_VENDOR_ID


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
    # Discover BIOPAC devices connected over USB.
    discovered_devices = []
    coinit = False
    pythoncom = None
    try:
        # Try to initialise COM for this thread so WMI works when called
        # from worker threads. If pythoncom isn't available, continue and
        # let wmi raise a helpful error.
        try:
            pythoncom = importlib.import_module("pythoncom")
            # Prefer CoInitializeEx for safety; fallback to CoInitialize
            if hasattr(pythoncom, "CoInitializeEx"):
                pythoncom.CoInitializeEx(0x2)  # COINIT_MULTITHREADED
            else:
                pythoncom.CoInitialize()
            coinit = True
        except Exception:
            pythoncom = None

        c = wmi.WMI()
        # Query USB devices from WMI
        for device in c.Win32_PnPEntity():
            if device.DeviceID and "USB" in device.DeviceID:
                vid = pid = None
                if "VID_" in device.DeviceID and "PID_" in device.DeviceID:
                    with contextlib.suppress(Exception):
                        vid_start = device.DeviceID.find("VID_") + 4
                        vid = device.DeviceID[vid_start : vid_start + 4]
                        pid_start = device.DeviceID.find("PID_") + 4
                        pid = device.DeviceID[pid_start : pid_start + 4]

                device_info = {
                    "device_id": device.DeviceID,
                    "name": device.Name or "Unknown",
                    "description": device.Description or "Unknown",
                    "manufacturer": device.Manufacturer or "Unknown",
                    "service": device.Service or "None",
                    "status": device.Status or "Unknown",
                    "present": device.Present,
                    "vid": vid,
                    "pid": pid,
                }

                # Normalise VID for numeric comparison
                vid_int = None
                if vid is not None:
                    try:
                        vid_int = int(vid, 16)
                    except Exception:
                        vid_int = None

                # Validate and add to list
                if vid_int == BIOPAC_VENDOR_ID or \
                'biopac' in (device_info['manufacturer'] or '').lower() or \
                'biopac' in (device_info['name'] or '').lower():  
                    discovered_devices.append(device_info)
    except Exception as e:
        logging.getLogger(__name__).error("Unable to discover BIOPAC devices: %s", e)

    finally:
        # Uninitialize pythoncom if we initialized it here
        if coinit and pythoncom is not None:
            try:
                if hasattr(pythoncom, "CoUninitialize"):
                    pythoncom.CoUninitialize()
            except Exception:
                pass

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


def update_device_firmware():
    """BIOPAC firmware updates are managed outside BioView."""
    pass


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


# Wrappers for BIOPAC operations
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
        raise Exception(f"BIOPAC Connection Failed with Error Code: {result_code}")


def configure_biopac_device(mpdev_handler, channels, sample_rate):
    # Set channels
    result_code = mpdev_handler.setAcqChannels(byref(channels))
    if BIOPAC_CONNECTION_CODES.get(result_code, None) != "MPSUCCESS":
        raise Exception(
            f"BIOPAC Channel Configuration Failed with Error Code: {result_code}"
        )

    # Set sample rate
    result_code = mpdev_handler.setSampleRate(c_double(1000.0 / sample_rate))
    if BIOPAC_CONNECTION_CODES.get(result_code, None) != "MPSUCCESS":
        raise Exception(
            f"BIOPAC Sample Rate Configuration Failed with Error Code: {result_code}"
        )


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
    if hasattr(mpdev_handler, "disconnectMPDev"):
        result_code = mpdev_handler.disconnectMPDev()
        if BIOPAC_CONNECTION_CODES.get(result_code, None) not in (
            "MPSUCCESS",
            None,
        ):
            raise Exception(
                f"BIOPAC Disconnect Failed with Error Code: {result_code}"
            )


def wrap_result_code(result, stage=""):
    result_code = BIOPAC_CONNECTION_CODES.get(result, "INVALID_CODE")
    if result_code == "MPSUCCESS":
        return True
    else:
        raise Exception(f"{stage} Failure - {result_code}")


def get_mpdev_path():
    cache_file = get_cache_file("mpdev_path")

    try:
        with open(cache_file) as fobj:
            dll_path = json.load(fobj)
    except Exception:
        print("mpdev path is not cached")
        return None

    return dll_path


def update_mpdev_path(dll_path):
    cache_file = get_cache_file("mpdev_path")

    try:
        with open(cache_file, "w") as fobj:
            json.dump(str(dll_path), fobj)
    except Exception as e:
        print(f"Error updating cache: {e}")
