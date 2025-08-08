import ctypes
import os
import wmi 
import json
from pathlib import Path
from ctypes import byref, c_double, c_int

from .constants import BIOPAC_VENDOR_ID, BIOPAC_CONNECTION_CODES

from bioview_server.utils import get_cache_file

def discover_devices(): 
    # Discover BIOPAC devices connected over USB.
    discovered_devices = []
    
    c = wmi.WMI()
    try:
        # Query USB devices from WMI
        for device in c.Win32_PnPEntity():
            if device.DeviceID and 'USB' in device.DeviceID:
                vid = pid = None
                if 'VID_' in device.DeviceID and 'PID_' in device.DeviceID:
                    try:
                        vid_start = device.DeviceID.find('VID_') + 4
                        vid = device.DeviceID[vid_start:vid_start + 4]
                        pid_start = device.DeviceID.find('PID_') + 4
                        pid = device.DeviceID[pid_start:pid_start + 4]
                    except:
                        pass
                
                device_info = {
                    'device_id': device.DeviceID,
                    'name': device.Name or 'Unknown',
                    'description': device.Description or 'Unknown',
                    'manufacturer': device.Manufacturer or 'Unknown',
                    'service': device.Service or 'None',
                    'status': device.Status or 'Unknown',
                    'present': device.Present,
                    'vid': vid,
                    'pid': pid
                }

                # Validate and add to list
                if device_info['vid'] == BIOPAC_VENDOR_ID or \
                'biopac' in device_info['manufacturer'].lower() or \
                'biopac' in device_info['name'].lower():  
                    discovered_devices.append(device_info)
                    
    except Exception as e:
        print(f"Unable to discover BIOPAC devices: {e}")

    return discovered_devices

def load_mpdev_dll(custom_loc: str = None):
    dll = None
    try:
        dll = ctypes.CDLL("mpdev.dll")
        print("mpdev.dll found!")
        return
    except FileNotFoundError:
        print("mpdev.dll is not located in $PATH")

    # Check custom loc
    if custom_loc is not None:
        print(f"Searching for mpdev.dll in {custom_loc}")
        dll_locs = Path(custom_loc).glob("**/mpdev.dll")
        for loc in dll_locs:
            dll = ctypes.CDLL(loc)
            print("mpdev.dll found!")
            return dll

    # Check root diretory - Check cache before searching
    dll_path = get_mpdev_path()
    if dll_path is not None:
        print("mpdev.dll found!")
        return ctypes.CDLL(dll_path)
    else:
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
def connect_biopac_device(mpdev_handler, device_code: int = 103, connection_code: int = 10):
    result_code = mpdev_handler.connectMPDev(c_int(device_code), c_int(connection_code), b"auto")
    if BIOPAC_CONNECTION_CODES.get(result_code, None)  != "MPSUCCESS":
        raise Exception(f"BIOPAC Connection Failed with Error Code: {result_code}")
    
def configure_biopac_device(mpdev_handler, channels, sample_rate):
    # Set channels 
    result_code = mpdev_handler.setAcqChannels(byref(channels))
    if BIOPAC_CONNECTION_CODES.get(result_code, None)  != "MPSUCCESS":
        raise Exception(f"BIOPAC Channel Configuration Failed with Error Code: {result_code}")
    
    # Set sample rate 
    result_code = mpdev_handler.setSampleRate(c_double(1000.0/sample_rate))
    if BIOPAC_CONNECTION_CODES.get(result_code, None)  != "MPSUCCESS":
        raise Exception(f"BIOPAC Sample Rate Configuration Failed with Error Code: {result_code}")

def start_acquisition(mpdev_handler): 
    result_code = mpdev_handler.startAcquisition()
    if BIOPAC_CONNECTION_CODES.get(result_code, None)  != "MPSUCCESS":
        raise Exception(f"BIOPAC Acquisition Start Failed with Error Code: {result_code}")
    
def stop_acquisition(mpdev_handler): 
    result_code = mpdev_handler.stopAcquisition()
    if BIOPAC_CONNECTION_CODES.get(result_code, None)  != "MPSUCCESS":
        raise Exception(f"BIOPAC Acquisition Stopping Failed with Error Code: {result_code}")

def wrap_result_code(result, stage=""):
    result_code = BIOPAC_CONNECTION_CODES.get(result, "INVALID_CODE")
    if result_code == "MPSUCCESS":
        return True
    else:
        raise Exception(f"{stage} Failure - {result_code}")

def get_mpdev_path():
    cache_file = get_cache_file("mpdev_path")

    try:
        with open(cache_file, "r") as fobj:
            dll_path = json.load(fobj)
    except Exception as e:
        print("mpdev path is not cached")
        return None

    return dll_path


def update_mpdev_path(dll_path):
    cache_file = get_cache_file("mpdev_path")

    try:
        with open(cache_file, "r") as fobj:
            dll_path = json.load(fobj)
    except Exception as e:
        print("mpdev path is not cached currently. Storing in cache.")

    # Update
    try:
        with open(cache_file, "w") as fobj:
            json.dump(str(dll_path), fobj)
    except Exception as e:
        print(f"Error updating cache: {e}")
    finally:
        print("Successfully stored in cache")