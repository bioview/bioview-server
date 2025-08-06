from .backend import BIOPACBackend
from .device import BIOPACDevice

from .config import BiopacConfiguration

def discover_devices(): 
    devices = []

    mpdev = load_mpdev_dll()
    
    if mpdev is not None:
        device_list = [] # TODO: Implement
        
        for device in device_list: 
            device_dict = dict(device)
            device_dict['handler_type'] = 'biopac'
            devices.append(device_dict)

    return device

__all__ = [
    "BIOPACBackend",
    "BIOPACDevice"
]