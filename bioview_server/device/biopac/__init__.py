from .config import BiopacConfiguration
from .device import BiopacDevice
from .utils import load_mpdev_dll

def get_device_object(device_name, config, resp_queue, data_queue, save=False, save_path=None):
    return BiopacDevice(
        device_name=device_name, config=config, resp_queue=resp_queue, data_queue=data_queue, save=save, save_path=save_path
    )

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

__all__ = ["BiopacDevice", "BiopacConfiguration", "get_device_object", "discover_devices"]
