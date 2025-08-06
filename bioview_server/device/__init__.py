'''
Each device must be imported with exception handling since all drivers are not guaranteed to be installed
'''

from ..constants.device import Device

__all__ = [
    "Device"
]

device_object_callbacks = {}

def _check_type(obj, typ):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return all(isinstance(x, typ) for x in obj)

    return isinstance(obj, typ)

from bioview_server.constants import Configuration

def get_device_object(device_name, config, resp_queue, data_queue, save, save_path):
    # config files must always be of type Configuration
    if not isinstance(config, Configuration):
        raise TypeError(f'Specified config must be of type Configuration but got {type(config)} instead.')
    
    try: 
        if config.type == 'usrp': 
            from .usrp import get_device_object
        elif config.type == 'biopac': 
            from .biopac import get_device_object
        else: 
            raise NotImplementedError(f'Unable to parse configuration for device type {config.type}')
    
        return get_device_object(
            device_name=device_name,
            config=config,
            resp_queue=resp_queue, 
            data_queue=data_queue, 
            save=save,
            save_path=save_path,
        )
    except Exception as e: 
        print(f'Unable to create {config.type} device: {e}')

def discover_devices(): 
    devices = []
    
    # For all backends, this will discover devices
    try:     
        from .usrp import discover_devices
        devices.extend(discover_devices())
    except Exception as e: 
        print(f'Error getting USRP devices: {e}')
    
    try: 
        from .biopac import discover_devices
        devices.extend(discover_devices())
    except Exception as e: 
        print(f'Error getting BIOPAC devices: {e}')
    
    return devices

__all__.extend([
    "get_device_object", 
    "discover_devices"
])