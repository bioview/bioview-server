# Try to load all backends and provide
import sys
import multiprocessing as mp 

from bioview_common import log_print, DeviceType, Configuration, SUPPORTED_DEVICES
from bioview_server.utils import suppress_stdout


__all__ = []

AVAILABLE_BACKENDS = {}

try:
    # Ensure uhd is available

    # Ensure device is importable
    from . import usrp

    __all__.append("usrp")
    AVAILABLE_BACKENDS[DeviceType.USRP.value] = usrp
except Exception as e:
    print(f"USRP backend not available: {e}")

try:
    # Ensure platform is windows
    if sys.platform != "win32":
        raise OSError(f"Invalid platfrom {sys.platform}. Ensure you are using Windows")

    from . import biopac

    # Ensure mpdev.dll exists
    with suppress_stdout():
        if biopac.load_mpdev_dll() is None:
            raise ValueError("mpdev.dll not found")

    __all__.append("biopac")
    AVAILABLE_BACKENDS[DeviceType.BIOPAC.value] = biopac
except Exception as e:
    print(f"BIOPAC backend not available: {e}")

def get_device_group_handler(
        group_dict, 
        response_queue: mp.Queue, 
        logger = None
    ): 
    '''
    group_dict = {
        'metadata': { 
            Common information such as type of devices, 
            overall group connectivity state, and so on.
        },
        'device_id_1': { # Device params },
        ...
        'device_id_N': { # Device params }
    }
    Each group has one associated handler
    '''

    metadata = group_dict.get("metadata", {})
    group_id = metadata.get("group_id", "")
    device_type = metadata.get("device_type", None)
    
    if device_type not in SUPPORTED_DEVICES:
        log_print(logger, 'error', f'Unsupported device type: {device_type}. Supported device types are: {SUPPORTED_DEVICES}')
        return 
    elif device_type not in AVAILABLE_BACKENDS:
        log_print(logger, 'warning', f'Unable to initialize group {group_id} as backend not available on machine. ') 
        return 
    elif not device_type:
        log_print(logger, 'error', f'Group {group_id} has no device type specified')
        return 
    
    # If everything works, initialize 
    match device_type: 
        case DeviceType.USRP.value: 

            handler = AVAILABLE_BACKENDS.get(DeviceType.USRP.value).USRPBackend(
                group_id = group_id, 
                samp_rate = metadata.get('samp_rate'),  
                devices = {k: v for k, v in group_dict.items() if k != "metadata"},
                response_queue = response_queue, 
            )
        
        case DeviceType.BIOPAC.value: 
            # We only support one BIOPAC device per group for now. Hence, 
            # we assume that there is only one dict provided
            device_dict = next(iter(group_dict.values())) 
            device_cfg = Configuration.from_dict(device_dict, DeviceType.BIOPAC.value)
            
            handler = AVAILABLE_BACKENDS.get(DeviceType.BIOPAC.value).BIOPACBackend(
                group_id = group_id, 
                response_queue = response_queue, 
                samp_rate = device_cfg.get_param('samp_rate'),  
                mpdev_path = device_cfg.get_param('mpdev_path'), 
                device_code = device_cfg.get_param('device_code')
            )
        
    return handler  

__all__ = ["AVAILABLE_BACKENDS", "get_device_group_handler"]
