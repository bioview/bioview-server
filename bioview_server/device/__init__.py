# Try to load all backends and provide
import sys
import multiprocessing as mp 

from bioview_common import log_print, DeviceType, SUPPORTED_DEVICES
from bioview_common.utils.logs import suppress_stdout


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

try:
    # Virtual device: always available, no hardware or platform requirements.
    from . import dummy

    __all__.append("dummy")
    AVAILABLE_BACKENDS[DeviceType.DUMMY.value] = dummy
except Exception as e:
    print(f"DUMMY backend not available: {e}")


def get_device_handler(
        device_id, 
        device_cfg, 
        response_queue: mp.Queue, 
        data_output_queue: mp.Queue,
        logger = None,
        discovered_devices: dict = None,
    ):
    device_type = device_cfg.get_param("device_type")
    
    if device_type not in SUPPORTED_DEVICES:
        log_print(logger, "error", f"Unsupported device type: {device_type}")
        return None
    elif device_type not in AVAILABLE_BACKENDS:
        log_print(logger, "warning", f"Backend not available for {device_type}") 
        return None
    
    match device_type: 
        case DeviceType.USRP.value:
            group_cfg = device_cfg.to_dict()
            hardware = group_cfg.get("hardware")
            if hardware:
                devices = {name: dict(hw) for name, hw in hardware.items()}
            else:
                devices = {device_id: group_cfg}

            handler = AVAILABLE_BACKENDS.get(DeviceType.USRP.value).USRPBackend(
                group_id=device_id,
                samp_rate=device_cfg.get_param("samp_rate"),
                devices=devices,
                group_config=group_cfg,
                response_queue=response_queue,
                data_output_queue=data_output_queue,
                display_ds=device_cfg.get_param("disp_ds", 10),
                display_imaginary=device_cfg.get_param("display_imaginary", False),
                save_ds=device_cfg.get_param("save_ds", 1),
                save_iq=device_cfg.get_param("save_iq", False),
                save_imaginary=device_cfg.get_param("save_imaginary", True),
                discovered_devices=discovered_devices,
            )
        
        case DeviceType.BIOPAC.value: 
            handler = AVAILABLE_BACKENDS.get(DeviceType.BIOPAC.value).BIOPACBackend(
                group_id=device_id,
                response_queue=response_queue,
                data_output_queue=data_output_queue,
                group_config=device_cfg.to_dict(),
                discovered_devices=discovered_devices,
            )

        case DeviceType.DUMMY.value:
            handler = AVAILABLE_BACKENDS.get(DeviceType.DUMMY.value).DummyBackend(
                group_id=device_id,
                response_queue=response_queue,
                data_output_queue=data_output_queue,
                group_config=device_cfg.to_dict(),
            )

        case _:
            handler = None
        
    return handler  
  

__all__ = ["AVAILABLE_BACKENDS", "get_device_handler"]
