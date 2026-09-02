from .backend import BIOPACBackend
from .utils import discover_devices, load_mpdev_dll


#: No Configurator-editable properties on this backend yet. The Configurator
#: reads this to decide whether the Edit button is available for a device.
EDITABLE_PROPERTIES = {}


def set_device_config(device_info: dict, new_config: dict, logger=None):
    return False, "This device type has no editable properties."


__all__ = [
    "BIOPACBackend",
    "discover_devices",
    "load_mpdev_dll",
    "EDITABLE_PROPERTIES",
    "set_device_config",
]
