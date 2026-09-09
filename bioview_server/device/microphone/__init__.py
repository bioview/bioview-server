from .backend import MicrophoneBackend
from .utils import (
    check_available,
    discover_devices,
    list_input_devices,
    resolve_input_device,
)


#: No Configurator-editable properties on this backend: a host audio input is
#: named by the operating system and BioView has no say in it. The Configurator
#: reads this to decide whether the Edit button is available for a device.
EDITABLE_PROPERTIES = {}


def set_device_config(device_info: dict, new_config: dict, logger=None):
    return False, "This device type has no editable properties."


__all__ = [
    "MicrophoneBackend",
    "check_available",
    "discover_devices",
    "list_input_devices",
    "resolve_input_device",
    "EDITABLE_PROPERTIES",
    "set_device_config",
]
