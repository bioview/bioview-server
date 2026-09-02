from .backend import DummyBackend, SineWaveWorker


def discover_devices():
    """Report the virtual dummy backend as present on the server."""
    return [
        {
            "name": "DummyVirtual",
            "type": "dummy",
            "device_type": "dummy",
            "serial": "virtual",
        }
    ]


#: No Configurator-editable properties on this backend yet. The Configurator
#: reads this to decide whether the Edit button is available for a device.
EDITABLE_PROPERTIES = {}


def set_device_config(device_info: dict, new_config: dict, logger=None):
    return False, "This device type has no editable properties."


__all__ = [
    "DummyBackend",
    "SineWaveWorker",
    "discover_devices",
    "EDITABLE_PROPERTIES",
    "set_device_config",
]
