"""USRP backend package.

Heavy dependencies (UHD) are loaded lazily so other backends (e.g. dummy RF
simulation) can import ``process`` without requiring USRP drivers.
"""


def __getattr__(name):
    if name == "USRPBackend":
        from .backend import USRPBackend

        return USRPBackend
    if name == "discover_devices":
        from .utils import discover_devices

        return discover_devices
    if name == "update_device_firmware":
        from .utils import update_device_firmware

        return update_device_firmware
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["USRPBackend", "discover_devices", "update_device_firmware"]
