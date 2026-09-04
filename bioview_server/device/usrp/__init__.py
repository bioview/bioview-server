"""USRP backend package.

Heavy dependencies (UHD) are loaded lazily so other backends (e.g. dummy RF
simulation) can import ``process`` without requiring USRP drivers.
"""

# Properties the Configurator may edit, as {name: field spec}. An empty
# mapping means "not editable" and greys out the Edit button.
EDITABLE_PROPERTIES = {
    "device_name": {
        "type": "text",
        "display_name": "Device Name",
        "help": "Name BioView uses for this radio in configuration files.",
        "required": True,
        "max_length": 64,
    }
}


def set_device_config(device_info: dict, new_config: dict, logger=None):
    """Apply Configurator edits to one device. Returns ``(ok, message)``.

    Names are BioView-side aliases keyed on serial, applied during discovery;
    no EEPROM is written. See bioview-docs/reference/usrp.md.
    """
    from .naming import get_device_aliases, set_device_alias, update_usrp_address

    serial = (device_info or {}).get("serial")
    if not serial:
        return False, "Device has no serial number; cannot store a name for it."

    new_name = str((new_config or {}).get("device_name", "")).strip()
    if not new_name:
        return False, "Device name cannot be empty."
    if len(new_name) > 64:
        return False, "Device name must be 64 characters or fewer."

    taken = {name: sn for sn, name in get_device_aliases().items() if sn != serial}
    if new_name in taken:
        return (
            False,
            f"Another device (serial {taken[new_name]}) already uses that name.",
        )

    if not set_device_alias(serial, new_name, logger=logger):
        return False, "Could not write the device name cache."

    update_usrp_address(new_name, serial, logger=logger)

    # The alias is overlaid while building the uhd.find cache, so the cache
    # has to be dropped for a rename to show up.
    try:
        from .utils import invalidate_discovery_cache

        invalidate_discovery_cache()
    except Exception:  # UHD absent: nothing was cached to drop
        pass

    return True, f"Renamed to {new_name}."


def __getattr__(name):
    if name == "USRPBackend":
        from .backend import USRPBackend

        return USRPBackend
    if name == "discover_devices":
        from .utils import discover_devices

        return discover_devices
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "USRPBackend",
    "discover_devices",
    "EDITABLE_PROPERTIES",
    "set_device_config",
]
