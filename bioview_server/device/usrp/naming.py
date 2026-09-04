"""USRP name/serial bookkeeping. Free of any UHD import, deliberately.

``usrp_serial_numbers`` maps name -> serial; ``usrp_device_aliases`` maps
serial -> user-assigned name. See bioview-docs/reference/usrp.md.
"""

from __future__ import annotations

import json

from bioview_common import get_cache_file, log_print


SERIAL_CACHE = "usrp_serial_numbers"
ALIAS_CACHE = "usrp_device_aliases"


def _read_cache(name: str) -> dict:
    try:
        with open(get_cache_file(name)) as fobj:
            data = json.load(fobj)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_cache(name: str, data: dict, logger=None) -> bool:
    try:
        with open(get_cache_file(name), "w") as fobj:
            json.dump(data, fobj, indent=2)
        return True
    except Exception as e:
        log_print(logger, "error", f"Error updating cache {name}: {e}")
        return False


def get_usrp_address(device_name: str, logger=None) -> str | None:
    """Serial for a device name, or None if this machine has never seen it."""
    return _read_cache(SERIAL_CACHE).get(device_name)


def update_usrp_address(device_name: str, device_serial: str, logger=None) -> bool:
    cache = _read_cache(SERIAL_CACHE)
    cache[device_name] = device_serial
    return _write_cache(SERIAL_CACHE, cache, logger)


def get_device_aliases() -> dict[str, str]:
    """serial -> user-assigned name."""
    return _read_cache(ALIAS_CACHE)


def set_device_alias(serial: str, name: str, logger=None) -> bool:
    """Assign, or with a falsy name clear, the user-facing name for a serial."""
    aliases = get_device_aliases()
    if name:
        aliases[str(serial)] = str(name)
    else:
        aliases.pop(str(serial), None)
    return _write_cache(ALIAS_CACHE, aliases, logger)


def apply_alias(device_dict: dict) -> dict:
    """Overlay the user-assigned name onto one ``uhd.find`` result."""
    aliases = get_device_aliases()
    serial = device_dict.get("serial", "")
    eeprom_name = device_dict.get("name", "invalid_usrp_device")
    device_dict["eeprom_name"] = eeprom_name
    device_dict["name"] = aliases.get(serial, eeprom_name)
    return device_dict
