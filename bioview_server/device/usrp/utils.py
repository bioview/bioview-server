"""
Ref: uhd examples
"""
import threading
import time
from datetime import datetime, timedelta

import uhd
from bioview_common import DISCOVERY_CACHE_TTL, DeviceType, log_print

# Name/serial bookkeeping lives in naming.py so it stays importable (and
# testable) without the UHD driver.
from .naming import (  # noqa: F401  - re-exported for existing callers
    apply_alias,
    get_device_aliases,
    get_usrp_address,
    set_device_alias,
    update_usrp_address,
)


if not hasattr(uhd, "usrp"):
    raise ImportError(
        "Invalid UHD Python bindings: module 'uhd' has no attribute 'usrp'. "
        "Install the USRP Hardware Driver (UHD) Python API from Ettus Research "
        "for your Python version, and ensure no local file named uhd.py is on "
        "PYTHONPATH."
    )


CLOCK_TIMEOUT = 1000  # 1000ms timeout for external clock locking

_discovery_cache: dict[str, dict] = {}
_discovery_cache_ts = 0.0

# ``uhd.find()`` is not safe to call concurrently: overlapping calls have
# taken the process down. The lock also makes the cache read/refresh atomic.
_discovery_lock = threading.Lock()


def invalidate_discovery_cache():
    """Clear cached UHD discovery results (e.g. after a device is unplugged)."""
    global _discovery_cache_ts
    with _discovery_lock:
        _discovery_cache.clear()
        _discovery_cache_ts = 0.0


def discover_devices(logger=None, use_cache: bool = True):
    """Wrap a ``uhd.find`` result (type, name, serial, product) into a payload."""
    global _discovery_cache_ts

    with _discovery_lock:
        return _discover_devices_locked(logger, use_cache)


def _discover_devices_locked(logger, use_cache):
    global _discovery_cache_ts

    if (
        use_cache
        and _discovery_cache
        and (time.monotonic() - _discovery_cache_ts) < DISCOVERY_CACHE_TTL
    ):
        log_print(logger, "debug", "Using cached USRP discovery results")
        return dict(_discovery_cache)

    discovered_devices = {}

    try:
        log_print(logger, "info", "Searching for USRP devices (uhd.find)...")
        device_list = uhd.find("")

        for device in device_list:
            device_dict = apply_alias(dict(device))
            device_dict.setdefault("device_type", DeviceType.USRP.value)
            device_id = device_dict["name"]
            discovered_devices[device_id] = device_dict

            if device_dict.get("serial"):
                update_usrp_address(device_id, device_dict["serial"])

        _discovery_cache.clear()
        _discovery_cache.update(discovered_devices)
        _discovery_cache_ts = time.monotonic()
        log_print(
            logger,
            "info",
            f"USRP discovery complete: {sorted(discovered_devices.keys())}",
        )
    except Exception as e:
        log_print(logger, "error", f"Error occured in UHD device discovery: {e}")

    return discovered_devices


def setup_pps(usrp, pps, num_mboards, logger=None):
    """Setup the PPS source."""
    if pps == "mimo":
        if num_mboards != 2:
            log_print(
                logger,
                "error",
                f'ref = "mimo" implies 2 motherboards; your system has '
                f"{num_mboards} boards",
            )
            return False
        # make mboard 1 a slave over the MIMO Cable
        usrp.set_time_source("mimo", 1)
    else:
        usrp.set_time_source(pps)
    return True


def setup_ref(usrp, ref, num_mboards, logger=None):
    """Setup the reference clock."""
    if ref == "mimo":
        if num_mboards != 2:
            log_print(
                logger,
                "error",
                f'ref = "mimo" implies 2 motherboards; your system has '
                f"{num_mboards} boards",
            )
            return False
        usrp.set_clock_source("mimo", 1)
    else:
        usrp.set_clock_source(ref)

    # Lock onto clock signals for all mboards
    if ref != "internal":
        log_print(logger, "debug", "Now confirming lock on clock signals...")
        end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)
        for i in range(num_mboards):
            if ref == "mimo" and i == 0:
                continue
            is_locked = usrp.get_mboard_sensor("ref_locked", i)
            while (not is_locked) and (datetime.now() < end_time):
                time.sleep(1e-3)
                is_locked = usrp.get_mboard_sensor("ref_locked", i)
            if not is_locked:
                log_print(
                    logger,
                    "error",
                    f"Unable to confirm clock signal locked on board {i}",
                )
                return False
    return True


def check_channels(usrp, rx_channels, tx_channels, logger=None):
    # Check that each Rx channel specified is less than the total number
    # of rx channels that the device can support
    dev_rx_channels = usrp.get_rx_num_channels()
    if not all(map((lambda chan: chan < dev_rx_channels), rx_channels)):
        log_print(logger, "warning", "Invalid RX channel(s) specified.")
        return [], []

    # Check that each Tx channel specified is less than the total number
    # of tx channels that the device can support
    dev_tx_channels = usrp.get_tx_num_channels()
    if not all(map((lambda chan: chan < dev_tx_channels), tx_channels)):
        log_print(logger, "warning", "Invalid TX channel(s) specified.")
        return [], []

    return rx_channels, tx_channels
