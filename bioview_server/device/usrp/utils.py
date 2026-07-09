"""
Ref: uhd examples
"""
import json
import time
import contextlib
from typing import Dict, List
from datetime import datetime, timedelta

import numpy as np
import uhd
from bioview_common import DISCOVERY_CACHE_TTL, DataSource, log_print

from bioview_common import get_cache_file


if not hasattr(uhd, "usrp"):
    raise ImportError(
        "Invalid UHD Python bindings: module 'uhd' has no attribute 'usrp'. "
        "Install the USRP Hardware Driver (UHD) Python API from Ettus Research "
        "for your Python version, and ensure no local file named uhd.py is on PYTHONPATH."
    )


CLOCK_TIMEOUT = 1000  # 1000ms timeout for external clock locking

_discovery_cache: Dict[str, dict] = {}
_discovery_cache_ts = 0.0


def invalidate_discovery_cache():
    """Clear cached UHD discovery results (e.g. after a device is unplugged)."""
    global _discovery_cache_ts
    _discovery_cache.clear()
    _discovery_cache_ts = 0.0


def update_device_firmware():
    pass

def discover_devices(logger=None, use_cache: bool = True):
    """
    Devices discovered using uhd.find contain the following keys -
    - 'type': eg. b200
    - 'name': eg. MyB210
    - 'serial'
    - 'product': 'B210'

    These props are wrapped into an appropriate payload
    """
    global _discovery_cache_ts

    if use_cache and _discovery_cache and (
        time.monotonic() - _discovery_cache_ts
    ) < DISCOVERY_CACHE_TTL:
        log_print(logger, "debug", "Using cached USRP discovery results")
        return dict(_discovery_cache)

    discovered_devices = {}

    try:
        log_print(logger, "info", "Searching for USRP devices (uhd.find)...")
        device_list = uhd.find("")

        for device in device_list:
            device_dict = dict(device)
            device_id = device_dict.get("name", "invalid_usrp_device")
            discovered_devices[device_id] = device_dict

            update_usrp_address(device_dict["name"], device_dict["serial"])

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


def _check_pairing(r_idx, t_idx, rx_cumsum, tx_cumsum, pair_list):
    fn = lambda x, y: (np.where(x - y < 0))[0][0]  # noqa: E731
    rx_dev = fn(r_idx, rx_cumsum)
    tx_dev = fn(t_idx, tx_cumsum)

    return (
        ((rx_dev, tx_dev) in pair_list)
        or ((tx_dev, rx_dev) in pair_list)
        or rx_dev == tx_dev
    )


def get_channel_map(
    group_id,
    n_devices: int,
    rx_per_dev: List,
    tx_per_dev: List,
    balance: bool = False,
    multi_pairs: List = None,
):
    """
    Provide base implementations of channel mappings for the following use-cases
    [1] MIMO
    [2] DPIC
    [3] Multi-Frequency
    These two modifications are on top of the multi-band pairing
    """
    data_sources = set()

    rx_cumsum = np.cumsum(rx_per_dev)
    tx_cumsum = np.cumsum(tx_per_dev)

    num_rxs = rx_cumsum[-1]
    num_txs = tx_cumsum[-1]

    if balance:
        rx_enabled = [r % 2 == 0 for r in range(2 * n_devices)]
        tx_enabled = [r % 2 == 0 for r in range(2 * n_devices)]
    else:
        rx_enabled = [True for _ in range(num_rxs)]
        tx_enabled = [True for _ in range(num_txs)]

    rx_ctr = 1
    ch_ctr = 0

    for r_idx, rx_state in enumerate(rx_enabled):
        if not rx_state:
            continue

        tx_ctr = 1
        for t_idx, tx_state in enumerate(tx_enabled):
            if not tx_state:
                continue

            if multi_pairs is None or _check_pairing(
                r_idx, t_idx, rx_cumsum, tx_cumsum, multi_pairs
            ):
                label = f"Tx{tx_ctr}Rx{rx_ctr}"
                source = DataSource(
                    group_id=group_id, channel=ch_ctr, label=label
                )
                source.tx_idx = t_idx
                source.rx_idx = r_idx
                data_sources.add(source)
                ch_ctr += 1

            tx_ctr += 1

        rx_ctr += 1

    return data_sources


def setup_pps(usrp, pps, num_mboards, logger = None):
    """Setup the PPS source."""
    if pps == "mimo":
        if num_mboards != 2:
            log_print(logger, "error", f'ref = "mimo" implies 2 motherboards; your system has {num_mboards} boards')
            return False
        # make mboard 1 a slave over the MIMO Cable
        usrp.set_time_source("mimo", 1)
    else:
        usrp.set_time_source(pps)
    return True


def setup_ref(usrp, ref, num_mboards, logger = None):
    """Setup the reference clock."""
    if ref == "mimo":
        if num_mboards != 2:
            log_print(logger, "error", f'ref = "mimo" implies 2 motherboards; your system has {num_mboards} boards')
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
                log_print(logger, "error", f"Unable to confirm clock signal locked on board {i}")
                return False
    return True


def check_channels(usrp, rx_channels, tx_channels, logger = None):
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


def get_usrp_address(device_name: str, logger = None):
    cache_file = get_cache_file("usrp_serial_numbers")
    map_dict = {}

    try:
        with open(cache_file) as fobj:
            map_dict = json.load(fobj)
    except Exception:
        log_print(logger, "error", "Cache is empty")
        return None

    return map_dict[device_name]


def update_usrp_address(device_name: str, device_serial: str, logger = None):
    cache_file = get_cache_file("usrp_serial_numbers")
    map_dict = {}

    with contextlib.suppress(Exception):
        with open(cache_file) as fobj:
            map_dict = json.load(fobj)

    map_dict[device_name] = device_serial

    # Update
    try:
        with open(cache_file, "w") as fobj:
            json.dump(map_dict, fobj)
    except Exception as e:
        log_print(logger, "error", f"Error updating cache: {e}")
