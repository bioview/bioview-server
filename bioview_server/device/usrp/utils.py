"""
Ref: uhd examples
"""

import time
import json 
from datetime import datetime, timedelta
import numpy as np

from .defaults import CLOCK_TIMEOUT
from bioview_server.constants import DataSource


def _check_pairing(r_idx, t_idx, rx_cumsum, tx_cumsum, pair_list):
    fn = lambda x, y: (np.where(x - y < 0))[0][0]
    rx_dev = fn(r_idx, rx_cumsum)
    tx_dev = fn(t_idx, tx_cumsum)

    return (
        ((rx_dev, tx_dev) in pair_list)
        or ((tx_dev, rx_dev) in pair_list)
        or rx_dev == tx_dev
    )


def get_channel_map(
    device,
    n_devices: int,
    rx_per_dev: list,
    tx_per_dev: list,
    balance: bool = False,
    multi_pairs: list = None,
):
    """
    Provide base implementations of channel mappings for the following use-cases
    [1] MIMO
    [2] DPIC
    [3] Multi-Frequency
    These two modifications are on top of the multi-band pairing
    """
    data_sources = []

    rx_cumsum = np.cumsum(rx_per_dev)
    tx_cumsum = np.cumsum(tx_per_dev)

    num_rxs = rx_cumsum[-1]
    num_txs = tx_cumsum[-1]

    if balance:
        rx_enabled = [True if r % 2 == 0 else False for r in range(2 * n_devices)]
        tx_enabled = [True if r % 2 == 0 else False for r in range(2 * n_devices)]
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
                source = DataSource(
                    device=device, channel=ch_ctr, label=f"Tx{tx_ctr}Rx{rx_ctr}"
                )
                source.tx_idx = t_idx
                source.rx_idx = r_idx
                data_sources.append(source)
                ch_ctr += 1

            tx_ctr += 1

        rx_ctr += 1

    return data_sources


def setup_pps(usrp, pps, num_mboards):
    """Setup the PPS source."""
    if pps == "mimo":
        if num_mboards != 2:
            print(
                'ref = "mimo" implies 2 motherboards; ' "your system has %d boards",
                num_mboards,
            )
            return False
        # make mboard 1 a slave over the MIMO Cable
        usrp.set_time_source("mimo", 1)
    else:
        usrp.set_time_source(pps)
    return True


def setup_ref(usrp, ref, num_mboards):
    """Setup the reference clock."""
    if ref == "mimo":
        if num_mboards != 2:
            print(
                'ref = "mimo" implies 2 motherboards; ' "your system has %d boards",
                num_mboards,
            )
            return False
        usrp.set_clock_source("mimo", 1)
    else:
        usrp.set_clock_source(ref)

    # Lock onto clock signals for all mboards
    if ref != "internal":
        print("Now confirming lock on clock signals...")
        end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)
        for i in range(num_mboards):
            if ref == "mimo" and i == 0:
                continue
            is_locked = usrp.get_mboard_sensor("ref_locked", i)
            while (not is_locked) and (datetime.now() < end_time):
                time.sleep(1e-3)
                is_locked = usrp.get_mboard_sensor("ref_locked", i)
            if not is_locked:
                print("Unable to confirm clock signal locked on board %d", i)
                return False
    return True


def check_channels(usrp, rx_channels, tx_channels):
    """Check that the device has sufficient RX and TX channels available."""
    # Check that each Rx channel specified is less than the number of total number of rx channels
    # the device can support
    dev_rx_channels = usrp.get_rx_num_channels()
    if not all(map((lambda chan: chan < dev_rx_channels), rx_channels)):
        print("Invalid RX channel(s) specified.")
        return [], []

    # Check that each Tx channel specified is less than the number of total number of tx channels
    # the device can support
    dev_tx_channels = usrp.get_tx_num_channels()
    if not all(map((lambda chan: chan < dev_tx_channels), tx_channels)):
        print("Invalid TX channel(s) specified.")
        return [], []

    return rx_channels, tx_channels

def get_usrp_address(device_name: str):
    cache_file = _get_cache_file("serial_maps")
    map_dict = {}

    try:
        with open(cache_file, "r") as fobj:
            map_dict = json.load(fobj)
    except Exception as e:
        print("Cache is empty")
        return None

    return map_dict[device_name]


def update_usrp_address(device_name: str, device_serial: str):
    cache_file = _get_cache_file("serial_maps")
    map_dict = {}

    try:
        with open(cache_file, "r") as fobj:
            map_dict = json.load(fobj)
    except Exception as e:
        print("Cache does not exist currently. Creating new cache file.")

    map_dict[device_name] = device_serial

    # Update
    try:
        with open(cache_file, "w") as fobj:
            json.dump(map_dict, fobj)
    except Exception as e:
        print(f"Error updating cache: {e}")
    finally:
        print("Cache updated successfully")
