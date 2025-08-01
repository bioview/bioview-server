CLOCK_TIMEOUT = 1000  # 1000ms timeout for external clock locking
INIT_DELAY = 0.05  # 50mS initial delay before transmit
SETTLING_TIME = 0.3
FILLING_TIME = 0.35

SAVE_BUFFER_SIZE = 20  # This is a good balance between real time display and spikes

"""
We make some general assumptions, specifically -
* Each device has two working channels
* Each device uses the default data formats
* Each device uses internal timing reference and clock
* Each device sends waveforms of amplitude 1
"""
BASE_USRP_CONFIG = {
    "tx_amplitude": [1, 1],
    "rx_channels": [0, 1],
    "tx_channels": [0, 1],
    "rx_subdev": "A:A A:B",
    "tx_subdev": "A:A A:B",
    "cpu_format": "fc32",
    "wire_format": "sc16",
    "clock": "internal",
    "pps": "internal",
    "if_filter_bw": 5e3,
    "save_ds": 100,
    "disp_ds": 10,
}
