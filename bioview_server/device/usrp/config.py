from bioview_common import Configuration


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


class USRPConfiguration(Configuration):
    def __init__(
        self,
        if_freq: list,
        tx_gain: list,
        rx_gain: list,
        carrier_freq: int,
        if_filter_bw=BASE_USRP_CONFIG["if_filter_bw"],
        tx_amplitude=BASE_USRP_CONFIG["tx_amplitude"],
        tx_channels=BASE_USRP_CONFIG["tx_channels"],
        rx_channels=BASE_USRP_CONFIG["rx_channels"],
        tx_subdev=BASE_USRP_CONFIG["tx_subdev"],
        rx_subdev=BASE_USRP_CONFIG["rx_subdev"],
        cpu_format=BASE_USRP_CONFIG["cpu_format"],
        wire_format=BASE_USRP_CONFIG["wire_format"],
        clock=BASE_USRP_CONFIG["clock"],
        pps=BASE_USRP_CONFIG["pps"],
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add inputs
        self.if_freq = if_freq
        self.rx_gain = rx_gain
        self.tx_gain = tx_gain
        self.carrier_freq = carrier_freq

        # Default values
        self.if_filter_bw = if_filter_bw

        # Add basic configuration
        self.tx_amplitude = tx_amplitude
        self.rx_channels = rx_channels
        self.tx_channels = tx_channels
        self.rx_subdev = rx_subdev
        self.tx_subdev = tx_subdev
        self.cpu_format = cpu_format
        self.wire_format = wire_format
        self.clock = clock
        self.pps = pps

        # Set-up default absolute channel mapping, assuming single device.
        # This assumes that Tx/Rx are always used in pairs
        # This must be updated if using MIMO with multiple USRPs
        self.absolute_channel_nums = self.tx_channels

    def get_filter_bw(self):
        if not isinstance(self.if_filter_bw, (list, tuple)):
            return [self.if_filter_bw for _ in self.tx_channels]
        elif len(self.if_filter_bw) == len(self.tx_channels):
            return self.if_filter_bw
