from .defaults import BASE_USRP_CONFIG
from bioview_server.constants import Configuration


class MultiUsrpConfiguration(Configuration):
    """
    Handles an arbitrary number of USRP devices, keeping a clear distinction between the common variables and non-common variables
    """

    def __init__(
        self,
        samp_rate: int,
        devices: list[dict],
        save_ds: int = BASE_USRP_CONFIG["save_ds"],
        save_iq: bool = False,
        save_imaginary: bool = True,
        disp_ds: int = BASE_USRP_CONFIG["disp_ds"],
        disp_imaginary: bool = False,
        display_sources: list = [],
    ):
        super().__init__()
        # Store common configuration values
        self.samp_rate = samp_rate
        self.save_ds = save_ds
        self.save_iq = save_iq
        self.save_imaginary = save_imaginary
        self.disp_ds = disp_ds
        self.disp_imaginary = disp_imaginary
        self.display_sources = display_sources

        # Initialize per-device configuration
        self.devices = {}
        for device in devices:
            if not isinstance(device, dict):
                raise ValueError(
                    f"Expected device configuration to be a dict but got {type(device)} instead"
                )
            self.devices[device["device_name"]] = UsrpConfiguration(**device)

            # Copy necessary common configuration values to all devices
            self.devices[device["device_name"]].samp_rate = samp_rate

    def get_disp_freq(self):
        return self.samp_rate / (self.save_ds * self.disp_ds)

    def get_individual_configs(self):
        return self.devices.values()

class UsrpConfiguration(Configuration):
    def __init__(
        self,
        device_name: str,
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
        self.device_name = device_name
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
        if not (
            isinstance(self.if_filter_bw, list) or isinstance(self.if_filter_bw, tuple)
        ):
            return [self.if_filter_bw for _ in self.tx_channels]
        elif len(self.if_filter_bw) == len(self.tx_channels):
            return self.if_filter_bw
