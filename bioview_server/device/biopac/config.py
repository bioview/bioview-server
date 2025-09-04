from bioview_common import Configuration


class BiopacConfiguration(Configuration):
    def __init__(
        self,
        device_name: str,
        channels: list,
        samp_rate: int = 1000,
        device_type: str = "MP36",
        mpdev_path: str = None,
    ):
        super().__init__()

        self.device_name = device_name
        self.channels = channels
        self.absolute_channel_nums = channels

        self.samp_rate = samp_rate

        # Essential for long-term generalizability of codebase
        self.device_type = device_type
        # By default, we load from $PATH and $ROOT but a custom path may be provided
        self.mpdev_path = mpdev_path

    def get_channels(self):
        # Since the API expects 16 channels, ensure we always pad
        # to return in the appropriate format
        return self.channels + [0] * (16 - len(self.channels))
