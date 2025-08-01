import uhd
from threading import Thread

from bioview_server.utils import emit_signal

from .utils import (
    check_channels,
    get_usrp_address,
    setup_pps,
    setup_ref,
    update_usrp_address,
)


class ConnectWorker(Thread):
    def __init__(self, config):
        super().__init__()
        # Signals 
        self.init_succeeded = None 
        self.init_failed = None 
        self.log_event = None 
        
        # Variables 
        self.config = config
        self.usrp = None

    def run(self):
        # Performs device discovery and initialization in its own separate thread.
        try:
            ser = get_usrp_address(self.config.device_name)
            if ser is not None:
                addr = uhd.find(f"serial={ser}")[0]
            else:
                addr = uhd.find(self.config.device_name)[0]
                ser = addr["serial"]
                # Store it for future use
                update_usrp_address(self.config.device_name, ser)

            self.usrp = uhd.usrp.MultiUSRP(f"serial={ser},num_recv_frames=1024")

            # Always select the subdevice first, the channel mapping affects the other settings
            self.usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(self.config.rx_subdev))
            self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(self.config.tx_subdev))

            # Set the reference clock - Return on failure
            if not setup_ref(self.usrp, self.config.clock, self.usrp.get_num_mboards()):
                emit_signal(self.init_failed, "Unable to lock reference clock")
                return
            emit_signal(self.log_event, "debug", "Reference Locked")

            # Set the PPS source - Return on failure
            if not setup_pps(self.usrp, self.config.pps, self.usrp.get_num_mboards()):
                emit_signal(self.init_failed, "Unable to lock timing source")
                return
            emit_signal(self.log_event, "debug", "Timing Source Locked")

            # At this point, we can assume our device has valid and locked clock and PPS
            rx_channels, tx_channels = check_channels(
                self.usrp, self.config.rx_channels, self.config.tx_channels
            )
            if not rx_channels and not tx_channels:
                # If the check returned two empty channel lists, that means something went wrong
                emit_signal(self.init_failed, "Mismatch between channel configuration specified and actual channels available on device")
                return

            emit_signal(self.log_event, "debug", "Channels Validated")

            samp_rate = self.config.samp_rate
            carrier_freq = self.config.carrier_freq

            # Setup Rx channels
            rx_gain = self.config.rx_gain
            for idx, chan in enumerate(rx_channels):
                self.usrp.set_rx_rate(samp_rate, chan)
                self.usrp.set_rx_freq(carrier_freq, chan)
                self.usrp.set_rx_gain(rx_gain[idx], chan)
                self.usrp.set_rx_antenna("RX2", chan)
            emit_signal(self.log_event, "debug", "Rx Channels Configured")

            # Setup Tx channels   v
            tx_gain = self.config.tx_gain
            for idx, chan in enumerate(tx_channels):
                self.usrp.set_tx_rate(samp_rate, chan)
                self.usrp.set_tx_freq(carrier_freq, chan)
                self.usrp.set_tx_gain(tx_gain[idx], chan)
                self.usrp.set_tx_antenna("TX1", chan)
            emit_signal(self.log_event, "debug", "Tx Channels Configured")

            # Setup streamer objects
            stream_args = uhd.usrp.StreamArgs(
                self.config.cpu_format, self.config.get_param("wire_format")
            )

            stream_args.channels = tx_channels
            self.tx_streamer = self.usrp.get_tx_stream(stream_args)

            stream_args.channels = rx_channels
            self.rx_streamer = self.usrp.get_rx_stream(stream_args)

            emit_signal(self.log_event, "debug", f"Connected to USRP: {addr}")
            # Emit success
            emit_signal(self.init_succeeded, self.usrp, self.tx_streamer, self.rx_streamer)
        except Exception as e:
            emit_signal(self.init_failed, f"Unable to initialize device: {e}")
