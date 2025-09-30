import multiprocessing as mp
import queue
from typing import Dict, List

import uhd
from bioview_common import log_print, DeviceStatus, Configuration, DeviceType

from bioview_server.common import DisplayWorker
from bioview_server.datatypes import Backend

from .process import ProcessWorker
from .receive import ReceiveWorker
from .transmit import TransmitWorker
from .utils import (
    check_channels,
    discover_devices,
    get_channel_map,
    setup_pps,
    setup_ref,
)


SETTLING_TIME = 0.3
FILLING_TIME = 0.35
# This is a good balance between real time display and spikes
SAVE_BUFFER_SIZE = 20


def initialize_usrp_device(
    serial,
    tx_subdev,
    rx_subdev,
    clock,
    pps,
    rx_channels,
    tx_channels,
    samp_rate,
    carrier_freq,
    rx_gain,
    tx_gain,
    cpu_format,
    wire_format,
    logger = None
):
    usrp = uhd.usrp.MultiUSRP(f"serial={serial},num_recv_frames=1024")

    # Always select the subdevice first, the channel mapping affects the other settings
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(rx_subdev))
    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(tx_subdev))

    # Set the reference clock - Return on failure
    if not setup_ref(usrp, clock, usrp.get_num_mboards()):
        log_print(logger, "error", "Unable to lock reference clock")
        return None

    log_print(logger, "debug", "Reference Locked")

    # Set the PPS source - Return on failure
    if not setup_pps(usrp, pps, usrp.get_num_mboards()):
        log_print(logger, "error", "Unable to lock timing source")
        return None

    log_print(logger, "debug", "Timing Source Locked")

    # At this point, we can assume our device has valid and locked clock and PPS
    rx_channels, tx_channels = check_channels(usrp, rx_channels, tx_channels)
    if not rx_channels and not tx_channels:
        # If the check returned two empty channel lists, that means something went wrong
        log_print(
            logger, "error",
            "Mismatch between specified channels and available channels",
        )
        return None

    log_print(logger, "debug", "Channels Validated")

    # Setup Rx channels
    for idx, chan in enumerate(rx_channels):
        usrp.set_rx_rate(samp_rate, chan)
        usrp.set_rx_freq(carrier_freq, chan)
        usrp.set_rx_gain(rx_gain[idx], chan)
        usrp.set_rx_antenna("RX2", chan)
    log_print(logger, "debug", "Rx Channels Configured")

    # Setup Tx channels
    for idx, chan in enumerate(tx_channels):
        usrp.set_tx_rate(samp_rate, chan)
        usrp.set_tx_freq(carrier_freq, chan)
        usrp.set_tx_gain(tx_gain[idx], chan)
        usrp.set_tx_antenna("TX1", chan)
    log_print(logger, "debug", "Tx Channels Configured")

    # Setup streamer objects
    stream_args = uhd.usrp.StreamArgs(cpu_format, wire_format)

    stream_args.channels = tx_channels
    tx_streamer = usrp.get_tx_stream(stream_args)

    stream_args.channels = rx_channels
    rx_streamer = usrp.get_rx_stream(stream_args)

    # Emit success
    return {"usrp": usrp, "tx_streamer": tx_streamer, "rx_streamer": rx_streamer}

# TODO: Command queue is only needed for param update. 
# Move it internally by adding a queue_update_param() in Backend

class USRPBackend(Backend):
    def __init__(
        self,
        group_id: str,
        samp_rate: int,
        devices: Dict,
        response_queue: mp.Queue,
        display_ds: int = 10,
        display_imaginary: bool = False,
        discovered_devices: List = None,
        logger = None 
    ):
        super().__init__(
            group_id=group_id, 
            response_queue=response_queue
        )
        # Store common variables
        self.samp_rate = samp_rate

        self.rx_data_queue = {}  # Need to keep this internal

        # Configuration for processing data to be displayed
        self.display_ds = display_ds
        self.display_imaginary = display_imaginary

        # Populate variables pertaining to device
        self.usrp_configs = {}
        self.usrp_handlers = {}
        self.usrp_states = {}

        # Create helper workers and queues
        self.transmit_workers = {}
        self.tx_command_queue = {}
        self.receive_workers = {}
        # Get decoded instructions from overall command queue
        self.rx_command_queue = {}
        self.display_command_queue = None

        self.logger = logger

        self.discovered_devices = discovered_devices 

        # A dict of dict based access is used instead of list of dict
        # simply because it offers a greater degree of convenience while
        # maintaining ordering (as of Python 3.7+)
        self.channel_ifs = []
        self.if_filter_bw = []

        for device_key, device_config in devices.items():
            if not isinstance(device_config, dict):
                raise ValueError(
                    f"Expected device configuration to be a dict \
                      but got {type(device_config)} instead"
                )

            cfg = Configuration.from_dict(device_config, DeviceType.USRP.value)
            self.usrp_configs[device_key] = cfg
            self.usrp_handlers[device_key] = None
            self.usrp_states[device_key] = DeviceStatus.DISCONNECTED
            self.transmit_workers[device_key] = None
            self.receive_workers[device_key] = None

            # Load channel IFs and IF_Filter_BW
            self.channel_ifs.extend(cfg.get_param("if_freq"))
            self.if_filter_bw.extend(cfg.get_filter_bw())

        # Populate data source mapping for client
        self.populate_data_sources()

        # Setup display
        self.display_queue = queue.Queue()

        self.display_worker = DisplayWorker(
            samp_rate=self.samp_rate,
            display_sources=self.display_sources,
            display_ds=display_ds,
            display_filter={},
            data_queue=self.display_queue,  # Gets data from ProcessWorker
            cmd_queue=self.display_command_queue,
            data_ready=self.data_ready,  # Puts data into shared display_data_queue
        )

        # Setup processing
        self.process_worker = ProcessWorker(
            data_sources = self.data_sources, 
            samp_rate = self.samp_rate,
            channel_ifs = self.channel_ifs,
            if_filter_bw = self.if_filter_bw,
            rx_queues = self.rx_data_queue,
            display_queue = self.display_queue,
            display_imaginary = self.display_imaginary,
        )

    def initialize(self):
        if self.discovered_devices is None:
            self.discovered_devices = discover_devices()

        for device_key, device_config in self.usrp_configs.items():
            device_serial = ""
            discovered_names = [v['name'] for v in self.discovered_devices]
            
            if device_config.device_name not in discovered_names:
                msg = f"Device {device_config.device_name} not connected to PC.",
                log_print(self.logger, "error", msg)
                return

            idx = discovered_names.index(device_config.device_name)
            device_serial = self.discovered_devices[idx]["serial"]

            try:
                rx_gain = device_config.get_param("rx_gain")
                tx_gain = device_config.get_param("tx_gain")

                rx_channels = device_config.get_param("rx_channels")
                tx_channels = device_config.get_param("tx_channels")

                response = initialize_usrp_device(
                    serial=device_serial,
                    rx_subdev=device_config.get_param("rx_subdev"),
                    tx_subdev=device_config.get_param("tx_subdev"),
                    clock=device_config.get_param("clock"),
                    pps=device_config.get_param("pps"),
                    rx_channels=rx_channels,
                    tx_channels=tx_channels,
                    samp_rate=device_config.get_param("samp_rate"),
                    carrier_freq=device_config.get_param("carrier_freq"),
                    rx_gain=rx_gain,
                    tx_gain=tx_gain,
                    cpu_format=device_config.get_param("cpu_format"),
                    wire_format=device_config.get_param("wire_format"),
                )

                if not response:
                    self.usrp_states[device_key] = DeviceStatus.DISCONNECTED
                    log_print(self.logger, "error", "Unable to initialize device")

                    return False
                else:
                    # Save objects
                    self.usrp_handlers[device_key] = response["usrp"]

                    self.rx_data_queue[device_key] = queue.Queue()
                    self.rx_command_queue[device_key] = queue.Queue()

                    self.receive_workers[device_key] = ReceiveWorker(
                        usrp=response["usrp"],
                        rx_gain=rx_gain,
                        rx_channels=rx_channels,
                        rx_streamer=response["rx_streamer"],
                        rx_queue=self.rx_data_queue[device_key],
                        cmd_queue=self.rx_command_queue[device_key],
                    )

                    self.tx_command_queue[device_key] = queue.Queue()
                    self.transmit_workers[device_key] = TransmitWorker(
                        usrp=response["usrp"],
                        tx_gain=tx_gain,
                        tx_amplitude=device_config.get("tx_amplitude"),
                        tx_channels=tx_channels,
                        samp_rate=self.samp_rate,
                        if_freq=device_config.get("if_freq"),
                        tx_streamer=response["tx_streamer"],
                        cmd_queue=self.tx_command_queue[device_key],
                    )

                    # Save state
                    self.usrp_states[device_key] = DeviceStatus.CONNECTED
                
                    return True 
            except Exception as e:
                log_print(self.logger, "error", f"Unable to initialize device: {e}")
                return False

    def setup_saving(
        self, 
        enable_save: bool = False, 
        save_path: str = None,
        save_ds: int = 100,
        save_iq: bool = True,
        save_imaginary: bool = True,
    ):        
        super().setup_saving(enable_save, save_path)

        self.save_ds = save_ds
        self.save_iq = save_iq
        self.save_imaginary = save_imaginary

        # Provide params to ProcessWorker
        self.process_worker.save_imaginary = save_imaginary
        self.process_worker.save_iq = save_iq
        self.process_worker.save_ds = save_ds
        self.process_worker.save_queue = self.save_queue

    def start_streaming(self):
        # Start transmit threads
        for worker in self.transmit_workers.values():
            worker.start()

        # Start receive threads
        for worker in self.transmit_workers.values():
            worker.stop()

        # Start saving
        self.process_worker.start() 

        if self.save_worker is not None:
            self.save_worker.start()

        # Start display
        if self.display_worker is not None:
            self.display_worker.start()

    def stop_streaming(self):
        if self.save_worker is not None:
            self.save_worker.stop()

        if self.display_worker is not None:
            self.display_worker.stop()

        self.process_worker.stop() 

        # Stop transmit threads
        for worker in self.transmit_workers.values():
            worker.stop()

        # Stop receive threads
        for worker in self.transmit_workers.values():
            worker.stop()

    def populate_data_sources(self):
        """
        We can arrange multiple USRPs in a variety of configurations, including -
        1. MIMO
        2. Multi-Frequency Band
        3. DPIC
        The above are supported configurations and the list may keep growing.
        In each case, our data source needs to know about what Tx and Rx
        (absolute indices) it is using. Hence, the code below does the following -
        1. From relative Tx/Rx mapping (which is what the API spec gives us), we
        create absolute Tx/Rx mapping
        2. We generate channel mapping and ensure each data source knows its sources
        """
        num_usrp_devices = len(self.usrp_configs)

        # Generate absolute Tx/Rx mapping across all USRP devices
        counter = 0
        for dev_cfg in self.usrp_configs.values():
            dev_cfg.absolute_channel_nums = [
                counter + val for val in dev_cfg.rx_channels
            ]
            counter += len(dev_cfg.rx_channels)

        # Generate sources with mapping
        rx_per_usrp = [len(x.rx_channels) for x in self.usrp_configs.values()]
        tx_per_usrp = [len(x.tx_channels) for x in self.usrp_configs.values()]

        self.data_sources = get_channel_map(
            group_id=self.group_id,
            n_devices=num_usrp_devices,
            rx_per_dev=rx_per_usrp,
            tx_per_dev=tx_per_usrp,
            balance=getattr(self, "balance", False),
            multi_pairs=getattr(self, "multi_pairs", None),
        )

        # Populate list of all channel frequencies - Number of Tx/Rx
        channel_ifs = [None] * sum(rx_per_usrp)
        for dev_cfg in self.usrp_configs.values():
            for idx, abs_idx in enumerate(dev_cfg.absolute_channel_nums):
                channel_ifs[abs_idx] = dev_cfg.if_freq[idx]
        self.channel_ifs = channel_ifs

    def disconnect(self):
        # Stop streaming
        self.stop_streaming()

        # Disconnect
        for device_key in self.usrp_handlers:
            # Close all object handlers
            self.usrp_handlers[device_key] = None
            self.usrp_states[device_key] = DeviceStatus.DISCONNECTED

            self.transmit_workers[device_key] = None
            self.receive_workers[device_key] = None

            # Close all device queues
            self.rx_data_queue = None
            self.rx_command_queue = None

        # Clear common queues
        self.display_queue.clear()
        self.save_queue.clear()

        # Inform UI
        return True 
