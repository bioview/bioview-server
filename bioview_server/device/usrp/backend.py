import uhd 
import multiprocessing as mp
from typing import Dict, List

from bioview_server.datatypes import Backend
from bioview_server.common import DisplayWorker, SaveWorker
from bioview_server.utils import emit_signal

from bioview_common import DeviceStatus, DataSource

from .config import USRPConfiguration
from .process import ProcessWorker
from .receive import ReceiveWorker
from .transmit import TransmitWorker 

from .utils import discover_devices, setup_pps, setup_ref, check_channels, update_usrp_address, get_channel_map

SETTLING_TIME = 0.3
FILLING_TIME = 0.35
SAVE_BUFFER_SIZE = 20  # This is a good balance between real time display and spikes

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
    init_failed, 
    log_event
): 
    usrp = uhd.usrp.MultiUSRP(f"serial={serial},num_recv_frames=1024")

    # Always select the subdevice first, the channel mapping affects the other settings
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(rx_subdev))
    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(tx_subdev))

    # Set the reference clock - Return on failure
    if not setup_ref(usrp, clock, usrp.get_num_mboards()):
        emit_signal(init_failed, "Unable to lock reference clock")
        return None 
    
    emit_signal(log_event, "debug", "Reference Locked")

    # Set the PPS source - Return on failure
    if not setup_pps(usrp, pps, usrp.get_num_mboards()):
        emit_signal(init_failed, "Unable to lock timing source")
        return None 
    
    emit_signal(log_event, "debug", "Timing Source Locked")

    # At this point, we can assume our device has valid and locked clock and PPS
    rx_channels, tx_channels = check_channels(
        usrp, rx_channels, tx_channels
    )
    if not rx_channels and not tx_channels:
        # If the check returned two empty channel lists, that means something went wrong
        emit_signal(init_failed, "Mismatch between channel configuration specified and actual channels available on device")
        return None

    emit_signal(log_event, "debug", "Channels Validated")

    # Setup Rx channels
    for idx, chan in enumerate(rx_channels):
        usrp.set_rx_rate(samp_rate, chan)
        usrp.set_rx_freq(carrier_freq, chan)
        usrp.set_rx_gain(rx_gain[idx], chan)
        usrp.set_rx_antenna("RX2", chan)
    emit_signal(log_event, "debug", "Rx Channels Configured")

    # Setup Tx channels 
    for idx, chan in enumerate(tx_channels):
        usrp.set_tx_rate(samp_rate, chan)
        usrp.set_tx_freq(carrier_freq, chan)
        usrp.set_tx_gain(tx_gain[idx], chan)
        usrp.set_tx_antenna("TX1", chan)
    emit_signal(log_event, "debug", "Tx Channels Configured")

    # Setup streamer objects
    stream_args = uhd.usrp.StreamArgs(
        cpu_format, wire_format
    )

    stream_args.channels = tx_channels
    tx_streamer = usrp.get_tx_stream(stream_args)

    stream_args.channels = rx_channels
    rx_streamer = usrp.get_rx_stream(stream_args)
    
    # Emit success
    return {
        'usrp': usrp, 
        'tx_streamer': tx_streamer, 
        'rx_streamer': rx_streamer
    } 

class USRPBackend(Backend):
    def __init__(
        self,
        id: str,
        samp_rate: int,
        devices: Dict,
        display_sources: List[DataSource],  
        display_data_queue: mp.Queue,
        command_queue: mp.Queue, 
        response_queue: mp.Queue, 
        enable_save: bool = True,
        save_path: str = None,
        save_ds: int = 100,
        save_iq: bool = True,
        save_imaginary: bool = True,
        display_ds: int = 10,
        display_imaginary: bool = False
    ):
        super().__init__(
            id = id,
            display_data_queue = display_data_queue,
            command_queue = command_queue, 
            response_queue = response_queue,
            enable_save = enable_save,
            save_path = save_path
        )
        # Store common variables 
        self.samp_rate = samp_rate
        
        # Configuration for processing data to be saved 
        self.save_ds = save_ds
        self.save_iq = save_iq
        self.save_imaginary = save_imaginary

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
        self.rx_command_queue = {}


        for device_key, device_config in devices.items():
            if not isinstance(device_config, dict):
                raise ValueError(
                    f"Expected device configuration to be a dict but got {type(device_config)} instead"
                )
            
            self.usrp_configs[device_key] = USRPConfiguration.from_dict('USRPConfiguration', device_config)
            self.usrp_handlers[device_key] = None 
            self.usrp_states[device_key] = DeviceStatus.DISCONNECTED
            self.transmit_workers[device_key] = None 
            self.receive_workers[device_key] = None 
        
        # Create data source mapping for client
        self.data_sources = []
        self.populate_data_sources()

        # Saving parameters 
        self.save_worker = None 
        self.save_queue = mp.Queue()

        if self.enable_save and self.save_path is not None:
            self.save_worker = SaveWorker(
                save_path = self.save_path,
                data_queue = self.save_queue,
                num_channels = len(self.data_sources),
                log_event = self.log_event
            )

        # Display parameters
        self.display_worker = None 
        self.display_sources = display_sources

        if self.display:
            self.display_worker = DisplayWorker(
                display_ds = display_ds,
                display_filter = {}, 
                data_queue = self.processed_queue, # Input data 
                display_queue = self.display_data_queue # Processed data to display
            )
            self.display_worker.log_event = self.log_event

        ############ REWORK ################
        # Make workers for saving/display
        self.process_worker = ProcessWorker(
            config=self.config,
            channel_ifs=self.channel_ifs,
            if_filter_bw=self.if_filter_bw,
            data_sources=self.data_sources,
            rx_queues=[x.rx_queue for x in self.handler.values()],
            save_queue=self.save_queue,
            disp_queue=self.display_queue,
            running=True,
        )
        self.process_worker.log_event = self.log_event

    def initialize(self):
        if self.discovered_devices is None: 
            self.discovered_devices = discover_devices()

        for device_key, device_config in self.usrp_handlers.items(): 
            device_serial = ''
            if device_config.device_name not in self.discovered_devices.keys(): 
                emit_signal(self.init_failed, f"Device {device_config.device_name} not connected to PC.")
                return 
            
            device_serial = self.discovered_devices[device_config.device_name]['serial']

            try:
                rx_gain = device_config.get_param('rx_gain')
                tx_gain = device_config.get_param('tx_gain') 
                
                rx_channels = device_config.get_param('rx_channels')
                tx_channels = device_config.get_param('tx_channels')

                response = initialize_usrp_device(
                    serial = device_serial,
                    rx_subdev = device_config.get_param('rx_subdev'),
                    tx_subdev = device_config.get_param('tx_subdev'),
                    clock = device_config.get_param('clock'),
                    pps = device_config.get_param('pps'),
                    rx_channels = rx_channels,
                    tx_channels = tx_channels,
                    samp_rate = device_config.get_param('samp_rate'), 
                    carrier_freq = device_config.get_param('carrier_freq'),
                    rx_gain = rx_gain,
                    tx_gain = tx_gain,
                    cpu_format = device_config.get_param('cpu_format'), 
                    wire_format = device_config.get_param('wire_format') 
                )

                if response is None: 
                    self.usrp_states[device_key] = DeviceStatus.DISCONNECTED
                    emit_signal(self.init_failed, f"Unable to initialize device: {e}")
                else: 
                    # Save objects
                    self.usrp_handlers[device_key] = response['usrp']
                    
                    self.rx_command_queue[device_key] = mp.Queue()
                    self.receive_workers[device_key] = ReceiveWorker(
                        usrp = response['usrp'],
                        rx_gain = rx_gain,
                        rx_channels = rx_channels,
                        rx_streamer = response['rx_streamer'],
                        rx_queue = self.save_data_queue,
                        cmd_queue = self.rx_command_queue,
                        log_event = self.log_event 
                    )

                    self.tx_command_queue[device_key] = mp.Queue()
                    self.transmit_workers[device_key] = TransmitWorker(
                        usrp = response['usrp'], 
                        tx_gain = tx_gain, 
                        tx_amplitude = device_config.get('tx_amplitude'), 
                        tx_channels = tx_channels, 
                        samp_rate = self.samp_rate,
                        if_freq = device_config.get('if_freq'),
                        tx_streamer = response['tx_streamer'],
                        cmd_queue = self.tx_command_queue[device_key],
                        log_event = self.log_event 
                    ) 

                    # Broadcast state change
                    self.status_changed(DeviceStatus.CONNECTED, device_key)
                    
                    # Save state 
                    self.usrp_states[device_key] = DeviceStatus.CONNECTED
            except Exception as e:
                self.usrp_states[device_key] = DeviceStatus.DISCONNECTED
                self.log_event('error', f"Unable to initialize device: {e}")

    def start_streaming(self): 
        # Start transmit threads 
        for worker in self.transmit_workers.values(): 
            worker.start() 

        # Start receive threads 
        for worker in self.transmit_workers.values(): 
            worker.stop() 

        # Start saving
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
        rx_per_usrp = [len(x.rx_channels) for x in self.config.devices.values()]
        tx_per_usrp = [len(x.tx_channels) for x in self.config.devices.values()]
        
        self.data_sources = get_channel_map(
            device=self,
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

    def _on_state_update(self, device, new_state):
        self.state[device] = new_state

        inited = True
        for _, dev_state in self.state.items():
            if dev_state != DeviceStatus.CONNECTED:
                inited = False
                break

        if inited:
            emit_signal(self.connection_state_changed, DeviceStatus.CONNECTED)