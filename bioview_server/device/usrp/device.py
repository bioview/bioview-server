import queue

from bioview_server.device import Device
from bioview_server.common import DisplayWorker, SaveWorker
from bioview_server.constants import ConnectionStatus
from bioview_server.utils import emit_signal

from .config import MultiUsrpConfiguration, UsrpConfiguration
from .connect import ConnectWorker
from .process import ProcessWorker
from .receive import ReceiveWorker
from .transmit import TransmitWorker


class MultiUsrpDevice(Device):
    def __init__(
        self,
        device_name,
        config: MultiUsrpConfiguration,
        resp_queue, 
        data_queue, 
        save=False,
        save_path=None,
        display=True,
    ):
        super().__init__(
            device_name=device_name,
            config=config,
            resp_queue=resp_queue,
            data_queue=data_queue,
            device_type="multi_usrp",
            save=save,
            save_path=save_path,
            display=display,
        )

        self.handler = {}
        self.state = {}

        self.if_filter_bw = []

        for dev_name, dev_cfg in config.devices.items():
            dev_handler = UsrpDeviceWrapper(device_name=dev_name, config=dev_cfg)
            
            # Connect callbacks 
            dev_handler.log_event = self.log_event
            dev_handler.connection_state_changed = self.connection_state_changed

            # Initialize state 
            self.handler[dev_name] = dev_handler
            self.state[dev_name] = ConnectionStatus.DISCONNECTED
            
            # Update filter BW
            self.if_filter_bw.extend(dev_cfg.get_filter_bw())
        
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
        
        if self.config.get_param("save_phase", True):
            self.num_channels = 2 * len(self.data_sources)
        else:
            self.num_channels = len(self.data_sources)

        if self.save and self.save_path is not None:
            self.save_worker = SaveWorker(
                save_path=self.save_path,
                data_queue=self.save_queue,
                num_channels=self.num_channels,
                running=True,
            )
            self.save_worker.log_event = self.log_event
        else:
            self.save_worker = None 

        # Create display thread
        if self.display:
            self.display_worker = DisplayWorker(
                config=self.config,
                data_queue=self.display_queue,
                running=True,
            )
            self.display_worker.data_ready = self.data_ready
            self.display_worker.log_event = self.log_event
        else:
            self.display_worker = None 

    def _populate_data_sources(self):
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
        num_usrp_devices = len(self.config.devices)

        # Generate absolute Tx/Rx mapping across all USRP devices
        counter = 0
        for dev_cfg in self.config.devices.values():
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
        for cfg in self.config.devices.values():
            for idx, abs_idx in enumerate(cfg.absolute_channel_nums):
                channel_ifs[abs_idx] = cfg.if_freq[idx]
        self.channel_ifs = channel_ifs

    def _on_state_update(self, device, new_state):
        self.state[device] = new_state

        inited = True
        for _, dev_state in self.state.items():
            if dev_state != ConnectionStatus.CONNECTED:
                inited = False
                break

        if inited:
            emit_signal(self.connection_state_changed, ConnectionStatus.CONNECTED)

    def connect(self):
        for handler in self.handler.values():
            handler.connect()

    def run(self):
        # Start transceiving
        for handler in self.handler.values():
            handler.run()

        # Start processing
        self.process_worker.start()
        
        # Start saving
        if self.save_worker is not None:
            self.save_worker.start() 
        
        # Start display             
        if self.display_worker is not None:
            self.display_worker.start()

    def stop(self):
        # Stop transceiving
        for handler in self.handler.values():
            handler.stop()

        # Stop threads 
        self.process_worker.stop() 
        
        if self.save_worker is not None:
            self.save_worker.stop() 
        
        if self.display_worker is not None:
            self.display_worker.stop()

class UsrpDeviceWrapper:
    def __init__(self, device_name, config: UsrpConfiguration):
        super().__init__()
        # Signals 
        self.log_event = None 
        self.connection_state_changed = None
        
        # Variables 
        self.device_name = device_name
        self.config = config
        self.rx_queue = queue.Queue()
        
        # Connect Worker
        self.connect_worker = ConnectWorker(self.config)
        self.connect_worker.init_succeeded = self._on_connect_success
        self.connect_worker.init_failed = self._on_connect_failure
        self.connect_worker.log_event = self.log_event
        
        # Tx/Rx workers 
        self.transmit_worker = None 
        self.receive_worker = None 

    def connect(self):
        # Let frontend know we are connecting
        emit_signal(self.connection_state_changed, ConnectionStatus.CONNECTING)
        self.connect_worker.daemon = True 
        # Start thread
        self.connect_worker.start()

    def run(self):
        if self.transmit_worker is None or self.receive_worker is None: 
            return 
        
        # Start streaming
        self.transmit_worker.start()
        self.receive_worker.start()         

    def stop(self):
        if self.transmit_worker is not None: 
            self.transmit_worker.stop()
        if self.receive_worker is not None: 
            self.receive_worker.stop()

    def balance_gains(self):
        pass

    def sweep_frequency(self):
        pass

    def _on_connect_success(self, usrp, tx_streamer, rx_streamer, idx=0):
        if usrp is None:
            emit_signal(self.log_event, "error", "No USRP object found")
            return

        if tx_streamer is None:
            emit_signal(self.log_event, "error", "No USRP Tx streamer found")
            return

        if rx_streamer is None:
            emit_signal(self.log_event, "error", "No USRP Rx streamer found")
            return
        
        self.handler = usrp
        self.tx_streamer = tx_streamer
        self.rx_streamer = rx_streamer

        # Make Tx/Rx workers
        self.transmit_worker = TransmitWorker(config=self.config, 
                                              usrp=self.handler, 
                                              tx_streamer=self.tx_streamer)
        self.transmit_worker.log_event = self.log_event
        
        self.receive_worker = ReceiveWorker(
            usrp=self.handler,
            config=self.config,
            rx_streamer=self.rx_streamer,
            rx_queue=self.rx_queue,
            running=True,
        )
        self.receive_worker.log_event = self.log_event
        
        # Update status bar
        emit_signal(self.connection_state_changed, ConnectionStatus.CONNECTED)

    def _on_connect_failure(self, msg): 
        emit_signal(self.log_event, "error", msg)
        emit_signal(self.connection_state_changed, ConnectionStatus.DISCONNECTED)