import queue

from bioview_server.device import Device
from bioview_server.common import SaveWorker, DisplayWorker

from bioview_common import DeviceStatus, DataSource

# Core functionality that should always be available
from .connect import ConnectWorker
from .process import ProcessWorker
from .receive import ReceiveWorker
from .config import BiopacConfiguration

class BiopacDevice(Device):    
    def __init__(
        self,
        device_name,
        config: BiopacConfiguration,
        resp_queue,
        data_queue, 
        save: bool = False,
        save_path=None,
        display=False,
    ):
        super().__init__(
            device_name=device_name,
            config=config,
            resp_queue=resp_queue,
            data_queue=data_queue,
            device_type="biopac",
            save=save,
            save_path=save_path,
            display=display,
        )
        self.rx_queue = queue.Queue()

        # Workers
        self.connect_worker = ConnectWorker(self.config)
        self.connect_worker.init_succeeded = self._on_connect_succeess
        self.connect_worker.init_failed = self._on_connect_failure
        self.connect_worker.log_event = self.log_event

        self.receive_worker = None 
        
        # Save/Display Workers
        self.num_channels = len(self.config.channels)
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
            
        if self.display and self.display_sources is not None:
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
        # Generate channel labels:data queue index mapping alongwith absolute channel numbers
        for idx, _ in enumerate(self.config.channels):
            label = f"Ch{idx + 1}"
            source = DataSource(device=self, channel=idx, label=label)
            self.data_sources.append(source)

            self.config.absolute_channel_nums[idx] = idx

    def connect(self):
        self.connect_worker.run()

    def run(self):
        if self.receive_worker is not None:
            self.receive_worker.run() 
        if self.save_worker is not None: 
            self.save_worker.run()
        if self.display_worker is not None: 
            self.display_worker.run()

    def stop(self):
        if self.receive_worker is not None:
            self.receive_worker.stop() 
        if self.save_worker is not None: 
            self.save_worker.stop()
        if self.display_worker is not None: 
            self.display_worker.stop()

    def _on_connect_succeess(self, biopac):
        self.handler = biopac
        
        if self.handler is None:
            self.log_event("error", "No BIOPAC object found")
            self.connection_state_changed(DeviceStatus.DISCONNECTED)
            return

        # Start receiving
        self.receive_worker = ReceiveWorker(
            biopac=self.handler, config=self.config, rx_queue=self.rx_queue
        )
        self.receive_worker.log_event = self.log_event
        
        # Update status bar
        self.connection_state_changed(DeviceStatus.CONNECTED)
