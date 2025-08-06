import numpy as np
import multiprocessing as mp

from bioview_common import DataSource, ConnectionStatus, Response 
from bioview_server.constants import Configuration, Message
from bioview_server.utils import emit_signal
from bioview_server.callbacks import connection_state_changed, log_event

""" 
The DeviceHandler class maintains all necessary device info 
& can also read & write data to & from the device 
along with storing the read data in a data dump in storage
"""

class DeviceHandler(mp.Process):
    def __init__(self, config: Configuration, exp_config: Configuration, data_queue: mp.Queue, save):
        # Device configuration
        self.config = config 
        self.exp_config = exp_config
        self.save = save 
         
        self.device_name = config.get_param('device_name', 'dummy_device')
        self.device = None 
        self.data_queue = data_queue
        
        # Device status 
        self.is_connected = False 
        self.is_streaming = False 
        
        self.running = False 
        
    def connect(self):
        # Create device object 
        self.device = get_device_object(
            device_name = self.device_name, 
            config = self.config,
            data_queue = self.data_queue, 
            resp_queue=None, 
            save = self.save,
            exp_config = self.exp_config
        )
        
        self.device.connect()
        
    def update_config(self, param, value): 
        self.device.update_config(param, value)    
    
    def update_param(self, param, value): 
        self.device.update_param(param, value)

class Device(mp.Process):
    def __init__(
        self,
        config: Configuration,
        device_name: str,
        device_type: str,
        response_queue: mp.Queue,
        display_data_queue: mp.Queue = None,
        save: bool = False,
        save_path: str = None
    ):
        super().__init__()
        # Store device-specific constants 
        self.config = config
        self.device_name = device_name 
        self.device_type = device_type
        
        # Track device state 
        self.state = ConnectionStatus.DISCONNECTED

        # Track queues 
        self.response_queue = response_queue # Send responses to server for forwarding
        self.display_data_queue = display_data_queue # Put data for display in client
        
        # Track device handler objects
        self.handler = None

        # Configuration for saving
        self.save = save
        self.save_path = save_path
        if self.save:
            self.save_data_queue = mp.Queue()
        else:
            self.save_data_queue = None

        # Configuration for display
        self.display_data_queue = display_data_queue

        # Signals 
        self.log_event = lambda x, y: log_event(self.response_queue, x, y)
        self.connection_state_changed = lambda x, y: connection_state_changed(self.response_queue, x, y)

        # Keep track of all data sources
        self.data_sources: list[DataSource] = []
        # Make data sources available, depending on config
        self._populate_data_sources()
            
    def get_data_sources(self):
        pass 
    
    def data_ready(self, data: np.ndarray, source: DataSource):
        resp = Message(msg_type=Response.DISPLAY, value=(data, source))
        
        try: 
            self.data_queue.put_nowait(resp)
        except queue.Full: 
            print('Unable to add to data queue as it is full.')
    
    def _populate_data_sources(self):
        raise NotImplementedError  # We expect subclasses to implement this

    def get_disp_freq(self):
        return self.config.get_disp_freq()

    def connect(self):
        self.state = ConnectionStatus.CONNECTED

    def disconnect(self):
        self.device.disconnect()

    def run(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def disconnect(self):
        self.state = ConnectionStatus.DISCONNECTED
    
    def update_param(self, param, value): 
        current_type = type(getattr(self, param, None))
        if current_type is not None:
            setattr(self, param, current_type(value))
        else:
            setattr(self, param, value)
        
    def update_config(self, param, value):
        self.config.set_param(param, value) 
            
    def _on_connect_success(self):
        emit_signal(self.connection_state_changed, ConnectionStatus.CONNECTED)

    def _on_connect_failure(self, msg):
        emit_signal(self.log_event, "error", msg)
        emit_signal(self.connection_state_changed, ConnectionStatus.DISCONNECTED)