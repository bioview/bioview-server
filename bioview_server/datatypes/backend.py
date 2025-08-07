import multiprocessing as mp

from bioview_common import DataSource, DeviceStatus 

from bioview_server.callbacks import connection_state_changed, data_ready, log_event

class Backend(mp.Process):
    def __init__(
        self,
        id: str, 
        display_data_queue: mp.Queue, 
        command_queue: mp.Queue, 
        response_queue: mp.Queue,
        enable_save: bool = True,
        save_path: str = None
    ):
        super().__init__()
        # Parameters
        self.id = id 
        self.data_sources: list[DataSource] = []

        # Queues 
        self.display_data_queue = display_data_queue # Sends to client
        self.command_queue = command_queue # Gets from client
        self.response_queue = response_queue # Sends to client
        
        # State 
        self.state = DeviceStatus.NOINIT
        
        # Signals 
        self.log_event = lambda x, y: log_event(self.response_queue, x, y)
        self.connection_state_changed = lambda x, y: connection_state_changed(self.response_queue, x, y)
        self.data_ready = lambda x, y: data_ready(self.display_data_queue, x, y)     

        # Handle streaming state
        self.is_streaming = False

        # Handle save 
        self.enable_save = enable_save
        self.save_path = save_path
        
    def populate_data_sources(self):
        raise NotImplementedError 
    
    def initialize(self, config): 
        # Basically, init the device with a config 
        self.data_sources = self.get_data_sources() 
        self.state = DeviceStatus.CONNECTING

        # TODO: Add code for connecting 
        self.state = DeviceStatus.CONNECTED

    def get_device_param(self):
        pass 

    def set_device_param(self): 
        pass

    def start_streaming(self): 
        self.is_streaming = True
        pass 

    def stop_streaming(self): 
        pass 

    def disconnect(self):
        self.device.disconnect()
        self.state = DeviceStatus.DISCONNECTED
    
    # Implement saving, which is usually common for all devices 
    def saving_thread(self): 
        while self.is_streaming: 
            pass 

    # Utility functions 

    # def update_param(self, param, value): 
    #     current_type = type(getattr(self, param, None))
    #     if current_type is not None:
    #         setattr(self, param, current_type(value))
    #     else:
    #         setattr(self, param, value)
        
    # def update_config(self, param, value):
    #     self.config.set_param(param, value) 
            
    # def _on_connect_success(self):
    #     emit_signal(self.connection_state_changed, ConnectionStatus.CONNECTED)

    # def _on_connect_failure(self, msg):
    #     emit_signal(self.log_event, "error", msg)
    #     emit_signal(self.connection_state_changed, ConnectionStatus.DISCONNECTED)