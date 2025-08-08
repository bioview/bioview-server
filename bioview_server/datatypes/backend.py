'''
A common spec shared by all device specific Backend subclasses, in order to provide
a semblance of sanity for every future stage of the codebase. While each device 
will have their own specific implementation, every backend is expected to provide a 
shared set of functionality, listed below. 

Queues:
- Display Data Queue: 
- Command Queue: 
- Response Queue: 


Functions: 
1. Device Control
- initialize()
- start_streaming()
- stop_streaming() 
- disconnect()

2. Parameter Handling
- get_param()
- set_param()


Signals: 
- status_changed(STATUS)
- log_event(LOG_LEVEL, LOG_MESSAGE)
- data_ready(DATA)


Variables: 
- id: str
- status: DeviceStatus
- display_sources: List[DataSource]
- enable_save: bool
- save_path: str 
'''

import multiprocessing as mp

from bioview_common import DataSource, DeviceStatus 
from bioview_server.callbacks import device_status_changed, data_ready, log_event

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
        self.status = DeviceStatus.DISCONNECTED
        
        # Signals 
        self.log_event = lambda x, y: log_event(self.response_queue, x, y)
        self.status_changed = lambda x, y = None: device_status_changed(self.response_queue, group_id = self.id, status = x, device_id = y)
        self.data_ready = lambda x, y: data_ready(self.display_data_queue, x, y)     

        # Handle save 
        self.enable_save = enable_save
        self.save_path = save_path
        
    def populate_data_sources(self):
        raise NotImplementedError 

    # Device Control
    def initialize(self): 
        raise NotImplementedError
    
    def start_streaming(self): 
        raise NotImplementedError 

    def stop_streaming(self): 
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError
    
    # Parameter handling 
    def get_param(self, param, default_value):
        try:
            value = getattr(self, param)
        except AttributeError:
            value = default_value
        return value

    def set_param(self, param, value): 
        current_type = type(getattr(self, param, None))
        if current_type is not None:
            setattr(self, param, current_type(value))
        else:
            setattr(self, param, value)