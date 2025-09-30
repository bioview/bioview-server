"""
A common spec shared by all device specific Backend subclasses, in order to provide
a semblance of sanity for every future stage of the codebase. While each device
will have their own specific implementation, every backend is expected to provide a
shared set of functionality, listed below.

Queues:
- Display Data Queue:
- Response Queue:


Functions:
1. Device Control
- initialize()
- queue_param_update()
- start_streaming()
- stop_streaming()
- disconnect()

2. Parameter Handling
- get_param()
- set_param()


Signals:
- data_ready(DATA)


Properties:
- group_id: str
- status: DeviceStatus
- enable_save: bool
- save_path: str
"""
import contextlib
import multiprocessing as mp

from typing import Dict, List 

from bioview_common import DataSource, DeviceStatus

from bioview_server.common import SaveWorker
from bioview_server.callbacks import data_ready, log_event


class Backend(mp.Process):
    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue
    ):
        super().__init__()
        # Parameters
        self.group_id = group_id
        self.data_sources: Dict[DataSource] = []
        self.display_sources: List[DataSource] = [] 

        # Queues        
        self.save_queue = None     
        self.display_queue = None 
        self.response_queue = response_queue  # Sends to client
        
        self.enable_save = False
            
        # State
        self.status = DeviceStatus.DISCONNECTED

        # Signals
        self.data_ready = lambda x, y: data_ready(self.display_queue, x, y)

    # Common setup 
    def setup_saving(self, save_path: str = None): 
        '''
        Sets up workers to save data in a common format
        '''    
        self.enable_save = True
        self.save_path = save_path
        
        self.save_worker = None
        
        if not self.save_queue:
            self.save_queue = mp.Queue()
        else:
            # Flush 
            while not self.save_queue.empty():
                self.save_queue.get_nowait()

        if self.enable_save and self.save_path is not None:
            self.save_worker = SaveWorker(
                save_path=save_path,
                data_queue=self.save_queue,
                num_channels=len(self.data_sources)
            )

        # Any other specific functionality can be implemented by subclasses

    def stop_saving(self): 
        self.enable_save = False
        self.save_worker.stop()
        
        # Flush save queue 
        while not self.save_queue.empty():
            self.save_queue.get_nowait()

    # Device Control
    def initialize(self):
        raise NotImplementedError

    def queue_param_update(self): 
        ''' 
        Backends that implement this function will be able to handle 
        real-time update of parameters by implementing multiprocessing 
        queues internally
        '''
        raise NotImplementedError
    
    def populate_data_sources(self):
        raise NotImplementedError
    
    def get_data_sources(self):
        return self.data_sources
        
    def get_display_sources(self):
        '''
        Broadcasts available display sources to server handler which can 
        then choose to enable/disable on a per-device basis, as specified
        by the client handler
        '''
        return self.display_sources
    
    def add_display_source(self, display_queue, source_id):
        '''
        On the basis of client requests, provides the handler with a 
        mechanism to register different sources from which to send display 
        information
        '''
        if not self.display_queue and display_queue: 
            self.display_queue = display_queue
            self.display_sources.append(source_id)
    
    def remove_display_source(self, source_id):
        '''
        Removes a display source if specified by the calling handler, typically
        because a client wants to replace it with something else
        '''
        with contextlib.suppress(Exception):
            self.display_sources.remove(source_id) 

    def start_streaming(self):
        raise NotImplementedError

    def stop_streaming(self):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    # Status
    def get_device_status(self):
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
