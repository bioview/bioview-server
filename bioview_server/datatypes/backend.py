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
import queue
import logging 
import contextlib
import multiprocessing as mp

from typing import Dict, List 
from threading import Thread 

from bioview_common import DataSource, DeviceStatus, log_print

from bioview_server.common import SaveWorker

class Backend(mp.Process):
    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None, 
    ):
        super().__init__()
        # Parameters
        self.group_id = group_id
        self.data_sources: Dict[DataSource] = []
        self.display_sources: List[DataSource] = [] 

        # Queues        
        self.save_queue = None     
        self.display_queue = mp.Queue() # Queue for internal data storage

        self.data_output_queue = data_output_queue 
        self.response_queue = response_queue  # Queue for responses to client commands
        
        self.enable_save = False
        
        # Common workers
        self.save_worker = None
        self.display_worker = None 

        # State
        self.status = DeviceStatus.DISCONNECTED

        # Create a new logger 
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
           level=logging.DEBUG, format="%(asctime)s %(name)s: (%(levelname)s) %(message)s",
            datefmt='%m/%d %H:%M:%S'
        )

    # Common setup 
    def setup_saving(self, save_path: str = None): 
        '''
        Sets up workers to save data in a common format
        '''    
        self.enable_save = True
        self.save_path = save_path
        
        if not self.save_queue:
            self.save_queue = mp.Queue()
        else:
            # Flush 
            while not self.save_queue.empty():
                self.save_queue.get_nowait()

        if self.enable_save and self.save_path:
            self.save_worker = SaveWorker(
                save_path=save_path,
                data_queue=self.save_queue,
                num_channels=len(self.data_sources),
                logger = self.logger
            )

        # Any other specific functionality can be implemented by subclasses

    def stop_saving(self): 
        if self.save_worker:
            self.save_worker.stop()
        
        # Flush save queue 
        if self.save_queue and not self.save_queue.empty():
            self.save_queue.get_nowait()

    def setup_display(self): 
        '''
        Sets up workers to save data in a common format
        '''    
        if not self.data_output_queue:
            self.data_output_queue = mp.Queue()
        else:
            # Flush at the start 
            while not self.data_output_queue.empty():
                self.data_output_queue.get_nowait()

        self.display_worker = Thread(
            target = self.display_handler,
            daemon = True
        )

        # Any other specific functionality can be implemented by subclasses

    def stop_display(self):
        if self.display_worker:
            self.display_worker.stop()
        
        # Flush display queue 
        if self.display_queue and not self.display_queue.empty():
            self.display_queue.get_nowait()

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
        '''
        Broadcasts available data sources to server handler which can 
        then choose to enable/disable on a per-device basis, as specified
        by the client handler
        '''
        return self.data_sources
        
    def get_display_sources(self):
        '''
        Get the subset of data sources are currently being used for display
        '''
        return self.display_sources
    
    def add_display_source(self, source_id):
        '''
        On the basis of client requests, provides the handler with a 
        mechanism to register different sources from which to send display 
        information
        '''
        self.display_sources.append(source_id)
    
    def remove_display_source(self, source_id):
        '''
        Removes a display source if specified by the calling handler, typically
        because a client wants to replace it with something else
        '''
        with contextlib.suppress(Exception):
            self.display_sources.remove(source_id) 

    def display_handler(self):
        '''
        The only role of display worker in the server is to keep polling
        for data in the display_queue and add it to the sending queue
        '''
        while self.running:
            if len(self.data_sources) == 0: 
                continue 

            try:
                # Get samples 
                samples = self.display_queue.get()
                buff = {} 

                for source in self.display_sources:
                    buff[source] = samples[source.channel]
                
                self.data_output_queue.put_nowait(buff)
            except queue.Empty: 
                log_print(self.logger, 'debug', 'No data available to send for display')
            except queue.Full:
                log_print(self.logger, 'warning', 'Display queue filled up. Unable to add any more data.')
            except Exception as e: 
                log_print(self.logger, 'error', 'Error occurred: {e}')

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
