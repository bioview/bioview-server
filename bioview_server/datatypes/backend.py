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

from bioview_common import Command, DataSource, DeviceStatus, log_print

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
        self.command_queue = mp.Queue() 
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
        self._running = mp.Event()
        self._streaming = mp.Event()

    # Internal Implementations 
    # Common setup 
    def _setup_saving(self, save_config: Dict = None): 
        '''
        Sets up workers to save data in a common format
        '''    
        self.enable_save = save_config.get('enable_save', False)
        self.save_path = save_config.get('save_path', None)
        
        if not self.save_queue:
            self.save_queue = mp.Queue()
        else:
            # Flush 
            while not self.save_queue.empty():
                self.save_queue.get_nowait()

        if self.enable_save and self.save_path:
            self.save_worker = SaveWorker(
                save_path=self.save_path,
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

    def _setup_display(self, display_config: Dict = None): 
        '''
        Sets up workers to save data in a common format
        '''    
        if not self.data_output_queue:
            self.data_output_queue = mp.Queue()
        else:
            # Flush at the start 
            while not self.data_output_queue.empty():
                self.data_output_queue.get_nowait()

        display_sources = display_config.get('display_sources', [])
        for source in display_sources:
            self.add_display_source(source)

        # TODO: Replace with PausableWorker
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
    def _initialize(self):
        raise NotImplementedError

    def _queue_param_update(self): 
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
        while self._streaming.is_set():
            if len(self.data_sources) == 0: 
                continue 

            try:
                # Get samples 
                samples = self.display_queue.get_nowait()
                buff = {} 

                for source in self.display_sources:
                    buff[source] = samples[source.channel]
                
                self.data_output_queue.put_nowait(buff)
            except queue.Empty: 
                log_print(self.logger, 'debug', 'No data available to send for display')
            except queue.Full:
                log_print(self.logger, 'warning', 'Display queue filled up. Unable to add any more data.')
            except Exception as e:
                log_print(self.logger, 'error', f'Error occurred: {e}')

    def _start_streaming(self):
        raise NotImplementedError

    def _stop_streaming(self):
        raise NotImplementedError

    def _disconnect(self):
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

    # Handle multiprocessing
    def run(self):
        # Create a new logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
           level=logging.DEBUG, format="%(asctime)s %(name)s: (%(levelname)s) %(message)s",
            datefmt='%m/%d %H:%M:%S'
        )

        self._running.set()

        # Start thread to handle display
        while self._running.is_set():
            try:
                # Process commands from parent
                cmd_data = self.command_queue.get(timeout=1)
                self._handle_command(cmd_data)                
            except queue.Empty:
                continue          
            except Exception as e:
                self.logger.error(f"Error in subprocess: {e}")

    def _handle_command(self, data):
        try:
            cmd = data['command']
            cmd_args = data.get('args', {})

            # TODO: Update to put conditionnal checks for responses - !IMPORTANT
            match cmd:
                case Command.CONNECT_DEVICES:
                    result = self._initialize()
                    if not result:
                        raise RuntimeError("Unable to initialize device")
                    self.response_queue.put({'status': 'success', 'result': result})
                
                case Command.START_STREAMING:
                    save_cfg = cmd_args.get('save_config', {})
                    display_cfg = cmd_args.get('display_config', {})

                    if save_cfg:
                        self._setup_saving(save_cfg)
                    
                    if display_cfg: 
                        self._setup_display(display_cfg)
                    
                    result = self._start_streaming()
                    print(result)
                    self.response_queue.put({'status': 'success', 'result': result})
                    self._streaming.set()
                
                case Command.STOP_STREAMING:
                    self._streaming.clear()
                    result = self._stop_streaming()
                    self.response_queue.put({'status': 'success', 'result': result})
                
                case Command.DISCONNECT_DEVICES:
                    result = self._disconnect()
                    self.response_queue.put({'status': 'success', 'result': result})
                
                case Command.UPDATE_RUNNING_PARAMETER:
                    self._queue_param_update(cmd_args)
                    self.response_queue.put({'status': 'success'})
                    
        except Exception as e:
            log_print(self.logger, 'error', f"Command {cmd} failed: {e}")
            self.response_queue.put({'status': 'error', 'message': str(e)})
    
    # Public API for non-blocking calls
    def initialize(self, **kwargs):
        self.command_queue.put({
            'command': Command.CONNECT_DEVICES,
            'args': kwargs
        })
        response = self.response_queue.get(timeout=150)
        return response

    def start_streaming(self, cfg_dict: Dict = None):
        self.command_queue.put({
            'command': Command.START_STREAMING,
            'args': cfg_dict
        })
        print(cfg_dict)
        response = self.response_queue.get(timeout=10)
        return response
    
    def stop_streaming(self):
        self.command_queue.put({'command': Command.STOP_STREAMING})
        response = self.response_queue.get(timeout=10)
        return response
    
    def queue_param_update(self, **params):
        self.command_queue.put({
            'command': Command.UPDATE_RUNNING_PARAMETER,
            'args': params
        })
        # Don't wait for response for real-time updates
        # TODO: Fix
    
    def disconnect(self):
        self.command_queue.put({'command': Command.DISCONNECT_DEVICES})
        response = self.response_queue.get(timeout=5)
        return response
    
    def shutdown(self):
        self.command_queue.put({'command': Command.SHUTDOWN})
        self.join(timeout=5)
        if self.is_alive():
            self.terminate()