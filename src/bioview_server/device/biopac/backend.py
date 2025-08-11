import queue
from ctypes import byref, c_double
import multiprocessing as mp 
from typing import List 

from bioview_server.datatypes import Backend
from bioview_server.common import SaveWorker, DisplayWorker

from bioview_common import DeviceStatus, DataSource

from .utils import connect_biopac_device, configure_biopac_device, start_acquisition, stop_acquisition, load_mpdev_dll

class BIOPACBackend(Backend):    
    def __init__(
        self,
        id,
        samp_rate: int, 
        display_sources: List[DataSource],
        display_data_queue: mp.Queue,
        command_queue: mp.Queue,
        response_queue: mp.Queue, 
        enable_save: bool = False,
        save_path: str = None,
        mpdev_path: str = None, # Does not need to be specified in general
        device_code: int = 103 # Default for MP36
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
        self.device_code = device_code
        
        # Track mpdev.dll handler
        self.mpdev_handler = load_mpdev_dll(mpdev_path)

        # Populate data sources 
        self.populate_data_sources()

        # Saving parameters 
        self.save_worker = None 
        self.save_queue = mp.Queue()

        if self.enable_save and self.save_path is not None:
            self.save_worker = SaveWorker(
                save_path = save_path,
                data_queue = self.save_queue,
                num_channels = len(self.data_sources),
                log_event = self.log_event
            )

        # Display parameters
        # TODO: Run this in a function which is constantly providing new data_sources
        self.display_worker = None 
        self.display_sources = display_sources
        self.display_worker = DisplayWorker(
            config=self.config,
            data_queue=self.display_queue,
            running=True,
        )
        self.display_worker.data_ready = self.data_ready
        self.display_worker.log_event = self.log_event

        # Initialize
        self.initialize()

    def populate_data_sources(self):
        # Generate channel labels:data queue index mapping alongwith absolute channel numbers
        for idx, _ in enumerate(self.config.channels):
            label = f"Ch{idx + 1}"
            source = DataSource(device=self, channel=idx, label=label)
            self.data_sources.append(source)

            self.absolute_channel_nums[idx] = idx

    # Essential functions for backend
    def initialize(self):
        if self.mpdev_handler is None:
            raise ValueError('A valid reference to mpdev.dll was not found')
        
        try:
            connect_biopac_device(
                mpdev_handler = self.mpdev_handler, 
                device_code = self.device_code
            )

            configure_biopac_device(
                mpdev_handler = self.mpdev_handler, 
                channels = self.channels, 
                sample_rate = self.samp_rate 
            )

            self.log_event('debug', "Successfully initialized BIOPAC device")
            self.status_changed(DeviceStatus.CONNECTED) 
        except Exception as e: 
            self.log_event('error', f"Unable to initialize device: {e}")
            self.status_changed(DeviceStatus.DISCONNECTED)
            
    def start_streaming(self):
        self.status = DeviceStatus.STREAMING
        
        # Start saving
        if self.save_worker is not None:
            self.save_worker.start() 
        
        # Start display
        if self.display_worker is not None:
            self.display_worker.start() 

        try: 
            start_acquisition(self.mpdev_handler)

            # Define buffer 
            num_channels = len(self.channels)
            data_buffer = (c_double * (num_channels + 1))()  # +1 for timestamp

            while self.status == DeviceStatus.STREAMING:
                # Get latest sample and add to queue 
                if self.mpdev_handler.getMostRecentSample(byref(data_buffer)) == 1: 
                    sample = [data_buffer[i] for i in range(num_channels + 1)]
                    self.save_queue.put(sample)

                    # TODO: Only add samples from display_sources
                    self.display_data_queue.put(sample)
        
        except queue.Full:
            self.log_event('warning', 'Queues full. Data not being consumed correctly.')
        except Exception as e: 
            self.log_event('error', f'BIOPAC Streaming Error: {e}')
    
    def stop_streaming(self):
        try: 
            stop_acquisition(self.mpdev_handler)
        except Exception as e: 
            self.log_event('error', f'BIOPAC Stopping Error: {e}')
        
        self.status = DeviceStatus.CONNECTED 
    
    def disconnect(self):
        if self.save_worker is not None: 
            self.save_worker.stop()

        self.status = DeviceStatus.DISCONNECTED 