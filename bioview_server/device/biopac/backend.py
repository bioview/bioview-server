import queue
import multiprocessing as mp
from ctypes import byref, c_double

from bioview_common import DataSource, DeviceStatus, log_print

from bioview_server.common import SaveWorker
from bioview_server.datatypes import Backend

from .utils import (
    configure_biopac_device,
    connect_biopac_device,
    load_mpdev_dll,
    start_acquisition,
    stop_acquisition,
)


class BIOPACBackend(Backend):
    def __init__(
        self,
        group_id,
        samp_rate: int,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None, 
        mpdev_path: str = None,  # Does not need to be specified in general
        device_code: int = 103  # Default for MP36
    ):
        super().__init__(
            group_id=group_id,
            response_queue=response_queue,
            data_output_queue=data_output_queue
        )
        # Store common variables
        self.samp_rate = samp_rate
        self.device_code = device_code

        # Track mpdev.dll handler
        self.mpdev_handler = load_mpdev_dll(mpdev_path)

        # Populate data sources
        self.populate_data_sources()

    def populate_data_sources(self):
        # Generate channel labels:data queue index mapping
        # alongwith absolute channel numbers
        for idx, _ in enumerate(self.config.channels):
            label = f"Ch{idx + 1}"
            source = DataSource(device=self, channel=idx, label=label)
            self.data_sources.add(source)

            self.absolute_channel_nums[idx] = idx

    # Essential functions for backend
    def _initialize(self):
        if self.mpdev_handler is None:
            raise ValueError("A valid reference to mpdev.dll was not found")

        try:
            connect_biopac_device(
                mpdev_handler=self.mpdev_handler, device_code=self.device_code
            )

            configure_biopac_device(
                mpdev_handler=self.mpdev_handler,
                channels=self.channels,
                sample_rate=self.samp_rate,
            )

            self.status = DeviceStatus.CONNECTED
            log_print(self.logger, "debug", "Successfully initialized BIOPAC device")
            return True 
        except Exception as e:
            log_print(self.logger, "error", f"Unable to initialize device: {e}")
            return False

    def _start_streaming(self):
        self.status = DeviceStatus.STREAMING

        # Start saving
        if self.save_worker:
            if self.save_worker._started.is_set():
                self.save_worker.running = True 
            else:
                self.save_worker.start() 

        # Start display
        if self.display_worker is not None:
            if self.display_worker._started.is_set():
                self.display_worker.running = True 
            else:
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

                    self.display_queue.put(sample)

        except queue.Full:
            log_print(self.logger, "warning", "Queues full. Data not being consumed correctly.")
        except Exception as e:
            log_print(self.logger, "error", f"BIOPAC Streaming Error: {e}")

    def _stop_streaming(self):
        try:
            stop_acquisition(self.mpdev_handler)
        except Exception as e:
            log_print(self.logger, "error", f"BIOPAC Stopping Error: {e}")

        self.status = DeviceStatus.CONNECTED

    def _disconnect(self):
        if self.save_worker is not None:
            self.save_worker.stop()

        self.status = DeviceStatus.DISCONNECTED
