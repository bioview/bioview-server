import queue
from ctypes import byref, c_double
from threading import Thread

from .config import BiopacConfiguration
from .utils import wrap_result_code

class ReceiveWorker(Thread):
    def __init__(
        self,
        biopac,
        config: BiopacConfiguration,
        rx_queue: queue.Queue,
        running: bool = True
    ):
        super().__init__()
        # Signals
        self.log_event = None 
        
        # Variables
        self.config = config
        self.biopac = biopac
        self.rx_queue = rx_queue
        self.running = running

    def run(self):
        try:
            # Start acquisition
            wrap_result_code(self.biopac.startAcquisition())
            num_channels = len(self.config.channels)
            data_buffer = (c_double * (num_channels + 1))()  # +1 for timestamp

            while self.running:
                # Get recent sample and add to queue
                if wrap_result_code(
                    self.biopac.getMostRecentSample(byref(data_buffer))
                ):
                    sample = [data_buffer[i] for i in range(num_channels + 1)]
                    self.rx_queue.put(sample)

        except Exception as e:
            self.log_event("error", e)

    def stop(self):
        wrap_result_code(self.mpdev.stopAcquisition())
        self.running = False
