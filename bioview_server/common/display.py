import queue
import multiprocessing as mp
import numpy as np
from typing import Dict 

from bioview_common import DataSource
from bioview_server.utils import apply_filter, get_filter, emit_signal

class DisplayWorker():
    def __init__(
        self,
        display_ds: int, 
        display_filter: Dict, 
        data_queue: mp.Queue,
        cmd_queue: mp.Queue, # To handle display filter changes, for example
        running: bool = False,
        parent = None,
    ):
        super().__init__(parent)
        self.display_ds = display_ds
        
        self.display_filter = get_filter(
            bounds = display_filter['bounds'],
            samp_rate = display_filter['samp_rate'],
            btype = display_filter['btype'],
            ftype = display_filter['ftype'],
        )

        self.data_queue = data_queue
        self.running = running

        # Define signals 
        self.data_ready = None
        self.log_event = None 
        
    def process(self, data):
        # Downsample
        # NOTE: This may be replaced by scipy.decimate()
        processed = data[:: self.display_ds] 

        # Filter
        if self.display_filter is not None:
            processed, _ = apply_filter(processed, self.disp_filter)
        return processed

    def run(self):
        emit_signal(self.log_event, "debug", "Display started")

        while self.running:
            if len(self.display_sources) == 0:
                continue

            try:
                # Load samples
                samples = self.data_queue.get()

                # Only process selected channels
                for source in enumerate(self.display_sources):
                    disp_samples = samples[source.channel, :]
                    processed = self.process(disp_samples)

                    # Add to display queue 
                    emit_signal(self.data_ready, np.array(processed), source)
            except queue.Empty:
                emit_signal(self.log_event, "error", "Queue Empty")
                continue
            except Exception as e:
                emit_signal(self.log_event, "error", f"Display error: {e}")
                continue

        emit_signal(self.log_event, "debug", "Display stopped")

    def stop(self):
        self.running = False
