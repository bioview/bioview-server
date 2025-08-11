import queue
import numpy as np
from typing import List, Dict, Callable

from bioview_common import DataSource
from bioview_server.utils import apply_filter, get_filter, emit_signal

MODIFIABLE_PARAMS = [
    'display_ds',
    'display_filter_type',
    'display_filter_f_low',
    'display_filter_f_high'
]

class DisplayWorker():
    def __init__(
        self,
        samp_rate: int, 
        display_ds: int, 
        display_filter: Dict, 
        display_sources: List[DataSource],
        data_queue: queue.Queue, # Data comes in
        cmd_queue: queue.Queue, # To handle display filter changes, for example
        data_ready: Callable, # Data pushed to client
        log_event: Callable,
        parent = None
    ):
        super().__init__(parent)
        # Sources 
        self.display_sources = display_sources

        # Processing 
        self.display_ds = display_ds
        
        self.display_filter = get_filter(
            bounds = display_filter['bounds'],
            samp_rate = samp_rate,
            btype = display_filter['btype'],
            ftype = display_filter['ftype'],
        )
        
        # Queues 
        self.data_queue = data_queue
        self.cmd_queue = cmd_queue

        # State 
        self.running = False
        
        # Signals 
        self.data_ready = data_ready
        self.log_event = log_event
        
    def process(self, data):
        # Downsample
        # NOTE: This may be replaced by scipy.decimate()
        processed = data[:: self.display_ds] 

        # Filter
        if self.display_filter is not None:
            processed, _ = apply_filter(processed, self.disp_filter)
        return processed

    def run(self):
        self.running = True 

        while self.running:
            # Get command from cmd queue
            try: 
                current_command = self.cmd_queue.get()
                if current_command['param'] == 'filter_type': 
                    pass 
                elif current_command['param'] == '':
                    pass 
            except queue.Empty: 
                pass 

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
