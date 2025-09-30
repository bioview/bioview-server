import queue
from typing import Callable, Dict, List

import numpy as np

from bioview_common import DataSource, log_print
from bioview_server.utils import apply_filter, emit_signal, get_filter

MODIFIABLE_PARAMS = [
    "display_ds",
    "display_filter_type",
    "display_filter_f_low",
    "display_filter_f_high",
]


class DisplayWorker:
    def __init__(
        self,
        samp_rate: int,
        display_ds: int,
        display_sources: List[DataSource],
        data_queue: queue.Queue,  # Data comes in
        cmd_queue: queue.Queue,  # To handle display filter changes, for example
        data_ready: Callable,  # Data pushed to client
        display_filter: Dict = None,
        logger = None 
    ):
        super().__init__()
        # Sources
        self.display_sources = display_sources

        # Processing
        self.display_ds = display_ds

        if display_filter:
            self.display_filter = get_filter(
                bounds=display_filter["bounds"],
                samp_rate=samp_rate,
                btype=display_filter["btype"],
                ftype=display_filter["ftype"],
            )
        else: 
            self.display_filter = None

        # Queues
        self.data_queue = data_queue
        self.cmd_queue = cmd_queue

        # State
        self.running = False

        # Signals
        self.data_ready = data_ready
        self.logger = logger 

    def process(self, data):
        # Downsample
        # NOTE: This may be replaced by scipy.decimate()
        processed = data[:: self.display_ds]

        # Filter
        if self.display_filter:
            processed, _ = apply_filter(processed, self.display_filter)
        return processed

    def run(self):
        self.running = True

        while self.running:
            # Get command from cmd queue
            try:
                current_command = self.cmd_queue.get()
                if current_command["param"] == "filter_type":
                    # TODO: Filter Implementation
                    pass
                elif current_command["param"] == "":
                    # TODO: Other implementation
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
                log_print(self.logger, "warning", "Queue Empty")
                continue
            except Exception as e:
                log_print(self.logger, "warning", f"Display error: {e}")
                continue

        log_print(self.logger, "debug", "Display stopped")

    def stop(self):
        self.running = False
