import queue

import numpy as np

from bioview_common import DataSource
from bioview_server.constants import Configuration
from bioview_server.utils import apply_filter, get_filter, emit_signal

class DisplayWorker():
    def __init__(
        self,
        config: Configuration,
        data_queue: queue.Queue,
        running: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.disp_ds = config.get_param("disp_ds", 1)

        self.disp_filter_spec = config.get_param("disp_filter_spec", None)
        self.disp_filter = None
        self._load_disp_filter()

        self.data_queue = data_queue
        self.running = running

        # Define signals 
        self.data_ready = None
        self.log_event = None 
        
    def _load_disp_filter(self):
        if self.disp_filter_spec is not None:
            self.disp_filter = get_filter(
                bounds=self.disp_filter_spec["bounds"],
                samp_rate=self.disp_filter_spec["samp_rate"],
                btype=self.disp_filter_spec["btype"],
                ftype=self.disp_filter_spec["ftype"],
            )

    def _update_disp_filter(self, param, value):
        if self.disp_filter_spec is None:
            self.disp_filter_spec = {}

        self.disp_filter_spec[param] = value

        # Update filter
        self._load_disp_filter()

    def _process(self, data):
        # Downsample
        processed = data[:: self.disp_ds]
        # Filter
        if self.disp_filter is not None:
            processed, _ = apply_filter(processed, self.disp_filter)
        return processed

    def run(self):
        emit_signal(self.log_event, "debug", "Display started")

        while self.running:
            self.display_sources = self.config.get_param("display_sources", [])
            if len(self.display_sources) == 0:
                continue

            try:
                # Load samples
                samples = self.data_queue.get()

                # Only process selected channels
                for source in enumerate(self.display_sources):
                    disp_samples = samples[source.channel, :]
                    processed = self._process(disp_samples)

                    # We send data with sources
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
