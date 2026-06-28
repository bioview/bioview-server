import queue
from typing import List

from bioview_common import log_print, DataSource, PausableWorker

MODIFIABLE_PARAMS = [
    "display_ds",
    "display_filter_type",
    "display_filter_f_low",
    "display_filter_f_high",
]


class DisplayWorker(PausableWorker):
    '''Wrapper worker that forwards processed data to the output queue for the
    client. It emits a contiguous (num_sources, num_samples) numpy array together
    with the ordered list of data sources describing each row, so the client can
    save the full stream and route individual rows to plots reliably.'''
    def __init__(
        self,
        display_sources: List[DataSource] = None, 
        data_input_queue: queue.Queue = None,  # Data comes in
        data_output_queue: queue.Queue = None,  # Data pushed to client
        logger = None
    ):
        super().__init__()
        # Ordered list of sources; row i of each emitted array corresponds to
        # display_sources[i]. Ordered by channel so it matches ProcessWorker output.
        self.display_sources = sorted(
            display_sources if display_sources is not None else [],
            key=lambda s: s.channel,
        )

        # Precompute serializable source descriptors (sent as chunk metadata)
        self._source_dicts = [s.to_dict() for s in self.display_sources]

        # Queues
        self.data_input_queue = data_input_queue
        self.data_output_queue = data_output_queue

        self.logger = logger 

    def work(self):
        # Nothing to do if we have no queues; return
        if self.data_input_queue is None or self.data_output_queue is None:
            return

        try:
            # Get a processed (num_sources, num_samples) chunk
            samples = self.data_input_queue.get(timeout=0.1)
        except queue.Empty:
            return

        try:
            payload = {"data": samples, "sources": self._source_dicts}
            self.data_output_queue.put_nowait(payload)
        except queue.Full:
            log_print(self.logger, 'warning', 'Display queue filled up. Unable to add any more data.')
        except Exception as e:
            log_print(self.logger, 'error', f'Error occurred: {e}')
