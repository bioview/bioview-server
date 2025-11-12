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
    '''Simple wrapper worker that adds information to display queue which is then sent to frontend'''
    def __init__(
        self,
        display_sources: List[DataSource] = None, 
        data_input_queue: queue.Queue = None,  # Data comes in
        data_output_queue: queue.Queue = None,  # Data pushed to client
        logger = None
    ):
        super().__init__()
        # Keep track of sources 
        self.display_sources = display_sources if display_sources is not None else []
        
        # Queues
        self.data_input_queue = data_input_queue
        self.data_output_queue = data_output_queue

        # State
        self.running = False

        self.logger = logger 

    def work(self):
        # Nothing to do if we have no queues; return
        if self.data_input_queue is None or self.data_output_queue is None:
            return
        
        while self.is_running:
            try:
                # Get samples 
                samples = self.data_input_queue.get_nowait()
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

        log_print(self.logger, "debug", "Display stopped") 

    # TODO: Check if we need cleanup