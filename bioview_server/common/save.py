import queue

from bioview_server.utils import init_save_file, update_save_file, emit_signal

class SaveWorker:
    def __init__(
        self, 
        save_path, 
        data_queue, 
        num_channels, 
        running: bool = False, 
        parent=None
    ):
        super().__init__(parent)
        # Signals 
        self.log_event = None 
        
        # Variables
        self.running = running

        # Load output file
        self.save_path = save_path
        self.data_queue = data_queue

        if self.saving:
            init_save_file(file_path=self.save_path, num_channels=num_channels)

    def run(self):
        if self.data_queue is None:
            return

        self.running = True 
        
        while self.running:
            try:
                data = self.data_queue.get()
            except queue.Empty:
                emit_signal(self.logEvent, "debug", "No data to save")
                continue

            update_save_file(self.save_path, data)

    def stop(self):
        self.running = False
