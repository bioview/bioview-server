import queue
import multiprocessing as mp

import h5py
import numpy as np

from typing import Callable

from bioview_server.utils import emit_signal

def init_save_file(file_path, num_channels: int, chunk_size: int = 500):
    with h5py.File(file_path, "w") as f:
        f.create_dataset(
            "data",
            shape=(num_channels, 0),
            maxshape=(num_channels, None),
            dtype="float64",
            chunks=(num_channels, chunk_size),
        )

def update_save_file(file_path, chunk):
    save_chunk = np.vstack(
        [chunk[:, :, 0], chunk[:, :, 1]]
    )  # num_channels x num_samples (all real followed by all imag)

    with h5py.File(file_path, "a") as f:
        dset = f["data"]
        cur_cols = dset.shape[1]
        new_cols = cur_cols + save_chunk.shape[1]
        dset.resize((save_chunk.shape[0], new_cols))
        dset[:, cur_cols:new_cols] = save_chunk


class SaveWorker:
    def __init__(
        self, 
        save_path, 
        data_queue: mp.Queue, 
        num_channels: int, 
        log_event: Callable = None, 
        parent = None
    ):
        super().__init__(parent)
        # Signals 
        self.log_event = None 
        
        # Variables
        self.running = False

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
