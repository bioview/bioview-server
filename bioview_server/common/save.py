import multiprocessing as mp
import queue

import h5py
import numpy as np

from bioview_common import PausableWorker

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


class SaveWorker(PausableWorker):
    def __init__(
        self,
        save_path,
        data_queue: mp.Queue,
        num_channels: int,
        logger = None 
    ):
        super().__init__()  
        self.logger = logger

        # Load output file
        self.save_path = save_path
        self.data_queue = data_queue

        if self.saving:
            init_save_file(file_path=self.save_path, num_channels=num_channels)

    def work(self):
        if self.data_queue is None:
            return

        while self.is_running:
            try:
                data = self.data_queue.get()
            except queue.Empty:
                continue

            update_save_file(self.save_path, data)

    # TODO: Check - Do we need any cleanup?