import h5py
import numpy as np
from pathlib import Path

def get_cache_file(file_name):
    cache_file = Path.home() / ".bioview" / file_name

    if not cache_file.exists():
        cache_file.parent.mkdir(parents=False, exist_ok=True)
        cache_file.touch()

    return cache_file


def get_unique_path(dirname, filename):
    f_path = Path(dirname) / filename
    base, ext = f_path.stem, f_path.suffix
    counter = 1
    while f_path.exists():
        counter += 1
        f_path = f_path.with_name(f"{base}_{counter}{ext}")
    return f_path


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
