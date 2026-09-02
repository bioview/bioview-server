"""HDF5 recorder for demodulated chunks.

The file handle is opened once for the session and held; chunks are accumulated
and appended in batches. The previous implementation reopened the file with
``h5py.File(path, "a")`` on *every* chunk, which at a 20 ms chunk cadence meant
50 open/flush/close cycles a second -- by far the most expensive thing on the
streaming path, and enough to back the whole pipeline up on a slow disk.
"""

import contextlib
import multiprocessing as mp
import queue

import h5py
import numpy as np
from bioview_common import PausableWorker, log_print


#: Chunks accumulated before one append. Each append is a resize + a write, so
#: batching amortizes both. At ~20 ms chunks this is a write roughly every third
#: of a second.
DEFAULT_BATCH_CHUNKS = 16

#: HDF5 dataset chunk width (columns). Reads and writes are aligned to this.
DEFAULT_CHUNK_COLS = 500


def flatten_chunk(chunk) -> np.ndarray:
    """Fold a chunk into a 2-D ``(rows, samples)`` block ready to append.

    A complex chunk arrives as ``(channels, samples, 2)`` and is stored as all
    real rows followed by all imaginary rows. A real-valued chunk is already
    ``(channels, samples)`` and passes through -- the old code indexed
    ``chunk[:, :, 1]`` unconditionally and raised on the real-valued case.
    """
    arr = np.asarray(chunk)
    if arr.ndim == 3:
        return np.vstack([arr[:, :, 0], arr[:, :, 1]])
    return arr


class SaveWorker(PausableWorker):
    #: How long ``work()`` blocks on the queue before checking for a pause.
    QUEUE_TIMEOUT_S = 0.1

    def __init__(
        self,
        save_path,
        data_queue: mp.Queue,
        num_channels: int,
        batch_chunks: int = DEFAULT_BATCH_CHUNKS,
        chunk_cols: int = DEFAULT_CHUNK_COLS,
        logger=None,
    ):
        super().__init__()
        self.logger = logger

        self.save_path = save_path
        self.data_queue = data_queue
        self.num_channels = int(num_channels)
        self.batch_chunks = max(int(batch_chunks), 1)
        self.chunk_cols = max(int(chunk_cols), 1)

        self._file = None
        self._dset = None
        self._pending = []
        self.samples_written = 0
        self.chunks_dropped = 0

    # ------------------------------------------------------------------ file

    def _ensure_open(self, n_rows: int):
        """Create the file on the first append, sized to the actual chunk.

        The row count is only known once a chunk arrives: it is
        ``num_channels`` for a real stream and ``2 * num_channels`` for a
        complex one. Creating it up front with ``num_channels`` and a matching
        ``maxshape`` meant the first complex append tried to resize a fixed
        axis and failed.
        """
        if self._dset is not None:
            return
        self._file = h5py.File(self.save_path, "w")
        self._dset = self._file.create_dataset(
            "data",
            shape=(n_rows, 0),
            maxshape=(n_rows, None),
            dtype="float64",
            chunks=(n_rows, self.chunk_cols),
        )

    def _flush(self):
        if not self._pending:
            return
        block = self._pending[0] if len(self._pending) == 1 else np.hstack(self._pending)
        self._pending = []

        try:
            self._ensure_open(block.shape[0])
            if block.shape[0] != self._dset.shape[0]:
                log_print(
                    self.logger,
                    "error",
                    f"[Save] Chunk has {block.shape[0]} rows but the dataset has "
                    f"{self._dset.shape[0]}; dropping the batch",
                )
                return
            start = self._dset.shape[1]
            self._dset.resize((self._dset.shape[0], start + block.shape[1]))
            self._dset[:, start:] = block
            self.samples_written = start + block.shape[1]
        except Exception as e:
            log_print(self.logger, "error", f"[Save] Write failed: {e}")

    # ---------------------------------------------------------------- worker

    def work(self):
        if self.data_queue is None:
            return

        while self.is_running:
            try:
                data = self.data_queue.get(timeout=self.QUEUE_TIMEOUT_S)
            except queue.Empty:
                # Idle: commit whatever is held so a paused or slow stream never
                # leaves data sitting only in memory.
                self._flush()
                continue
            except (OSError, ValueError):
                break

            try:
                self._pending.append(flatten_chunk(data))
            except Exception as e:
                log_print(self.logger, "error", f"[Save] Bad chunk: {e}")
                continue

            if len(self._pending) >= self.batch_chunks:
                self._flush()

        # Paused or stopping: do not strand buffered chunks.
        self._flush()
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.flush()

    def cleanup(self):
        self._flush()
        if self._file is not None:
            try:
                self._file.close()
                log_print(self.logger, "debug", f"[Save] Closed {self.save_path}")
            except Exception as e:
                log_print(self.logger, "error", f"[Save] Close failed: {e}")
            finally:
                self._file = None
                self._dset = None


# --------------------------------------------------------------------------
# Standalone helpers, kept for one-shot writes and tests.
# --------------------------------------------------------------------------


def init_save_file(file_path, num_channels: int, chunk_size: int = DEFAULT_CHUNK_COLS):
    with h5py.File(file_path, "w") as f:
        f.create_dataset(
            "data",
            shape=(num_channels, 0),
            maxshape=(num_channels, None),
            dtype="float64",
            chunks=(num_channels, chunk_size),
        )


def update_save_file(file_path, chunk):
    save_chunk = flatten_chunk(chunk)
    with h5py.File(file_path, "a") as f:
        dset = f["data"]
        cur_cols = dset.shape[1]
        new_cols = cur_cols + save_chunk.shape[1]
        dset.resize((dset.shape[0], new_cols))
        dset[:, cur_cols:new_cols] = save_chunk
