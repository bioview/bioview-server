"""HDF5 save path: batching, handle reuse, and both chunk layouts."""

import queue
import time

import h5py
import numpy as np
import pytest

from bioview_server.common.save import SaveWorker, flatten_chunk


def _drain_into(worker, chunks, timeout=5.0):
    for chunk in chunks:
        worker.data_queue.put(chunk)
    worker.start()
    worker.resume()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.data_queue.empty() and worker.samples_written:
            break
        time.sleep(0.02)
    worker.stop()
    worker.join(timeout=timeout)


def test_flatten_chunk_handles_complex_and_real_layouts():
    """The old code indexed chunk[:, :, 1] unconditionally and raised on 2-D."""
    complex_chunk = np.arange(2 * 4 * 2, dtype=float).reshape(2, 4, 2)
    flat = flatten_chunk(complex_chunk)
    assert flat.shape == (4, 4)
    np.testing.assert_array_equal(flat[:2], complex_chunk[:, :, 0])
    np.testing.assert_array_equal(flat[2:], complex_chunk[:, :, 1])

    real_chunk = np.arange(2 * 4, dtype=float).reshape(2, 4)
    np.testing.assert_array_equal(flatten_chunk(real_chunk), real_chunk)


@pytest.mark.parametrize("complex_chunks", [False, True])
def test_save_worker_round_trip(tmp_path, complex_chunks):
    path = tmp_path / "out.h5"
    n_ch, n_samp, n_chunks = 3, 50, 8
    rng = np.random.default_rng(0)
    if complex_chunks:
        chunks = [rng.normal(size=(n_ch, n_samp, 2)) for _ in range(n_chunks)]
        expected_rows = 2 * n_ch
    else:
        chunks = [rng.normal(size=(n_ch, n_samp)) for _ in range(n_chunks)]
        expected_rows = n_ch

    worker = SaveWorker(
        save_path=str(path),
        data_queue=queue.Queue(),
        num_channels=n_ch,
        batch_chunks=3,
    )
    _drain_into(worker, chunks)

    with h5py.File(path, "r") as f:
        data = f["data"][...]
    assert data.shape == (expected_rows, n_ch and n_chunks * n_samp)
    expected = np.hstack([flatten_chunk(c) for c in chunks])
    np.testing.assert_allclose(data, expected)


def test_save_worker_keeps_one_file_handle(tmp_path):
    """Regression: the file used to be reopened for every single chunk."""
    path = tmp_path / "out.h5"
    opens = {"n": 0}
    real_open = h5py.File

    class CountingFile(real_open):
        def __init__(self, *a, **kw):
            opens["n"] += 1
            super().__init__(*a, **kw)

    worker = SaveWorker(
        save_path=str(path),
        data_queue=queue.Queue(),
        num_channels=2,
        batch_chunks=2,
    )
    import bioview_server.common.save as save_mod

    save_mod.h5py.File = CountingFile
    try:
        _drain_into(worker, [np.ones((2, 10)) for _ in range(20)])
    finally:
        save_mod.h5py.File = real_open

    assert opens["n"] == 1, f"file opened {opens['n']} times, expected once"


def test_save_worker_flushes_buffered_chunks_on_stop(tmp_path):
    """A partial batch must not be stranded in memory when streaming stops."""
    path = tmp_path / "out.h5"
    worker = SaveWorker(
        save_path=str(path),
        data_queue=queue.Queue(),
        num_channels=2,
        batch_chunks=1000,  # far more than we will send
    )
    _drain_into(worker, [np.full((2, 10), 7.0) for _ in range(3)])

    with h5py.File(path, "r") as f:
        assert f["data"].shape == (2, 30)
        assert np.all(f["data"][...] == 7.0)
