"""Unit tests for the ``bioview-raw-v3`` writer and the per-device forwarder."""

import json
import multiprocessing as mp
import queue
import struct
import time

import numpy as np
import pytest
from bioview_common import (
    BVR3_MAGIC,
    BVR_TRAILER_MAGIC,
    RECORD_HEADER_SIZE,
    unpack_record,
)

from bioview_server.common.save import BvrWriter, SaveForwarder, flatten_chunk


def _devices():
    return [
        {"device_id": "A", "fs": 1000.0, "n_rows": 2, "dtype": "float32", "sources": []},
        {"device_id": "B", "fs": 100.0, "n_rows": 3, "dtype": "float32", "sources": []},
    ]


def _parse(path):
    blob = path.read_bytes()
    assert blob[:4] == BVR3_MAGIC
    (header_len,) = struct.unpack("!I", blob[4:8])
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    pos = 8 + header_len

    trailer, end = None, len(blob)
    if blob[-8:] == BVR_TRAILER_MAGIC:
        (trailer_len,) = struct.unpack("!Q", blob[-16:-8])
        start = len(blob) - 16 - trailer_len
        trailer = json.loads(blob[start : start + trailer_len].decode("utf-8"))
        end = start

    records = []
    while pos + RECORD_HEADER_SIZE <= end:
        idx, flags, n_samples, t_us, sample_idx = unpack_record(blob, pos)
        pos += RECORD_HEADER_SIZE
        dtype = np.float64 if flags & 1 else np.float32
        n_rows = header["devices"][idx]["n_rows"]
        nbytes = n_rows * n_samples * np.dtype(dtype).itemsize
        block = np.frombuffer(blob[pos : pos + nbytes], dtype=dtype).reshape(
            n_rows, n_samples
        )
        pos += nbytes
        records.append(
            {
                "device_id": header["devices"][idx]["device_id"],
                "sample_idx": sample_idx,
                "t_offset_us": t_us,
                "block": block,
            }
        )
    assert pos == end
    return header, records, trailer


def _run(writer, items, timeout=5.0):
    for item in items:
        writer.data_queue.put(item)
    writer.start()
    writer.resume()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if writer.data_queue.empty() and writer.records_written >= len(items):
            break
        time.sleep(0.02)
    writer.stop()
    writer.join(timeout=timeout)


def test_flatten_chunk_handles_complex_and_real_layouts():
    complex_chunk = np.arange(2 * 4 * 2, dtype=float).reshape(2, 4, 2)
    flat = flatten_chunk(complex_chunk)
    assert flat.shape == (4, 4)
    np.testing.assert_array_equal(flat[:2], complex_chunk[:, :, 0])
    np.testing.assert_array_equal(flat[2:], complex_chunk[:, :, 1])

    real_chunk = np.arange(2 * 4, dtype=float).reshape(2, 4)
    np.testing.assert_array_equal(flatten_chunk(real_chunk), real_chunk)


def test_round_trip_preserves_samples_and_device_identity(tmp_path):
    path = tmp_path / "rec.bvr"
    writer = BvrWriter(path, mp.Queue(), _devices())
    writer.open()

    a = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    b = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) * 2
    _run(
        writer,
        [
            {
                "device_id": "A",
                "data": a,
                "sample_idx": 0,
                "t_wall": writer.t0_unix + 0.1,
            },
            {
                "device_id": "B",
                "data": b,
                "sample_idx": 0,
                "t_wall": writer.t0_unix + 0.2,
            },
        ],
    )

    header, records, trailer = _parse(path)
    assert header["format"] == "bioview-raw-v3"
    assert [d["device_id"] for d in header["devices"]] == ["A", "B"]
    assert len(records) == 2

    np.testing.assert_array_equal(records[0]["block"], a)
    np.testing.assert_array_equal(records[1]["block"], b)
    assert records[0]["device_id"] == "A"
    assert records[1]["device_id"] == "B"
    # Offsets are relative to t0, and t0 is the only absolute time in the file.
    assert records[0]["t_offset_us"] == pytest.approx(100_000, abs=2000)
    assert records[1]["t_offset_us"] == pytest.approx(200_000, abs=2000)
    assert trailer["Annotations"] == []


def test_gap_in_sample_counter_is_recorded(tmp_path):
    path = tmp_path / "rec.bvr"
    writer = BvrWriter(path, mp.Queue(), _devices())
    writer.open()

    block = np.zeros((2, 10), dtype=np.float32)
    _run(
        writer,
        [
            {"device_id": "A", "data": block, "sample_idx": 0},
            # Jumps to 25 instead of 10: 15 samples were dropped upstream.
            {"device_id": "A", "data": block, "sample_idx": 25},
        ],
    )

    _header, _records, trailer = _parse(path)
    stats = {d["device_id"]: d for d in trailer["devices"]}
    assert stats["A"]["gaps"] == 1
    assert stats["A"]["dropped_samples"] == 15
    assert stats["A"]["samples"] == 20
    assert stats["B"]["gaps"] == 0


def test_wrong_row_count_is_rejected_not_written(tmp_path):
    """A chunk that does not match its header entry must never reach the file."""
    path = tmp_path / "rec.bvr"
    writer = BvrWriter(path, mp.Queue(), _devices())
    writer.open()

    good = np.ones((2, 3), dtype=np.float32)
    bad = np.ones((5, 3), dtype=np.float32)  # device A declares 2 rows
    _run(
        writer,
        [
            {"device_id": "A", "data": good, "sample_idx": 0},
            {"device_id": "A", "data": bad, "sample_idx": 3},
            {"device_id": "Nonexistent", "data": good, "sample_idx": 0},
        ],
        timeout=2.0,
    )

    _header, records, _trailer = _parse(path)
    assert len(records) == 1
    np.testing.assert_array_equal(records[0]["block"], good)
    assert writer.unknown_device_records == 1


def test_annotations_and_changes_are_relative_offsets(tmp_path):
    path = tmp_path / "rec.bvr"
    writer = BvrWriter(path, mp.Queue(), _devices())
    writer.open()

    writer.record_annotation("inhale", t_wall=writer.t0_unix + 1.5)
    writer.record_change("A", "rx_gain", [30, 31], t_wall=writer.t0_unix + 2.25)
    _run(
        writer,
        [{"device_id": "A", "data": np.zeros((2, 2), np.float32), "sample_idx": 0}],
    )

    header, _records, trailer = _parse(path)
    assert trailer["Annotations"] == [{"offset_us": 1_500_000, "text": "inhale"}]
    change = trailer["param_changes"][0]
    assert change["offset_us"] == 2_250_000
    assert change["device_id"] == "A"
    assert change["param"] == "rx_gain"
    assert "t0_unix" in header


def test_unfinished_file_has_no_trailer_but_keeps_records(tmp_path):
    """A recording killed mid-run must still yield every complete record."""
    path = tmp_path / "rec.bvr"
    writer = BvrWriter(path, mp.Queue(), _devices())
    writer.open()
    block = np.ones((2, 4), dtype=np.float32)
    writer._append({"device_id": "A", "data": block, "sample_idx": 0})
    writer._flush()
    # No cleanup(): simulates a process that died before closing.

    blob = path.read_bytes()
    assert blob[-8:] != BVR_TRAILER_MAGIC
    header, records, trailer = _parse(path)
    assert trailer is None
    assert len(records) == 1
    np.testing.assert_array_equal(records[0]["block"], block)


def test_forwarder_tags_chunks_with_device_and_counter():
    src, dst = queue.Queue(), queue.Queue()
    fwd = SaveForwarder(device_id="A", data_input_queue=src, data_output_queue=dst)

    src.put({"data": np.zeros((2, 5), np.float32), "sample_idx": 40, "t_wall": 123.0})
    src.put(np.zeros((2, 7), np.float32))  # bare array: counter is derived

    fwd.start()
    fwd.resume()
    first = dst.get(timeout=5)
    second = dst.get(timeout=5)
    fwd.stop()
    fwd.join(timeout=5)

    assert first["device_id"] == "A"
    assert first["sample_idx"] == 40
    assert first["t_wall"] == 123.0
    # Falls back to continuing from where the tagged chunk ended.
    assert second["sample_idx"] == 45
    assert second["t_wall"] > 0
