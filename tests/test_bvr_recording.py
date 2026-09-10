"""End-to-end tests for the server-side ``bioview-raw-v3`` recorder.

The case that matters is two devices at *different* save rates in one session:
each emits its own chunks of its own width, and the file has to keep them
separable. Uses the fake backend, so no hardware is needed.
"""

import json
import struct
import time

import numpy as np
import pytest
from bioview_common import (
    BVR3_MAGIC,
    BVR_TRAILER_MAGIC,
    RECORD_HEADER_SIZE,
    Command,
    Response,
    unpack_record,
)
from fakes import FakeConfiguration


def _fake_group(name, samp_rate, num_channels, save_ds, signal_freq=1.0):
    return {
        name: FakeConfiguration.from_dict(
            {
                "type": "FAKE",
                "samp_rate": samp_rate,
                "num_channels": num_channels,
                "signal_freq": signal_freq,
                "amplitude": 1.0,
                "noise_std": 0.0,
                "chunk_duration": 0.05,
                "save_ds": save_ds,
                "disp_ds": 1,
            }
        ).to_dict()
    }


def read_bvr(path):
    """Parse a v3 file into (header, per-device records, trailer).

    Records come back as ``{device_id: [(sample_idx, t_offset_us, block), ...]}``
    in file order.
    """
    blob = path.read_bytes()
    assert blob[:4] == BVR3_MAGIC, "not a bioview-raw-v3 file"
    (header_len,) = struct.unpack("!I", blob[4:8])
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    pos = 8 + header_len

    end = len(blob)
    trailer = None
    if blob[-8:] == BVR_TRAILER_MAGIC:
        (trailer_len,) = struct.unpack("!Q", blob[-16:-8])
        start = len(blob) - 16 - trailer_len
        trailer = json.loads(blob[start : start + trailer_len].decode("utf-8"))
        end = start

    devices = header["devices"]
    out = {d["device_id"]: [] for d in devices}
    while pos + RECORD_HEADER_SIZE <= end:
        device_idx, flags, n_samples, t_offset_us, sample_idx = unpack_record(blob, pos)
        pos += RECORD_HEADER_SIZE
        dev = devices[device_idx]
        dtype = np.float64 if flags & 1 else np.float32
        n_rows = dev["n_rows"]
        nbytes = n_rows * n_samples * np.dtype(dtype).itemsize
        block = np.frombuffer(blob[pos : pos + nbytes], dtype=dtype).reshape(
            n_rows, n_samples
        )
        pos += nbytes
        out[dev["device_id"]].append((sample_idx, t_offset_us, block))
    assert pos == end, f"trailing bytes: parsed to {pos}, records end at {end}"
    return header, out, trailer


def _record(client, tmp_path, groups, seconds=1.5, label=None):
    """Run one full initialize/stream/stop cycle and return the written file."""
    resp_type, _ = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": groups}
    )
    assert resp_type == Response.SUCCESS.name

    experiment = {
        "type": "EXPERIMENT",
        "file_name": "session.bvr",
        "save_dir": str(tmp_path),
    }
    if label:
        experiment["save_label"] = label

    resp_type, payload = client.command(
        Command.START_STREAMING, {"Experiment": experiment, **groups}
    )
    assert resp_type == Response.SUCCESS.name, payload

    time.sleep(seconds)

    resp_type, payload = client.command(Command.STOP_STREAMING, {})
    assert resp_type == Response.SUCCESS.name, payload

    files = list(tmp_path.glob("*.bvr"))
    assert len(files) == 1, f"expected one recording, got {files}"
    return files[0]


def test_records_a_single_device(client, tmp_path):
    groups = _fake_group("FakeA", samp_rate=500, num_channels=4, save_ds=1)
    path = _record(client, tmp_path, groups)

    header, records, trailer = read_bvr(path)
    assert header["format"] == "bioview-raw-v3"
    assert header["t0_unix"] > 0
    assert len(header["devices"]) == 1
    assert header["devices"][0]["device_id"] == "FakeA"
    assert header["devices"][0]["n_rows"] == 4
    assert header["devices"][0]["fs"] == pytest.approx(500.0)

    blocks = records["FakeA"]
    assert blocks, "no records written"
    assert all(b.shape[0] == 4 for _idx, _t, b in blocks)
    assert trailer is not None and trailer["devices"][0]["samples"] > 0


def test_two_devices_at_different_rates_stay_separable(client, tmp_path):
    """The case the old single-matrix format could not represent."""
    groups = {
        **_fake_group("FastDev", samp_rate=1000, num_channels=2, save_ds=1),
        **_fake_group("SlowDev", samp_rate=1000, num_channels=3, save_ds=10),
    }
    path = _record(client, tmp_path, groups, seconds=2.0)

    header, records, trailer = read_bvr(path)
    by_id = {d["device_id"]: d for d in header["devices"]}
    assert by_id["FastDev"]["fs"] == pytest.approx(1000.0)
    assert by_id["SlowDev"]["fs"] == pytest.approx(100.0)
    assert by_id["FastDev"]["n_rows"] == 2
    assert by_id["SlowDev"]["n_rows"] == 3

    fast = records["FastDev"]
    slow = records["SlowDev"]
    assert fast and slow, "both devices must have written records"

    # Each device's rows are its own; nothing is interleaved into a shared width.
    assert all(b.shape[0] == 2 for _i, _t, b in fast)
    assert all(b.shape[0] == 3 for _i, _t, b in slow)

    fast_samples = sum(b.shape[1] for _i, _t, b in fast)
    slow_samples = sum(b.shape[1] for _i, _t, b in slow)
    # 10x the save rate means ~10x the samples over the same wall-clock window.
    assert fast_samples > slow_samples * 5, (fast_samples, slow_samples)

    # Sample counters are contiguous per device, so the file proves no gaps.
    for blocks in (fast, slow):
        expected = 0
        for sample_idx, _t, block in blocks:
            assert sample_idx == expected
            expected += block.shape[1]

    stats = {d["device_id"]: d for d in trailer["devices"]}
    assert stats["FastDev"]["gaps"] == 0
    assert stats["SlowDev"]["gaps"] == 0
    assert stats["FastDev"]["samples"] == fast_samples
    assert stats["SlowDev"]["samples"] == slow_samples


def test_saved_rate_follows_save_ds_not_disp_ds(client, tmp_path):
    """The recording keeps the save stream; disp_ds must not decimate it."""
    groups = _fake_group("FakeA", samp_rate=1000, num_channels=2, save_ds=4)
    groups["FakeA"]["disp_ds"] = 10

    path = _record(client, tmp_path, groups, seconds=2.0)
    header, records, trailer = read_bvr(path)

    assert header["devices"][0]["fs"] == pytest.approx(250.0)
    samples = sum(b.shape[1] for _i, _t, b in records["FakeA"])
    elapsed_s = trailer["t_end_offset_us"] / 1e6
    achieved = samples / elapsed_s
    # Must land near samp_rate/save_ds (250 Hz), nowhere near
    # samp_rate/(save_ds*disp_ds) (25 Hz), which is the display rate.
    assert 150 < achieved < 350, achieved


def test_signal_content_is_intact(client, tmp_path):
    """A known sine must survive the round trip undistorted."""
    groups = _fake_group(
        "FakeA", samp_rate=500, num_channels=2, save_ds=1, signal_freq=5.0
    )
    path = _record(client, tmp_path, groups, seconds=2.0)

    _header, records, _trailer = read_bvr(path)
    joined = np.hstack([b for _i, _t, b in records["FakeA"]])
    assert joined.shape[0] == 2

    row = joined[0].astype(np.float64)
    assert np.isfinite(row).all()
    # A clean sine is smooth sample-to-sample; interleaving would destroy this.
    assert np.corrcoef(row[:-1], row[1:])[0, 1] > 0.9
    assert 0.9 < np.abs(row).max() <= 1.01


def test_annotations_and_offsets_are_relative(client, tmp_path):
    groups = _fake_group("FakeA", samp_rate=500, num_channels=2, save_ds=1)
    path = _record(client, tmp_path, groups, label="RunA")

    header, _records, trailer = read_bvr(path)
    # Only t0 is absolute; every other time in the file is an offset from it.
    assert "t0_unix" in header and "t0_utc" in header
    assert trailer["t_end_offset_us"] > 0
    assert isinstance(trailer["Annotations"], list)
    assert isinstance(trailer["param_changes"], list)
    assert path.name.startswith("session_RunA")


def test_no_recording_without_a_file_name(client, tmp_path):
    groups = _fake_group("FakeA", samp_rate=500, num_channels=2, save_ds=1)
    resp_type, _ = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": groups}
    )
    assert resp_type == Response.SUCCESS.name

    resp_type, _ = client.command(
        Command.START_STREAMING,
        {"Experiment": {"type": "EXPERIMENT", "save_dir": str(tmp_path)}, **groups},
    )
    assert resp_type == Response.SUCCESS.name
    time.sleep(0.5)
    client.command(Command.STOP_STREAMING, {})

    assert list(tmp_path.glob("*.bvr")) == []
