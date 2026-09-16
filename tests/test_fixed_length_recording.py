"""A routine of a given length must produce a file of exactly one size.

Before the recorder held itself to a sample budget, the length of a run came
out of whatever the 200 ms routine timer, the stop command's round trip and the
radios' spin-up happened to cost that time -- so two runs of the same routine
produced files that differed by a few thousand samples.
"""

import numpy as np
import pytest
from test_bvr_recording import read_bvr

from bioview_server.common.save import BvrWriter


FS = 1000.0
N_ROWS = 2


def _writer(tmp_path, limits=None):
    writer = BvrWriter(
        save_path=tmp_path / "run.bvr",
        data_queue=None,
        devices=[{"device_id": "RF", "fs": FS, "n_rows": N_ROWS, "dtype": "float32"}],
        sample_limits=limits,
    )
    writer.open(t0_unix=0.0)
    return writer


def _feed(writer, chunk_sizes):
    """Push chunks through the recorder the way the save queue would."""
    idx = 0
    for n in chunk_sizes:
        writer._append(
            {
                "device_id": "RF",
                "data": np.arange(N_ROWS * n, dtype=np.float32).reshape(N_ROWS, n),
                "sample_idx": idx,
                "t_wall": 0.0,
            }
        )
        idx += n
    writer._flush()


def _samples_in(path):
    _header, records, _trailer = read_bvr(path)
    return sum(block.shape[1] for _idx, _t, block in records["RF"])


@pytest.mark.parametrize(
    "chunk_sizes",
    [
        [400] * 30,  # the stop lands mid-chunk
        [400] * 26,  # the stop lands one whole chunk late
        [137] * 90,  # a chunk size that does not divide the budget
    ],
)
def test_the_file_holds_exactly_the_requested_seconds(tmp_path, chunk_sizes):
    """Ten seconds is 10_000 samples however the chunks happen to fall."""
    writer = _writer(tmp_path, limits={"RF": int(10 * FS)})
    _feed(writer, chunk_sizes)
    writer.cleanup()

    assert _samples_in(tmp_path / "run.bvr") == 10_000


def test_without_a_limit_the_length_follows_the_stream(tmp_path):
    """Free-running recordings are untouched by the cap machinery."""
    writer = _writer(tmp_path)
    _feed(writer, [400] * 30)
    writer.cleanup()

    assert _samples_in(tmp_path / "run.bvr") == 12_000


def test_surplus_chunks_cost_nothing_once_the_budget_is_met(tmp_path):
    writer = _writer(tmp_path, limits={"RF": 1000})
    _feed(writer, [400] * 5)
    writer.cleanup()

    assert writer._stats["RF"]["samples"] == 1000
    assert _samples_in(tmp_path / "run.bvr") == 1000


def test_the_trailer_records_the_budget(tmp_path):
    """A reader should be able to tell a capped file from a truncated one."""
    writer = _writer(tmp_path, limits={"RF": 1000})
    _feed(writer, [400] * 5)
    writer.cleanup()

    _header, _records, trailer = read_bvr(tmp_path / "run.bvr")
    assert trailer["devices"][0]["sample_limit"] == 1000


def test_a_free_running_file_says_it_had_no_budget(tmp_path):
    writer = _writer(tmp_path)
    _feed(writer, [400] * 3)
    writer.cleanup()

    _header, _records, trailer = read_bvr(tmp_path / "run.bvr")
    assert trailer["devices"][0]["sample_limit"] is None


def test_the_limit_is_derived_from_each_device_s_own_rate():
    """Groups running at different rates each get their own budget."""
    from bioview_server.server import Server

    limits = Server._sample_limits(
        type("S", (), {"logger": None})(),
        {"record_duration_s": 2.5},
        [
            {"device_id": "RF", "fs": 10_000.0},
            {"device_id": "MIC", "fs": 16_000.0},
            {"device_id": "BROKEN", "fs": None},
        ],
    )
    assert limits == {"RF": 25_000, "MIC": 40_000}
