"""Real-time suitability of the processing pipeline, on dummy data.

Two things are checked, both against the wall-clock budget a chunk actually has:

1. Per-stage cost. Each stage is timed separately and printed as a table so a
   regression shows up as a number rather than as "streaming feels laggy".
2. Sustained behaviour. A dummy-RF stream is run long enough that any leak or
   imbalance would show, then the drop counters are required to be zero.

The budget matters more than the absolute numbers: a chunk of N samples at
``samp_rate`` must be fully processed in under ``N / samp_rate`` seconds, or the
queues fill and data is dropped no matter how deep they are.
"""

import time

import numpy as np
import pytest
from bioview_common import DataSource, apply_filter, get_filter
from bioview_common.signal_schemes import CwScheme

from bioview_server.common.save import flatten_chunk
from bioview_server.device.usrp.process import ProcessWorker


SAMP_RATE = 1e6
IF_HZ = 100e3
SAVE_DS = 10
CHUNK_SAMPLES = 40_000  # 40 ms of audio at 1 MSps
CHUNK_SECONDS = CHUNK_SAMPLES / SAMP_RATE
N_SOURCES = 4

#: Total pipeline budget as a fraction of real time. Leaves room for the socket,
#: the OS, and a second device group.
REALTIME_BUDGET = 0.5


def _make_worker(n_sources=N_SOURCES, save_ds=SAVE_DS):
    """A worker with ``n_sources`` Tx/Rx pairs, as a real 2x2 group would have."""
    n_tx = n_rx = int(np.sqrt(n_sources)) or 1
    ifs = [IF_HZ + i * 10e3 for i in range(n_tx)]

    sources = set()
    channel = 0
    for rx in range(n_rx):
        for tx in range(n_tx):
            src = DataSource(group_id="g", channel=channel, label=f"Tx{tx}Rx{rx}")
            src.tx_idx = tx
            src.rx_idx = rx
            sources.add(src)
            channel += 1

    scheme = CwScheme(SAMP_RATE, ifs, [1.0] * n_tx, [0.0] * n_tx)
    worker = ProcessWorker(
        data_sources=sources,
        cal_ref_sources=[],
        samp_rate=SAMP_RATE,
        channel_ifs=ifs,
        if_filter_bw=[5e3] * n_tx,
        rx_queues={"d": None},
        rx_device_order=["d"],
        schemes_by_device={"d": scheme},
        global_tx_to_device={i: ("d", i) for i in range(n_tx)},
        save_ds=save_ds,
    )
    return worker, n_rx


def _buffer(n_rx, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(CHUNK_SAMPLES) / SAMP_RATE
    rows = []
    for _ in range(n_rx):
        row = 0.5 * np.exp(1j * 2 * np.pi * IF_HZ * t)
        row = row + 0.01 * (
            rng.standard_normal(CHUNK_SAMPLES) + 1j * rng.standard_normal(CHUNK_SAMPLES)
        )
        rows.append(row)
    return np.vstack(rows).astype(np.complex64)


def _time(fn, repeats):
    fn()  # warm up (filter state, allocations)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def test_pipeline_stage_profile(capsys):
    """Print a per-stage cost table and hold the total under the budget."""
    worker, n_rx = _make_worker()
    buffer = _buffer(n_rx)
    repeats = 20

    filt = get_filter([IF_HZ - 2.5e3, IF_HZ + 2.5e3], SAMP_RATE, btype="band", order=2)
    stages = {}

    state = {"zi": None}

    def stage_filter():
        _out, state["zi"] = apply_filter(buffer[0], filt, zi=state["zi"])

    stages["band-pass filter (1 source)"] = _time(stage_filter, repeats)

    def stage_demod():
        worker._process_mimo_chunk(buffer)

    stages[f"demodulate ({N_SOURCES} sources)"] = _time(stage_demod, repeats)

    def stage_full():
        results = worker._process_mimo_chunk(buffer)
        worker._assemble_outputs(buffer, results)

    stages["demodulate + assemble"] = _time(stage_full, repeats)

    results = worker._process_mimo_chunk(buffer)
    save_data, _display = worker._assemble_outputs(buffer, results)

    def stage_flatten():
        flatten_chunk(save_data)

    stages["flatten for save"] = _time(stage_flatten, repeats)

    def stage_serialize():
        np.ascontiguousarray(save_data, dtype=np.float32).tobytes()

    stages["serialize to bytes"] = _time(stage_serialize, repeats)

    with capsys.disabled():
        print(
            f"\n  chunk = {CHUNK_SAMPLES} samples "
            f"({CHUNK_SECONDS * 1000:.0f} ms) x {N_SOURCES} sources"
        )
        print(f"  {'stage':<34} {'ms/chunk':>10} {'% real time':>13}")
        print(f"  {'-' * 34} {'-' * 10} {'-' * 13}")
        for name, seconds in stages.items():
            print(
                f"  {name:<34} {seconds * 1000:>10.2f} "
                f"{seconds / CHUNK_SECONDS * 100:>12.1f}%"
            )

    total = stages["demodulate + assemble"] + stages["serialize to bytes"]
    budget = total / CHUNK_SECONDS
    assert budget < REALTIME_BUDGET, (
        f"pipeline uses {budget:.0%} of real time for {N_SOURCES} sources; "
        f"budget is {REALTIME_BUDGET:.0%}"
    )


@pytest.mark.parametrize("save_ds", [1, 10, 100])
def test_decimation_factor_does_not_break_the_budget(save_ds):
    """save_ds=1 is the worst case: no decimation, one output per input sample."""
    worker, n_rx = _make_worker(save_ds=save_ds)
    buffer = _buffer(n_rx)

    def run():
        results = worker._process_mimo_chunk(buffer)
        worker._assemble_outputs(buffer, results)

    per_chunk = _time(run, 10)
    budget = per_chunk / CHUNK_SECONDS
    assert budget < REALTIME_BUDGET, f"save_ds={save_ds} uses {budget:.0%} of real time"


def test_repeated_processing_does_not_grow_memory():
    """A leak in the per-chunk path would show as unbounded growth here."""
    tracemalloc = pytest.importorskip("tracemalloc")
    worker, n_rx = _make_worker()
    buffer = _buffer(n_rx)

    for _ in range(5):
        worker._assemble_outputs(buffer, worker._process_mimo_chunk(buffer))

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    for _ in range(50):
        worker._assemble_outputs(buffer, worker._process_mimo_chunk(buffer))
    current = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    growth_kb = (current - baseline) / 1024
    assert growth_kb < 512, f"retained {growth_kb:.0f} KiB over 50 chunks"


def test_sustained_dummy_rf_stream_drops_nothing(server, client, capsys):
    """Run the dummy RF backend for a while; the drop counters must stay at zero."""
    from bioview_common import Command, DummyConfiguration, Response

    group_cfg = {
        "type": "DUMMY",
        "signal_scheme": "cw",
        "samp_rate": 200_000,
        "chunk_duration": 0.005,
        "noise_std": 0.001,
        "hardware": {
            "VirtA": {
                "tx_channels": [0, 1],
                "rx_channels": [0, 1],
                "if_freq": [10_000, 20_000],
                "tx_amplitude": [1, 1],
                "tx_phase": [0, 0],
            }
        },
        "channel_map": {"layout": "full_nxn", "dpic": []},
        "rf_simulation": {"on_axis_gain": 0.4, "cross_coupling": 0.08},
    }
    device_groups = {"ProfileGroup": DummyConfiguration.from_dict(group_cfg).to_dict()}

    resp_type, payload = client.device_command(
        Command.INITIALIZE_DEVICES, {"device_groups": device_groups}
    )
    assert resp_type in (Response.SUCCESS.name, Response.WARNING.name), payload

    resp_type, payload = client.command(Command.START_STREAMING, {})
    assert resp_type == Response.SUCCESS.name, payload

    received = 0
    timeouts = 0
    started = time.monotonic()
    deadline = started + 3.0
    while time.monotonic() < deadline:
        try:
            data, _sources = client.recv_data_chunk(timeout=1.0)
        except Exception:
            timeouts += 1
            continue
        received += 1
        assert np.all(np.isfinite(data)), "non-finite samples reached the client"
    elapsed = time.monotonic() - started

    client.command(Command.STOP_STREAMING)

    # DummyRfWorker emits one buffer per chunk_duration * SAVE_BUFFER_SCALE,
    # mirroring the USRP receive worker's 20-packet buffering. Derive the
    # expected rate from the config rather than assuming one chunk per
    # chunk_duration.
    from bioview_server.device.dummy.rf_worker import SAVE_BUFFER_SCALE

    emit_period = group_cfg["chunk_duration"] * SAVE_BUFFER_SCALE
    expected = elapsed / emit_period

    with capsys.disabled():
        print(
            f"\n  sustained stream: {received} chunks in {elapsed:.1f} s "
            f"({received / elapsed:.1f}/s, expected ~{expected / elapsed:.1f}/s, "
            f"{timeouts} stalls)"
        )

    assert (
        received >= 0.6 * expected
    ), f"only {received} chunks in {elapsed:.1f} s; expected about {expected:.0f}"
    assert timeouts == 0, f"{timeouts} stalls longer than 1 s while streaming"
