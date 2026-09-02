"""Demodulation correctness and real-time headroom.

``_process_chunk`` was rewritten from a per-save-window Python loop into whole-
chunk numpy. These tests pin the numerics against a literal port of the old
loop, so the rewrite cannot silently change what gets recorded, and guard the
throughput that made the rewrite necessary in the first place.
"""

import time

import numpy as np
import pytest
from bioview_common import DataSource, apply_filter
from bioview_common.signal_schemes import CwScheme
from bioview_common.signal_schemes.normalization import (
    differential_phase,
    normalized_amplitude,
)

from bioview_server.device.usrp.process import ProcessWorker


SAMP_RATE = 1e6
IF_HZ = 100e3
SAVE_DS = 10


def _make_worker(save_ds=SAVE_DS, save_iq=False):
    source = DataSource(group_id="g", channel=0, label="Tx1Rx1")
    source.tx_idx = 0
    source.rx_idx = 0
    scheme = CwScheme(SAMP_RATE, [IF_HZ], [1.0], [30.0])
    worker = ProcessWorker(
        data_sources={source},
        cal_ref_sources=[],
        samp_rate=SAMP_RATE,
        channel_ifs=[IF_HZ],
        if_filter_bw=[5e3],
        rx_queues={"d": None},
        rx_device_order=["d"],
        schemes_by_device={"d": scheme},
        global_tx_to_device={0: ("d", 0)},
        save_ds=save_ds,
        save_iq=save_iq,
    )
    return worker, source, scheme


def _reference_process_chunk(worker, data, source, filt, if_freq, scheme):
    """Per-window loop equivalent of the vectorized implementation.

    This is the pre-vectorization code with one deliberate change: it subtracts
    ``tx_phase_offset`` rather than ``tx_phase_at(center_idx)``. The original
    subtracted the running carrier phase that the downconversion had already
    removed, which is fixed separately and pinned by
    ``test_phase_channel_is_not_a_carrier_ramp``.
    """
    if len(data) == 0:
        return np.array([]), np.array([])

    filt_data, new_state = apply_filter(data, filt, zi=source.filter_state)
    source.filter_state = new_state

    phase_increment = 2 * np.pi * if_freq / worker.samp_rate
    phases = source.accumulated_phase + np.arange(len(filt_data)) * phase_increment
    baseband = filt_data * np.exp(-1j * phases)
    source.accumulated_phase = phases[-1] + phase_increment
    source.accumulated_sample_idx += len(filt_data)

    step = worker.save_ds
    end_idx = len(baseband) - step + 1
    num_windows = (end_idx + step - 1) // step
    if num_windows <= 0:
        return np.array([]), np.array([])

    start_indices = np.arange(0, end_idx, step)
    windows = baseband[start_indices[:, np.newaxis] + np.arange(step)]

    tx_amp = scheme.get_tx_amplitude(0)
    if worker.save_iq:
        return np.mean(np.real(windows), axis=1), np.mean(np.imag(windows), axis=1)

    amps, phases_out = [], []
    for w_i, win in enumerate(windows):
        center = (
            source.accumulated_sample_idx
            - len(baseband)
            + int(start_indices[w_i] + step // 2)
        )
        del center
        amps.append(normalized_amplitude(win, tx_amp))
        ph, source.prev_phase = differential_phase(
            win, scheme.tx_phase_offset(0), source.prev_phase
        )
        phases_out.append(ph)
    return np.array(amps), np.array(phases_out)


def _test_signal(n_total, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n_total) / SAMP_RATE
    return (
        0.7
        * np.exp(1j * (2 * np.pi * IF_HZ * t + np.deg2rad(30.0)))
        * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
        + 0.01 * (rng.standard_normal(n_total) + 1j * rng.standard_normal(n_total))
    ).astype(np.complex64)


@pytest.mark.parametrize("save_iq", [False, True])
@pytest.mark.parametrize("save_ds", [1, 10, 37])
def test_vectorized_demod_matches_reference_loop(save_iq, save_ds):
    n, n_chunks = 20000, 5
    signal = _test_signal(n * n_chunks)

    new_w, new_src, new_scheme = _make_worker(save_ds=save_ds, save_iq=save_iq)
    ref_w, ref_src, ref_scheme = _make_worker(save_ds=save_ds, save_iq=save_iq)

    for c in range(n_chunks):
        chunk = signal[c * n : (c + 1) * n]
        first, second, _ = new_w._process_chunk(
            chunk, new_src, new_w.if_filts[0], IF_HZ, new_scheme
        )
        ref_first, ref_second = _reference_process_chunk(
            ref_w, chunk, ref_src, ref_w.if_filts[0], IF_HZ, ref_scheme
        )
        np.testing.assert_allclose(first, ref_first, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(second, ref_second, rtol=1e-9, atol=1e-10)


def test_window_count_matches_assemble_outputs_expectation():
    """_assemble_outputs sizes its arrays as buffer_len // save_ds."""
    for n in (999, 1000, 1001, 1009, 40000):
        worker, source, scheme = _make_worker()
        first, _second, _m = worker._process_chunk(
            _test_signal(n), source, worker.if_filts[0], IF_HZ, scheme
        )
        assert len(first) == n // SAVE_DS, n


def _continuous_tone(n_total, amplitude=0.5, tx_phase_deg=30.0, channel_phase=0.0):
    t = np.arange(n_total) / SAMP_RATE
    return (
        amplitude
        * np.exp(1j * (2 * np.pi * IF_HZ * t + np.deg2rad(tx_phase_deg) + channel_phase))
    ).astype(np.complex64)


def test_demod_recovers_amplitude_of_a_clean_tone():
    worker, source, scheme = _make_worker()
    n = 40000
    tone = _continuous_tone(2 * n)
    worker._process_chunk(tone[:n], source, worker.if_filts[0], IF_HZ, scheme)
    amp, _phase, _m = worker._process_chunk(
        tone[n:], source, worker.if_filts[0], IF_HZ, scheme
    )
    assert abs(np.median(amp) - 0.5) < 0.02


def test_phase_channel_is_not_a_carrier_ramp():
    """Regression: the recorded phase used to be a pure 2*pi*f_if*save_ds/fs ramp.

    The demodulator subtracted ``tx_phase_at(center_idx)``, which includes the
    running carrier phase -- but the downconversion had already removed it, so
    every recorded phase sample stepped by a full carrier increment.
    """
    worker, source, scheme = _make_worker()
    n = 40000
    tone = _continuous_tone(2 * n)
    worker._process_chunk(tone[:n], source, worker.if_filts[0], IF_HZ, scheme)
    _amp, phase, _m = worker._process_chunk(
        tone[n:], source, worker.if_filts[0], IF_HZ, scheme
    )

    ramp_per_window = 2 * np.pi * IF_HZ / SAMP_RATE * SAVE_DS
    observed = np.diff(phase)
    assert np.max(np.abs(observed)) < 1e-6, "phase should be flat on a static channel"
    assert abs(np.mean(observed)) < ramp_per_window * 1e-6


def test_phase_channel_tracks_a_channel_phase_step():
    worker, source, scheme = _make_worker()
    n = 40000
    step_rad = 0.5
    t = np.arange(4 * n) / SAMP_RATE
    channel = np.where(t < t[-1] / 2, 0.0, step_rad)
    tone = _continuous_tone(4 * n, channel_phase=channel)

    phases = []
    for c in range(4):
        _amp, phase, _m = worker._process_chunk(
            tone[c * n : (c + 1) * n], source, worker.if_filts[0], IF_HZ, scheme
        )
        phases.append(phase)
    phase = np.concatenate(phases)

    before = np.median(phase[: len(phase) // 4])
    after = np.median(phase[3 * len(phase) // 4 :])
    assert abs((after - before) - step_rad) < 1e-3


def test_phase_channel_cancels_the_programmed_tx_phase():
    """Changing tx_phase_deg must not move the measured channel phase."""
    results = []
    for tx_phase_deg in (0.0, 75.0):
        worker, source, scheme = _make_worker()
        scheme.update_param("tx_phase", [tx_phase_deg])
        n = 40000
        tone = _continuous_tone(2 * n, tx_phase_deg=tx_phase_deg)
        worker._process_chunk(tone[:n], source, worker.if_filts[0], IF_HZ, scheme)
        _amp, phase, _m = worker._process_chunk(
            tone[n:], source, worker.if_filts[0], IF_HZ, scheme
        )
        results.append(np.median(phase))
    assert abs(results[0] - results[1]) < 1e-6


def test_metric_is_magnitude_based_under_save_iq():
    """The DPIC metric must not become mean(Re{.}) when save_iq is on."""
    for save_iq in (False, True):
        worker, source, scheme = _make_worker(save_iq=save_iq)
        _f, _s, metric = worker._process_chunk(
            _test_signal(20000), source, worker.if_filts[0], IF_HZ, scheme
        )
        magnitude, phasor = metric
        assert magnitude > 0
        assert isinstance(phasor, complex)


def test_demod_keeps_up_with_real_time():
    """Guard the throughput: the loop version ran at ~250% of real time."""
    n, n_chunks = 40000, 30
    chunk_seconds = n / SAMP_RATE
    worker, source, scheme = _make_worker()
    signal = _test_signal(n)

    worker._process_chunk(signal, source, worker.if_filts[0], IF_HZ, scheme)
    start = time.perf_counter()
    for _ in range(n_chunks):
        worker._process_chunk(signal, source, worker.if_filts[0], IF_HZ, scheme)
    per_chunk = (time.perf_counter() - start) / n_chunks

    budget = per_chunk / chunk_seconds
    assert budget < 0.5, (
        f"demod uses {budget:.0%} of real time per source; it must leave room "
        "for multiple sources plus the save and display paths"
    )
