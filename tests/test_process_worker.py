"""Demodulation correctness and real-time headroom."""

import time

import numpy as np
import pytest
from bioview_common import DataSource, apply_filter
from bioview_common.signal_schemes import CwScheme
from bioview_common.signal_schemes.normalization import normalized_amplitude

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
    """Per-window loop equivalent of the vectorized implementation."""
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

    amps = [normalized_amplitude(win, tx_amp) for win in windows]
    phases_out = np.angle(windows.mean(axis=1) * np.exp(-1j * scheme.tx_phase_offset(0)))
    return np.array(amps), phases_out


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
        np.testing.assert_allclose(first, ref_first, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(second, ref_second, rtol=2e-5, atol=1e-5)


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
    """Regression: the recorded phase used to be a pure 2*pi*f_if*save_ds/fs ramp."""
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


def test_phase_stays_wrapped_on_a_drifting_channel():
    """A residual frequency offset must not run the phase off to infinity.

    The old estimator unwrapped across chunk boundaries, so a fraction of a Hz
    of LO offset turned the recorded phase into an ever-growing ramp instead of
    a bounded angle.
    """
    worker, source, scheme = _make_worker()
    n = 40000
    offset_hz = 2.0
    t = np.arange(8 * n) / SAMP_RATE
    tone = (0.5 * np.exp(1j * 2 * np.pi * (IF_HZ + offset_hz) * t)).astype(np.complex64)

    seen = []
    for c in range(8):
        _amp, phase, _m = worker._process_chunk(
            tone[c * n : (c + 1) * n], source, worker.if_filts[0], IF_HZ, scheme
        )
        seen.append(phase)
    phase = np.concatenate(seen[1:])

    assert np.all(
        np.abs(phase) <= np.pi + 1e-6
    ), f"phase left (-pi, pi]: max |phase| was {np.max(np.abs(phase)):.3f}"


def test_phase_recovers_a_static_channel_phase():
    """arg(mean(window)) lands on the channel phase, within the filter's own shift.

    The band-pass contributes a few milliradians of its own at IF, which is why
    this is not held to the 1e-6 that the differential tests are.
    """
    worker, source, scheme = _make_worker()
    n = 40000
    tone = _continuous_tone(2 * n, channel_phase=0.4)
    worker._process_chunk(tone[:n], source, worker.if_filts[0], IF_HZ, scheme)
    _amp, phase, _m = worker._process_chunk(
        tone[n:], source, worker.if_filts[0], IF_HZ, scheme
    )
    assert abs(np.median(phase) - 0.4) < 1e-2


def test_phasor_ramp_is_shared_across_every_rx_on_one_tx():
    """The downconversion exponential is paid once per IF, not once per source."""
    worker, source, scheme = _make_worker()
    n = 20000
    for _ in range(3):
        worker._process_chunk(_test_signal(n), source, worker.if_filts[0], IF_HZ, scheme)
    assert list(worker._ramp_cache) == [(IF_HZ, n)]


def test_accumulated_phase_stays_bounded():
    """A long session must not spend mantissa on a runaway phase accumulator."""
    worker, source, scheme = _make_worker()
    for _ in range(50):
        worker._process_chunk(
            _test_signal(40000), source, worker.if_filts[0], IF_HZ, scheme
        )
    assert 0.0 <= source.accumulated_phase < 2 * np.pi


def test_metric_is_magnitude_based_under_save_iq():
    """The DPIC metric must not become mean(Re{.}) when save_iq is on."""
    for save_iq in (False, True):
        worker, source, scheme = _make_worker(save_iq=save_iq)
        _f, _s, metric = worker._process_chunk(
            _test_signal(20000), source, worker.if_filts[0], IF_HZ, scheme
        )
        assert isinstance(metric, float)
        assert metric > 0


def test_full_mimo_group_keeps_up_with_real_time():
    """A 4x4 group is 16 demodulated pairs and used to sit at 89% of real time.

    That left nothing for the save and display paths, so on a slow machine the
    Rx queue backed up and the receiver dropped buffers.
    """
    n_tx = n_rx = 4
    ifs = [IF_HZ + i * 10e3 for i in range(n_tx)]
    sources = set()
    for rx in range(n_rx):
        for tx in range(n_tx):
            src = DataSource(group_id="g", channel=rx * n_tx + tx, label=f"T{tx}R{rx}")
            src.tx_idx, src.rx_idx = tx, rx
            sources.add(src)
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
        save_ds=100,
    )

    n = 40000
    buffer = np.vstack([_test_signal(n, seed=r) for r in range(n_rx)])

    worker._process_mimo_chunk(buffer)
    start = time.perf_counter()
    for _ in range(10):
        worker._process_mimo_chunk(buffer)
    budget = ((time.perf_counter() - start) / 10) / (n / SAMP_RATE)

    assert budget < 0.5, (
        f"a 4x4 group uses {budget:.0%} of real time; it must leave room for "
        "the save and display paths and for the client sharing the machine"
    )


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


def _dual_component_worker(save_ds=SAVE_DS):
    """One Tx/Rx pair advertised as two rows: amplitude then phase."""
    amp = DataSource(group_id="g", channel=0, label="Tx1Rx1")
    amp.tx_idx = amp.rx_idx = 0
    amp.component = "amplitude"
    phase = DataSource(group_id="g", channel=1, label="Tx1Rx1_Phase")
    phase.tx_idx = phase.rx_idx = 0
    phase.component = "phase"
    scheme = CwScheme(SAMP_RATE, [IF_HZ], [1.0], [30.0])
    worker = ProcessWorker(
        data_sources={amp, phase},
        cal_ref_sources=[],
        samp_rate=SAMP_RATE,
        channel_ifs=[IF_HZ],
        if_filter_bw=[5e3],
        rx_queues={"d": None},
        rx_device_order=["d"],
        schemes_by_device={"d": scheme},
        global_tx_to_device={0: ("d", 0)},
        save_ds=save_ds,
    )
    return worker, amp, phase


def test_component_rows_match_the_single_row_pipeline():
    """Splitting a pair into two rows must not change either quantity."""
    n = 4 * SAVE_DS
    signal = _test_signal(n)
    buffer = signal.reshape(1, n)

    ref_worker, _src, _scheme = _make_worker()
    ref_worker.save_imaginary = True
    ref_results = ref_worker._process_mimo_chunk(buffer)
    ref_save, _ref_display = ref_worker._assemble_outputs(buffer, ref_results)
    assert ref_save.shape == (1, n // SAVE_DS, 2)

    worker, _amp, _phase = _dual_component_worker()
    results = worker._process_mimo_chunk(buffer)
    save_data, display_data = worker._assemble_outputs(buffer, results)

    assert save_data.shape == (2, n // SAVE_DS)
    # Both sides are float32 pipelines over separately allocated buffers, so
    # the window reduction can land an ULP apart; assert_allclose's float64
    # default tolerance is not the right yardstick for that.
    np.testing.assert_allclose(save_data[0], ref_save[0, :, 0], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(save_data[1], ref_save[0, :, 1], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(display_data, save_data)


def test_pair_is_demodulated_once_per_chunk():
    """Two rows share one demodulator, so the phase accumulator advances once."""
    n = 4 * SAVE_DS
    buffer = _test_signal(n).reshape(1, n)

    worker, amp, phase = _dual_component_worker()
    assert len(worker._demod_sources) == 1
    assert worker._per_source_components is True

    worker._process_mimo_chunk(buffer)
    ref_worker, ref_src, _scheme = _make_worker()
    ref_worker._process_mimo_chunk(buffer)

    assert amp.accumulated_sample_idx == n
    assert phase.accumulated_sample_idx == 0
    assert amp.accumulated_phase == pytest.approx(ref_src.accumulated_phase)
