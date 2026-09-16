"""Settings the UI accepted and the radio never took, and the buffer beneath."""

import copy
import multiprocessing as mp
import queue
import types

import numpy as np
import pytest
from bioview_common import DataSource
from bioview_common.signal_schemes import scheme_from_config

from bioview_server.device.usrp import backend as backend_mod
from bioview_server.device.usrp.backend import USRPBackend
from bioview_server.device.usrp.process import ProcessWorker
from bioview_server.device.usrp.transmit import MAX_CYCLIC_SAMPLES, TransmitWorker


SAMP_RATE = 1e6

BASE_CONFIG = {
    "carrier_freq": 900e6,
    "samp_rate": SAMP_RATE,
    "hardware": {
        "A": {
            "tx_channels": [0, 1],
            "rx_channels": [0, 1],
            "if_freq": [100e3, 110e3],
            "if_filter_bw": 5000,
        }
    },
    "channel_map": {"layout": "full_nxn", "dpic": []},
}


class _FakeStreamer:
    def get_max_num_samps(self):
        return 2000


def _transmit_worker(**scheme_overrides):
    cfg = {
        "if_freq": [100e3, 110e3],
        "tx_amplitude": [1.0, 1.0],
        "tx_phase": [0.0, 0.0],
        "signal_scheme": "cw",
        "calibration": {"enabled": False},
        **scheme_overrides,
    }
    scheme = scheme_from_config(SAMP_RATE, 2, cfg, global_tx_offset=0)
    return TransmitWorker(
        usrp=None,
        tx_gain=[30, 30],
        tx_channels=[0, 1],
        samp_rate=SAMP_RATE,
        tx_streamer=_FakeStreamer(),
        scheme=scheme,
        cmd_queue=queue.Queue(),
    )


def _peaks(worker, n=2000):
    chunk = worker._generate_chunk(n)
    return [float(np.abs(row).max()) for row in chunk]


def test_cw_transmits_from_a_cyclic_buffer():
    worker = _transmit_worker()
    assert worker._use_cyclic


def test_lowering_tx_amplitude_lowers_the_samples_sent():
    worker = _transmit_worker()
    assert _peaks(worker) == pytest.approx([1.0, 1.0], abs=1e-5)

    worker._apply_command("tx_amplitude", [0.25, 0.25])

    assert _peaks(worker) == pytest.approx([0.25, 0.25], abs=1e-5)


def test_a_per_channel_amplitude_moves_only_that_channel():
    worker = _transmit_worker()

    worker._apply_command("global_tx_amplitude", (1, 0.5))

    assert _peaks(worker) == pytest.approx([1.0, 0.5], abs=1e-5)


def test_a_per_channel_phase_moves_only_that_channel():
    worker = _transmit_worker()
    before = worker._generate_chunk(64)[1][0]

    worker._apply_command("global_tx_phase", (1, 90.0))
    worker._sample_idx = 0
    after = worker._generate_chunk(64)[1][0]

    assert np.angle(after) - np.angle(before) == pytest.approx(np.pi / 2, abs=1e-5)


def test_an_analog_gain_change_leaves_the_digital_waveform_alone():
    worker = _transmit_worker()
    worker.usrp = types.SimpleNamespace(set_tx_gain=lambda *_: None)
    buffer_before = worker.tx_waveform

    worker._apply_command("global_tx_gain", (1, 40.0))

    assert worker.tx_waveform is buffer_before
    assert worker.tx_gain[1] == 40.0


class _FakeUSRP:
    """Records what was tuned, and answers the read-back."""

    def __init__(self):
        self.rx_freq = {}
        self.tx_freq = {}

    def set_rx_freq(self, tune, chan):
        self.rx_freq[chan] = float(tune.target_freq)

    def set_tx_freq(self, tune, chan):
        self.tx_freq[chan] = float(tune.target_freq)

    def get_rx_freq(self, chan):
        return self.rx_freq[chan]


@pytest.fixture
def fake_uhd(monkeypatch):
    """A stand-in for the driver: CI has no UHD, and no radio to tune."""
    fake = types.SimpleNamespace(
        types=types.SimpleNamespace(
            TuneRequest=lambda freq: types.SimpleNamespace(target_freq=freq)
        )
    )
    monkeypatch.setattr(backend_mod, "uhd", fake)
    return fake


def _backend():
    return USRPBackend(
        group_id="grp",
        samp_rate=SAMP_RATE,
        devices={},
        group_config=copy.deepcopy(BASE_CONFIG),
        response_queue=mp.Queue(),
        data_output_queue=mp.Queue(),
    )


def _connected_backend():
    be = _backend()
    radio = _FakeUSRP()
    be.usrp_handlers["A"] = radio
    return be, radio


def test_a_carrier_edit_retunes_both_chains_of_every_radio(fake_uhd):
    be, radio = _connected_backend()

    be._queue_param_update({"carrier_freq": 915e6})

    assert radio.rx_freq == {0: 915e6, 1: 915e6}
    assert radio.tx_freq == {0: 915e6, 1: 915e6}


def test_the_new_carrier_survives_a_reconnect(fake_uhd):
    be, _radio = _connected_backend()

    be._queue_param_update({"carrier_freq": 915e6})

    assert be.group_config["carrier_freq"] == 915e6
    assert be.hardware["A"]["carrier_freq"] == 915e6
    assert be.usrp_configs["A"].get_param("carrier_freq") == 915e6


def test_a_carrier_edit_before_connect_is_kept_for_the_open(fake_uhd):
    be = _backend()

    be._queue_param_update({"carrier_freq": 433e6})

    assert be.usrp_configs["A"].get_param("carrier_freq") == 433e6


def test_the_carrier_is_refused_while_the_group_is_streaming(fake_uhd):
    be, radio = _connected_backend()
    be._streaming.set()

    be._queue_param_update({"carrier_freq": 915e6})

    assert radio.rx_freq == {}
    assert be.group_config["carrier_freq"] == 900e6


def test_a_carrier_that_is_not_a_number_changes_nothing(fake_uhd):
    be, radio = _connected_backend()

    be._queue_param_update({"carrier_freq": "wideband please"})

    assert radio.tx_freq == {}
    assert be.group_config["carrier_freq"] == 900e6


def test_the_carrier_is_not_forwarded_to_a_worker_queue(fake_uhd):
    be, _radio = _connected_backend()
    be.rx_command_queue["A"] = queue.Queue()
    be.tx_command_queue["A"] = queue.Queue()

    be._queue_param_update({"carrier_freq": 915e6})

    assert be.rx_command_queue["A"].empty()
    assert be.tx_command_queue["A"].empty()


def test_the_buffer_holds_a_whole_number_of_cycles():
    worker = _transmit_worker()
    period = worker.scheme.cycle_length()

    assert worker.tx_waveform.shape[1] % period == 0


def test_the_buffer_is_still_at_least_a_send_buffer_and_twenty_cycles():
    worker = _transmit_worker()
    period = worker.scheme.cycle_length()

    length = worker.tx_waveform.shape[1]
    assert length >= worker.tx_buffer_size
    assert length >= 20 * period


def test_the_tone_is_continuous_across_the_wrap():
    worker = _transmit_worker(if_freq=[110e3, 110e3])
    length = worker.tx_waveform.shape[1]
    worker._sample_idx = length - 64

    span = worker._generate_chunk(128)[0]
    steps = np.diff(np.angle(span))
    steps = (steps + np.pi) % (2 * np.pi) - np.pi

    assert steps.std() == pytest.approx(0.0, abs=1e-5)


def test_a_waveform_too_large_to_hold_is_generated_live_instead():
    worker = _transmit_worker(if_freq=[100e3, 100e3])
    worker.scheme.set_samp_rate(10e6)
    worker.scheme.if_freq = [100_001.0, 100_001.0]
    assert worker.scheme.cycle_length() > MAX_CYCLIC_SAMPLES

    worker._refresh_cyclic()

    assert not worker._use_cyclic
    assert _peaks(worker) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_live_generation_is_also_continuous_across_chunks():
    worker = _transmit_worker(calibration={"enabled": True})
    assert not worker._use_cyclic

    span = np.concatenate([worker._generate_chunk(64)[0] for _ in range(4)])
    steps = np.diff(np.angle(span))
    steps = (steps + np.pi) % (2 * np.pi) - np.pi

    assert steps.std() == pytest.approx(0.0, abs=1e-5)


def test_a_rate_change_re_rates_the_transmit_waveform():
    worker = _transmit_worker(if_freq=[100e3, 100e3])
    assert worker.scheme.cycle_length() == 10

    worker.scheme.set_samp_rate(2e6)
    worker.set_samp_rate(2e6)

    assert worker.samp_rate == 2e6
    assert worker.scheme.cycle_length() == 20
    assert worker.tx_waveform.shape[1] % 20 == 0


def _process_worker(samp_rate=SAMP_RATE, ifs=(100e3, 110e3)):
    sources = []
    for channel, tx_idx in enumerate(range(len(ifs))):
        source = DataSource(group_id="grp", channel=channel, label=f"Tx{tx_idx}")
        source.tx_idx = tx_idx
        source.rx_idx = 0
        source.component = "amplitude"
        sources.append(source)
    return ProcessWorker(
        data_sources=sources,
        cal_ref_sources=[],
        samp_rate=samp_rate,
        channel_ifs=list(ifs),
        if_filter_bw=[5e3] * len(ifs),
        rx_queues={},
        rx_device_order=["A"],
        schemes_by_device={},
        global_tx_to_device={i: ("A", i) for i in range(len(ifs))},
    )


def _passes(worker, tx_idx, freq):
    """Gain of one channel's band-pass at a frequency, in the worker's rate."""
    from scipy import signal

    _w, h = signal.sosfreqz(
        worker.if_filts[tx_idx], worN=[2 * np.pi * freq / worker.samp_rate]
    )
    return float(abs(h[0]))


def test_the_demodulation_filters_follow_the_rate():
    worker = _process_worker()
    assert _passes(worker, 0, 100e3) > 0.5

    worker.set_samp_rate(2e6)

    assert _passes(worker, 0, 100e3) > 0.5
    assert _passes(worker, 0, 200e3) < 0.1


def test_a_rate_change_drops_demodulator_state():
    worker = _process_worker()
    worker.mimo_sources[0].accumulated_phase = 1.234
    worker.mimo_sources[0].filter_state = "stale"

    worker.set_samp_rate(2e6)

    assert worker.mimo_sources[0].accumulated_phase == 0.0
    assert worker.mimo_sources[0].filter_state is None


def test_a_rate_that_puts_the_ifs_past_nyquist_is_refused_intact():
    worker = _process_worker()
    designed = list(worker.if_filts)

    with pytest.raises(ValueError):
        worker.set_samp_rate(100e3)

    assert worker.samp_rate == SAMP_RATE
    assert all(
        np.array_equal(a, b) for a, b in zip(worker.if_filts, designed, strict=True)
    )


class _RateRadio(_FakeUSRP):
    """A radio that quantises the rate, as a real front end does."""

    def __init__(self, quantum=None):
        super().__init__()
        self.quantum = quantum
        self.rx_rate = {}
        self.tx_rate = {}

    def _coerce(self, rate):
        if self.quantum is None:
            return float(rate)
        return float(round(rate / self.quantum) * self.quantum)

    def set_rx_rate(self, rate, chan):
        self.rx_rate[chan] = self._coerce(rate)

    def set_tx_rate(self, rate, chan):
        self.tx_rate[chan] = self._coerce(rate)

    def get_rx_rate(self, chan=0):
        return self.rx_rate[chan]


def _rate_backend(quantum=None):
    be = _backend()
    radio = _RateRadio(quantum)
    be.usrp_handlers["A"] = radio
    be.process_worker = _process_worker()
    return be, radio


def test_a_rate_edit_reaches_both_chains_of_every_radio(fake_uhd):
    be, radio = _rate_backend()

    be._queue_param_update({"samp_rate": 2e6})

    assert radio.rx_rate == {0: 2e6, 1: 2e6}
    assert radio.tx_rate == {0: 2e6, 1: 2e6}


def test_the_rate_the_hardware_landed_on_is_the_one_everything_uses(fake_uhd):
    be, _radio = _rate_backend(quantum=250e3)

    be._queue_param_update({"samp_rate": 1.6e6})

    assert be.samp_rate == 1.5e6
    assert be.group_config["samp_rate"] == 1.5e6
    assert be.process_worker.samp_rate == 1.5e6


def test_a_rate_edit_re_rates_the_schemes_and_the_plot_axis(fake_uhd):
    be, _radio = _rate_backend()
    before = be.get_display_frequency()

    be._queue_param_update({"samp_rate": 2e6})

    assert be.schemes_by_device["A"].samp_rate == 2e6
    assert be.get_display_frequency() == pytest.approx(2 * before)
    assert all(src.disp_freq == be.get_display_frequency() for src in be.mimo_sources)


def test_the_new_rate_survives_a_reconnect(fake_uhd):
    be, _radio = _rate_backend()

    be._queue_param_update({"samp_rate": 2e6})

    assert be.group_config["samp_rate"] == 2e6
    assert be.hardware["A"]["samp_rate"] == 2e6
    assert be.usrp_configs["A"].get_param("samp_rate") == 2e6


def test_the_rate_is_refused_while_the_group_is_streaming(fake_uhd):
    be, radio = _rate_backend()
    be._streaming.set()

    be._queue_param_update({"samp_rate": 2e6})

    assert radio.rx_rate == {}
    assert be.samp_rate == SAMP_RATE


def test_a_rate_the_pipeline_cannot_demodulate_puts_the_radio_back(fake_uhd):
    be, radio = _rate_backend()

    be._queue_param_update({"samp_rate": 100e3})

    assert be.samp_rate == SAMP_RATE
    assert be.group_config["samp_rate"] == SAMP_RATE
    assert be.process_worker.samp_rate == SAMP_RATE
    assert radio.rx_rate == {0: SAMP_RATE, 1: SAMP_RATE}


def test_a_rate_that_is_not_a_number_changes_nothing(fake_uhd):
    be, radio = _rate_backend()

    be._queue_param_update({"samp_rate": "as fast as it goes"})

    assert radio.rx_rate == {}
    assert be.samp_rate == SAMP_RATE


def test_a_rate_edit_before_connect_is_kept_for_the_open(fake_uhd):
    be = _backend()

    be._queue_param_update({"samp_rate": 5e6})

    assert be.samp_rate == 5e6
    assert be.usrp_configs["A"].get_param("samp_rate") == 5e6
