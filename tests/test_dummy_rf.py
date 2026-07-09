"""Tests for dummy RF MIMO / DPIC simulation."""

from pathlib import Path

import numpy as np

from bioview_common.datatypes.configuration.usrp_channel_map import (
    DpicPair,
    resolve_channel_map,
)
from bioview_common.signal_schemes.cw import CwScheme
from bioview_common.signal_schemes.dpic import DpicBalancer
from bioview_server.device.dummy.rf_simulation import MimoChannelModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMMY_DPIC_CFG = REPO_ROOT / "dummy_dpic_2x2_mimo_cfg.json"


def _build_rf_context():
    hardware = {
        "MyB210_4": {
            "tx_channels": [0, 1],
            "rx_channels": [0, 1],
            "if_freq": [100e3, 110e3],
        },
        "MyB210_7": {
            "tx_channels": [0],
            "rx_channels": [0, 1],
            "if_freq": [120e3],
        },
    }
    channel_map = {
        "layout": "hybrid_mimo",
        "mimo": {"tx_global": [0, 1], "rx_global": [0, 1]},
        "dpic": [{"inject_tx": 2, "measure_tx": 0}],
    }
    _, registry, dpic_pairs = resolve_channel_map("grp", channel_map, hardware)

    samp_rate = 1e6
    scheme_a = CwScheme(
        samp_rate,
        if_freq=[100e3, 110e3],
        tx_amplitude=[1.0, 1.0],
        tx_phase_deg=[0.0, 0.0],
    )
    scheme_b = CwScheme(
        samp_rate,
        if_freq=[120e3],
        tx_amplitude=[1.0],
        tx_phase_deg=[0.0],
    )
    schemes = {"MyB210_4": scheme_a, "MyB210_7": scheme_b}
    global_tx_to_device = {0: ("MyB210_4", 0), 1: ("MyB210_4", 1), 2: ("MyB210_7", 0)}

    model = MimoChannelModel(
        samp_rate=samp_rate,
        if_freq=list(registry.tx_if_freq),
        dpic_pairs=dpic_pairs,
        num_rx=registry.num_rx,
        noise_std=0.0,
    )
    return model, schemes, global_tx_to_device, dpic_pairs


def _measure_rx_power(model, schemes, global_tx_to_device, inject_phase, inject_amp):
    schemes["MyB210_7"].tx_phase_deg[0] = inject_phase
    schemes["MyB210_7"].tx_amplitude[0] = inject_amp

    n_samples = 4000
    tx_all = np.zeros((3, n_samples), dtype=np.complex64)
    wave_a = schemes["MyB210_4"].generate(n_samples, 0)
    wave_b = schemes["MyB210_7"].generate(n_samples, 0)
    tx_all[0] = wave_a[0]
    tx_all[1] = wave_a[1]
    tx_all[2] = wave_b[0]

    rx = model.synthesize(tx_all, schemes, global_tx_to_device, 0, n_samples)
    return float(np.mean(np.abs(rx[0]) ** 2))


def test_dpic_channel_model_has_cancellable_leakage():
    model, schemes, global_tx_to_device, _ = _build_rf_context()

    baseline = _measure_rx_power(model, schemes, global_tx_to_device, 0.0, 0.0)
    cancelled = _measure_rx_power(model, schemes, global_tx_to_device, 180.0, 1.0)

    assert cancelled < baseline * 0.5


def test_dpic_balancer_finds_minimum_on_channel_model():
    model, schemes, global_tx_to_device, dpic_pairs = _build_rf_context()
    pair = dpic_pairs[0]

    def set_phase(_inject_tx, phase_deg):
        schemes["MyB210_7"].tx_phase_deg[0] = phase_deg

    def set_amplitude(_inject_tx, amp):
        schemes["MyB210_7"].tx_amplitude[0] = amp

    def read_metric():
        return _measure_rx_power(
            model, schemes, global_tx_to_device, schemes["MyB210_7"].tx_phase_deg[0], schemes["MyB210_7"].tx_amplitude[0]
        )

    balancer = DpicBalancer(phase_step_deg=10.0, amp_step=0.1, settle_time_s=0.0)
    result = balancer.run_pair(
        pair.inject_tx,
        pair.measure_tx,
        set_phase,
        set_amplitude,
        read_metric,
        lambda: None,
    )

    assert result.min_metric < _measure_rx_power(model, schemes, global_tx_to_device, 0.0, 0.0)


def test_dummy_dpic_config_file_exists():
    assert DUMMY_DPIC_CFG.is_file()
