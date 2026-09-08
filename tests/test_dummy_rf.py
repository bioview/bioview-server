"""Tests for dummy RF MIMO / DPIC simulation."""

import time
from pathlib import Path

import numpy as np
from bioview_common.datatypes.configuration.usrp_channel_map import (
    DpicPair,
    resolve_channel_map,
)
from bioview_common.signal_schemes.cw import CwScheme
from bioview_common.signal_schemes.dpic import DpicBalancer, DpicChannel

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
        # Inject Tx must share the measure Tx's IF (100 kHz) -- an injection on
        # a different IF is removed by the receive band-pass and can never null
        # the direct path.
        "MyB210_7": {
            "tx_channels": [0],
            "rx_channels": [0, 1],
            "if_freq": [100e3],
        },
    }
    channel_map = {
        "layout": "hybrid_mimo",
        "mimo": {"tx_global": [0, 1], "rx_global": [0, 1]},
        "dpic": [{"inject_tx": 2, "measure_tx": 0, "measure_rx": 0}],
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
        if_freq=[100e3],
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


def test_dpic_pair_measure_rx_defaults_and_overrides():
    """measure_rx is an Rx index; it must not be inferred from measure_tx."""
    assert DpicPair(inject_tx=2, measure_tx=0).target_rx == 0
    assert DpicPair(inject_tx=2, measure_tx=1, measure_rx=0).target_rx == 0

    _, _, pairs = resolve_channel_map(
        "grp",
        {
            "layout": "hybrid_mimo",
            "mimo": {"tx_global": [0, 1], "rx_global": [0, 1]},
            "dpic": [{"inject_tx": 2, "measure_tx": 1, "measure_rx": 0}],
        },
        {
            "A": {"tx_channels": [0, 1], "rx_channels": [0, 1]},
            "B": {"tx_channels": [0], "rx_channels": []},
        },
    )
    assert pairs[0].measure_rx == 0
    assert pairs[0].target_rx == 0


def _channel(schemes, pair, read_metric, **kw):
    return DpicChannel(
        inject_tx=pair.inject_tx,
        measure_tx=pair.measure_tx,
        measure_rx=pair.target_rx,
        set_phase=lambda v: schemes["MyB210_7"].tx_phase_deg.__setitem__(0, v),
        set_amplitude=lambda v: schemes["MyB210_7"].tx_amplitude.__setitem__(0, v),
        read_metric=read_metric,
        start_phase_deg=schemes["MyB210_7"].tx_phase_deg[0],
        start_amplitude=schemes["MyB210_7"].tx_amplitude[0],
        **kw,
    )


def test_dpic_balancer_finds_minimum_on_channel_model():
    model, schemes, global_tx_to_device, dpic_pairs = _build_rf_context()

    def read_metric():
        return _measure_rx_power(
            model,
            schemes,
            global_tx_to_device,
            schemes["MyB210_7"].tx_phase_deg[0],
            schemes["MyB210_7"].tx_amplitude[0],
        )

    balancer = DpicBalancer(time_budget_s=30.0)
    result = balancer.balance(_channel(schemes, dpic_pairs[0], read_metric))

    baseline = _measure_rx_power(model, schemes, global_tx_to_device, 0.0, 0.0)
    assert result.converged
    assert result.min_metric < baseline
    assert result.best_amplitude > 0.0


def test_dpic_balancer_restores_settings_when_metric_unavailable():
    """A silent measurement path must not leave the injection switched off.

    The original search seeded its incumbent with +inf and amplitude 0, so a
    run where every read returned None ended with the injection amplitude at 0.
    """
    _model, schemes, _map, dpic_pairs = _build_rf_context()
    schemes["MyB210_7"].tx_phase_deg[0] = 137.0
    schemes["MyB210_7"].tx_amplitude[0] = 0.6

    balancer = DpicBalancer(
        coarse_phase_step_deg=30.0,
        phase_step_deg=10.0,
        coarse_amp_step=0.2,
        amp_step=0.1,
    )
    result = balancer.balance(_channel(schemes, dpic_pairs[0], lambda: None))

    assert not result.converged
    assert result.best_phase_deg == 137.0
    assert result.best_amplitude == 0.6
    assert schemes["MyB210_7"].tx_amplitude[0] == 0.6


def test_dpic_grid_is_cheaper_than_a_flat_sweep():
    """Coarse-to-fine must not cost a full-resolution sweep of the whole range."""
    _model, schemes, _map, dpic_pairs = _build_rf_context()
    calls = []

    def read_metric():
        calls.append(1)
        return abs(
            0.4 * np.exp(1j * 2.1)
            + 0.5
            * schemes["MyB210_7"].tx_amplitude[0]
            * np.exp(1j * np.deg2rad(schemes["MyB210_7"].tx_phase_deg[0]))
        )

    balancer = DpicBalancer(time_budget_s=60.0)
    result = balancer.balance(_channel(schemes, dpic_pairs[0], read_metric))

    assert result.converged
    assert result.method == "grid"
    # A flat 0.2 deg sweep alone would be 1800 points.
    assert len(calls) == 241
    assert abs(result.best_phase_deg - (np.rad2deg(2.1) + 180.0) % 360.0) < 0.3


def test_dpic_grid_resolves_a_minimum_off_the_coarse_lattice():
    """The fine pass must cover +/- a full coarse step, either side of the winner."""
    _model, schemes, _map, dpic_pairs = _build_rf_context()
    # 3 deg off the 6 deg coarse lattice, so the coarse winner is on one side.
    target_phase, target_amp = 123.0, 0.427

    def read_metric():
        return abs(
            target_amp * np.exp(1j * np.deg2rad(target_phase))
            - schemes["MyB210_7"].tx_amplitude[0]
            * np.exp(1j * np.deg2rad(schemes["MyB210_7"].tx_phase_deg[0]))
        )

    balancer = DpicBalancer(time_budget_s=60.0)
    result = balancer.balance(_channel(schemes, dpic_pairs[0], read_metric))

    assert result.converged
    assert abs(result.best_phase_deg - target_phase) <= 0.2
    assert abs(result.best_amplitude - target_amp) <= 0.001


def test_dpic_time_budget_keeps_the_best_point_seen():
    """A sweep cut short must not restore the start point or leave amplitude 0."""
    _model, schemes, _map, dpic_pairs = _build_rf_context()
    schemes["MyB210_7"].tx_phase_deg[0] = 137.0
    schemes["MyB210_7"].tx_amplitude[0] = 0.6

    # Budget expires partway through the first coarse sweep.
    balancer = DpicBalancer(time_budget_s=0.05)
    result = balancer.balance(
        _channel(
            schemes,
            dpic_pairs[0],
            lambda: 1.0 - schemes["MyB210_7"].tx_phase_deg[0] / 1e4,
            wait_settle=lambda _s: time.sleep(0.005),
        )
    )

    assert result.converged
    assert result.num_measurements < 241
    assert schemes["MyB210_7"].tx_amplitude[0] == result.best_amplitude


def test_dummy_dpic_config_file_exists():
    assert DUMMY_DPIC_CFG.is_file()
