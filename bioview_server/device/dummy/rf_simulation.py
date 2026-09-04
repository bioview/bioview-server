"""Virtual MIMO channel model for dummy RF simulation."""

from __future__ import annotations

import numpy as np
from bioview_common.datatypes.configuration.usrp_channel_map import DpicPair
from bioview_common.signal_schemes.base import SignalScheme


class MimoChannelModel:
    """Synthesize complex RX buffers from global TX waveforms.

    DPIC pairs add a controllable leakage term at the measure-Tx IF on the
    target Rx so grid search can find a minimum.
    """

    def __init__(
        self,
        samp_rate: float,
        if_freq: list[float],
        dpic_pairs: list[DpicPair],
        num_rx: int,
        noise_std: float = 0.01,
        cross_coupling: float = 0.08,
        on_axis_gain: float = 0.4,
        direct_leak: float = 0.4,
        dpic_coupling: float = 0.4,
    ):
        self.samp_rate = float(samp_rate)
        self.if_freq = list(if_freq)
        self.dpic_pairs = list(dpic_pairs)
        self.noise_std = float(noise_std)
        self.direct_leak = float(direct_leak)
        self.dpic_coupling = float(dpic_coupling)

        self.num_tx = len(if_freq)
        self.num_rx = max(int(num_rx), 1)
        self.H = np.full((self.num_tx, self.num_rx), cross_coupling, dtype=np.float64)
        for idx in range(min(self.num_tx, self.num_rx)):
            self.H[idx, idx] = on_axis_gain

    def synthesize(
        self,
        tx_waveforms: np.ndarray,
        schemes_by_device: dict[str, SignalScheme],
        global_tx_to_device: dict[int, tuple],
        start_sample: int,
        n_samples: int,
    ) -> np.ndarray:
        rx = np.zeros((self.num_rx, n_samples), dtype=np.complex64)
        for tx_idx in range(tx_waveforms.shape[0]):
            for rx_idx in range(self.num_rx):
                rx[rx_idx] += self.H[tx_idx, rx_idx] * tx_waveforms[tx_idx]

        for pair in self.dpic_pairs:
            measure_tx = pair.measure_tx
            inject_tx = pair.inject_tx
            measure_rx = pair.target_rx
            if measure_tx >= tx_waveforms.shape[0]:
                continue

            s_meas = tx_waveforms[measure_tx]
            rx[measure_rx] += self.direct_leak * s_meas

            # The inject Tx's own samples: using the measure Tx's would hide
            # that a different IF cannot cancel once band-passed.
            if inject_tx >= tx_waveforms.shape[0]:
                continue
            rx[measure_rx] += (self.dpic_coupling * tx_waveforms[inject_tx]).astype(
                np.complex64
            )

        if self.noise_std > 0:
            rx += (
                np.random.normal(0, self.noise_std, rx.shape)
                + 1j * np.random.normal(0, self.noise_std, rx.shape)
            ).astype(np.complex64)

        return rx
