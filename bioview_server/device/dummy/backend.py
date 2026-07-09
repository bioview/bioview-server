"""
Virtual "dummy" device backend.

Legacy mode synthesizes phase-shifted sine waves for pipeline testing.
RF simulation mode reuses the USRP signal-scheme / ProcessWorker path with a
virtual MIMO channel model so calibration and DPIC balance can be exercised
without hardware.
"""

from __future__ import annotations

import queue
import time
import multiprocessing as mp

from typing import Dict, List, Optional

import numpy as np

from bioview_common import DataSource, DeviceStatus, PausableWorker, log_print
from bioview_common.datatypes.configuration.usrp_channel_map import resolve_channel_map
from bioview_common.datatypes.configuration.hardware_params import (
    GLOBAL_TX_PARAMS,
    apply_global_tx_param_to_schemes,
)
from bioview_common.signal_schemes import DpicBalancer, scheme_from_config

from bioview_server.datatypes import Backend

from .rf_simulation import MimoChannelModel
from .rf_worker import DummyRfWorker


class SineWaveWorker(PausableWorker):
    """Legacy multi-channel sine generator for simple pipeline tests."""

    def __init__(
        self,
        samp_rate: float,
        num_channels: int,
        signal_freq: float,
        amplitude: float,
        noise_std: float,
        chunk_duration: float,
        display_queue: mp.Queue,
        logger=None,
    ):
        super().__init__(logger=logger)
        self.samp_rate = float(samp_rate)
        self.num_channels = int(num_channels)
        self.signal_freq = float(signal_freq)
        self.amplitude = float(amplitude)
        self.noise_std = float(noise_std)
        self.display_queue = display_queue

        self.chunk_size = max(1, int(round(self.samp_rate * float(chunk_duration))))
        self.chunk_duration = self.chunk_size / self.samp_rate
        self.phase_offsets = (
            2.0 * np.pi * np.arange(self.num_channels) / max(1, self.num_channels)
        )
        self._sample_idx = 0
        self._next_emit = None

    def work(self):
        if self.display_queue is None:
            return

        now = time.monotonic()
        if self._next_emit is None:
            self._next_emit = now

        n = np.arange(self._sample_idx, self._sample_idx + self.chunk_size)
        t = n / self.samp_rate
        angle = 2.0 * np.pi * self.signal_freq * t
        chunk = self.amplitude * np.sin(
            angle[np.newaxis, :] + self.phase_offsets[:, np.newaxis]
        )
        if self.noise_std > 0:
            chunk = chunk + np.random.normal(0.0, self.noise_std, size=chunk.shape)

        self._sample_idx += self.chunk_size

        try:
            self.display_queue.put_nowait(np.ascontiguousarray(chunk, dtype=float))
        except queue.Full:
            log_print(self.logger, "warning", "[DUMMY] Display queue full; dropping chunk")

        self._next_emit += self.chunk_duration
        sleep_for = self._next_emit - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._next_emit = time.monotonic()


class DummyBackend(Backend):
    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None,
        group_config: Optional[dict] = None,
        samp_rate: int = 500,
        num_channels: int = 4,
        signal_freq: float = 1.0,
        amplitude: float = 1.0,
        noise_std: float = 0.0,
        chunk_duration: float = 0.05,
    ):
        super().__init__(
            group_id=group_id,
            response_queue=response_queue,
            data_output_queue=data_output_queue,
        )
        self.group_config = dict(group_config or {})
        self.rf_mode = bool(self.group_config.get("hardware"))

        self.samp_rate = int(self.group_config.get("samp_rate", samp_rate))
        self.num_channels = int(self.group_config.get("num_channels", num_channels))
        self.signal_freq = float(self.group_config.get("signal_freq", signal_freq))
        self.amplitude = float(self.group_config.get("amplitude", amplitude))
        self.noise_std = float(self.group_config.get("noise_std", noise_std))
        self.chunk_duration = float(
            self.group_config.get("chunk_duration", chunk_duration)
        )

        self.generator_worker = None
        self.rf_worker = None
        self.process_worker = None

        self.hardware = {}
        self.mimo_sources = set()
        self.cal_ref_sources = []
        self.dpic_pairs = []
        self.registry = None
        self.schemes_by_device = {}
        self.global_tx_to_device = {}
        self.global_tx_offsets = {}
        self.rx_device_order = []
        self.rx_data_queue = {}
        self.channel_ifs = []
        self.if_filter_bw = []
        self.channel_model = None
        self._cal_enabled_at_start = False
        self.display_ds = int(self.group_config.get("disp_ds", 10))
        self.display_imaginary = bool(
            self.group_config.get("display_imaginary", False)
        )
        self.save_ds = int(self.group_config.get("save_ds", 100))
        self.save_iq = bool(self.group_config.get("save_iq", False))
        self.save_imaginary = bool(self.group_config.get("save_imaginary", True))

        self.populate_data_sources()

    def populate_data_sources(self):
        if not self.rf_mode:
            for ch in range(self.num_channels):
                source = DataSource(
                    group_id=self.group_id,
                    channel=ch,
                    label=f"{self.group_id} Ch{ch + 1}",
                    disp_freq=float(self.samp_rate),
                )
                self.data_sources.add(source)
            return

        self.hardware = dict(self.group_config.get("hardware") or {})

        channel_map = self.group_config.get("channel_map")
        self.mimo_sources, self.registry, self.dpic_pairs = resolve_channel_map(
            self.group_id,
            channel_map,
            self.hardware,
        )
        self.data_sources = set(self.mimo_sources)

        self.global_tx_to_device = {}
        tx_offset = 0
        for device_name, hw in self.hardware.items():
            n_tx = len(hw.get("tx_channels", [0]))
            self.global_tx_offsets[device_name] = tx_offset
            for local in range(n_tx):
                self.global_tx_to_device[tx_offset + local] = (device_name, local)
            tx_offset += n_tx

        self.channel_ifs = list(self.registry.tx_if_freq)
        self.if_filter_bw = list(self.registry.tx_filter_bw)

        cal_cfg = self.group_config.get("calibration", {})
        self._cal_enabled_at_start = bool(cal_cfg.get("enabled", False))
        self.cal_ref_sources = []
        if cal_cfg.get("record_reference", True):
            inject = cal_cfg.get("inject_channels", [0])
            ch_base = len(self.mimo_sources)
            for i, tx_idx in enumerate(inject):
                label = f"CalRef_Tx{tx_idx + 1}"
                source = DataSource(
                    group_id=self.group_id, channel=ch_base + i, label=label
                )
                source.tx_idx = tx_idx
                source.rx_idx = -1
                source.is_cal_ref = True
                self.cal_ref_sources.append(source)

        for device_name, hw in self.hardware.items():
            offset = self.global_tx_offsets[device_name]
            n_tx = len(hw.get("tx_channels", [0]))
            merged = {**self.group_config, **hw}
            self.schemes_by_device[device_name] = scheme_from_config(
                self.samp_rate, n_tx, merged, global_tx_offset=offset
            )

        self.rx_device_order = list(self.hardware.keys())

    def _initialize(self):
        if self.rf_mode:
            from bioview_server.device.usrp.process import ProcessWorker

            sim_cfg = self.group_config.get("rf_simulation", {})
            self.channel_model = MimoChannelModel(
                samp_rate=self.samp_rate,
                if_freq=self.channel_ifs,
                dpic_pairs=self.dpic_pairs,
                num_rx=self.registry.num_rx,
                noise_std=sim_cfg.get("noise_std", self.noise_std),
                cross_coupling=sim_cfg.get("cross_coupling", 0.08),
                on_axis_gain=sim_cfg.get("on_axis_gain", 0.4),
                direct_leak=sim_cfg.get("direct_leak", 0.4),
                dpic_coupling=sim_cfg.get("dpic_coupling", 0.4),
            )
            self.rx_data_queue = {
                name: queue.Queue(maxsize=4) for name in self.rx_device_order
            }

            fmcw_scheme = None
            for scheme in self.schemes_by_device.values():
                if scheme.scheme_type == "fmcw":
                    fmcw_scheme = scheme
                    break

            self.process_worker = ProcessWorker(
                data_sources=self.mimo_sources,
                cal_ref_sources=self.cal_ref_sources,
                samp_rate=self.samp_rate,
                channel_ifs=self.channel_ifs,
                if_filter_bw=self.if_filter_bw,
                rx_queues=self.rx_data_queue,
                rx_device_order=self.rx_device_order,
                schemes_by_device=self.schemes_by_device,
                global_tx_to_device=self.global_tx_to_device,
                signal_scheme=self.group_config.get("signal_scheme", "cw"),
                fmcw_scheme=fmcw_scheme,
                display_queue=self.display_queue,
                display_imaginary=self.display_imaginary,
                save_imaginary=self.save_imaginary,
                save_iq=self.save_iq,
                save_ds=self.save_ds,
                record_cal_ref=bool(
                    self.group_config.get("calibration", {}).get(
                        "record_reference", True
                    )
                ),
                logger=self.logger,
            )

        log_print(
            self.logger,
            "debug",
            f"[DUMMY] Initialized {self.group_id} "
            f"({'RF MIMO' if self.rf_mode else f'{self.num_channels} ch sine'})",
        )
        self.status = DeviceStatus.CONNECTED
        return True

    def _setup_saving(self, save_config: Dict):
        super()._setup_saving(save_config)
        if self.process_worker:
            self.process_worker.save_imaginary = self.save_imaginary
            self.process_worker.save_iq = self.save_iq
            self.process_worker.save_ds = self.save_ds
            self.process_worker.save_queue = self.save_queue

    def _setup_display(self, display_config=None):
        if not self.rf_mode:
            super()._setup_display(display_config)
            return

        if not self.data_output_queue:
            self.data_output_queue = mp.Queue()
        else:
            while not self.data_output_queue.empty():
                self.data_output_queue.get_nowait()

        from bioview_server.common import DisplayWorker

        self.display_worker = DisplayWorker(
            display_sources=list(self.mimo_sources),
            data_input_queue=self.display_queue,
            data_output_queue=self.data_output_queue,
            logger=self.logger,
        )

    def _start_streaming(self):
        if self.rf_mode:
            return self._start_rf_streaming()
        return self._start_legacy_streaming()

    def _start_legacy_streaming(self):
        if self.display_worker is not None:
            if not self.display_worker.is_alive():
                self.display_worker.start()
            self.display_worker.resume()

        if self.save_worker is not None:
            if not self.save_worker.is_alive():
                self.save_worker.start()
            self.save_worker.resume()

        if self.generator_worker is None:
            self.generator_worker = SineWaveWorker(
                samp_rate=self.samp_rate,
                num_channels=self.num_channels,
                signal_freq=self.signal_freq,
                amplitude=self.amplitude,
                noise_std=self.noise_std,
                chunk_duration=self.chunk_duration,
                display_queue=self.display_queue,
                logger=self.logger,
            )

        if not self.generator_worker.is_alive():
            self.generator_worker.start()
        self.generator_worker.resume()

        self.status = DeviceStatus.STREAMING
        return True

    def _start_rf_streaming(self):
        if self.rf_worker is None:
            self.rf_worker = DummyRfWorker(
                samp_rate=self.samp_rate,
                hardware=self.hardware,
                rx_device_order=self.rx_device_order,
                rx_queues=self.rx_data_queue,
                schemes_by_device=self.schemes_by_device,
                global_tx_to_device=self.global_tx_to_device,
                global_tx_offsets=self.global_tx_offsets,
                channel_model=self.channel_model,
                chunk_duration=self.chunk_duration,
                logger=self.logger,
            )

        if not self.rf_worker.is_alive():
            self.rf_worker.start()

        if self.process_worker:
            if not self.process_worker.is_alive():
                self.process_worker.start()

        self.rf_worker.resume()
        if self.process_worker:
            self.process_worker.resume()

        time.sleep(0.35)

        dpic_cfg = self.group_config.get("dpic_balance", {})
        if dpic_cfg.get("auto_on_start") and self.dpic_pairs:
            self._run_dpic_balance()

        if self._cal_enabled_at_start:
            for scheme in self.schemes_by_device.values():
                scheme.set_calibration_enabled(True)
        else:
            for scheme in self.schemes_by_device.values():
                scheme.set_calibration_enabled(False)

        if self.save_worker:
            if not self.save_worker.is_alive():
                self.save_worker.start()
            self.save_worker.resume()

        if self.display_worker:
            if not self.display_worker.is_alive():
                self.display_worker.start()
            self.display_worker.resume()

        self.status = DeviceStatus.STREAMING
        return True

    def _stop_streaming(self):
        if self.generator_worker is not None:
            self.generator_worker.pause()
        if self.rf_worker is not None:
            self.rf_worker.pause()
        if self.process_worker is not None:
            self.process_worker.pause()
        if self.display_worker is not None:
            self.display_worker.pause()
        if self.save_worker is not None:
            self.save_worker.pause()

        self.status = DeviceStatus.CONNECTED
        return True

    def _run_dpic_balance(self):
        if not self.rf_mode or not self.dpic_pairs:
            return None

        dpic_cfg = self.group_config.get("dpic_balance", {})
        prev_cal = self._cal_enabled_at_start
        for scheme in self.schemes_by_device.values():
            scheme.set_calibration_enabled(False)

        balancer = DpicBalancer(
            phase_step_deg=dpic_cfg.get("phase_step_deg", 0.1),
            amp_step=dpic_cfg.get("amp_step", 0.05),
            amp_target=dpic_cfg.get("amp_target", 0.5),
            settle_time_s=dpic_cfg.get("settle_time_s", 0.5),
        )

        def set_phase(global_tx, phase_deg):
            dev_name, local = self.global_tx_to_device[global_tx]
            self.schemes_by_device[dev_name].tx_phase_deg[local] = float(phase_deg)

        def set_amplitude(global_tx, amp):
            dev_name, local = self.global_tx_to_device[global_tx]
            self.schemes_by_device[dev_name].tx_amplitude[local] = float(amp)

        def wait_settle():
            time.sleep(dpic_cfg.get("settle_time_s", 0.5))

        def read_metric(inject_tx, measure_tx):
            return self.process_worker.get_metric(measure_tx, measure_tx)

        pairs = [(p.inject_tx, p.measure_tx) for p in self.dpic_pairs]
        results = balancer.run_all(
            pairs, set_phase, set_amplitude, read_metric, wait_settle
        )

        if "dpic_balance" not in self.group_config:
            self.group_config["dpic_balance"] = {}
        self.group_config["dpic_balance"]["last_results"] = [
            {
                "inject_tx": r.inject_tx,
                "measure_tx": r.measure_tx,
                "best_phase_deg": r.best_phase_deg,
                "best_amplitude": r.best_amplitude,
                "min_metric": r.min_metric,
            }
            for r in results
        ]

        if prev_cal:
            for scheme in self.schemes_by_device.values():
                scheme.set_calibration_enabled(True)

        return results

    def get_data_sources(self):
        if self.rf_mode:
            return set(self.mimo_sources) | set(self.cal_ref_sources)
        return self.data_sources

    def _queue_param_update(self, params):
        if self.rf_mode:
            for param, value in (params or {}).items():
                if param == "calibration.enabled":
                    enabled = bool(value)
                    for scheme in self.schemes_by_device.values():
                        scheme.set_calibration_enabled(enabled)
                elif param in GLOBAL_TX_PARAMS:
                    apply_global_tx_param_to_schemes(
                        self.schemes_by_device,
                        self.global_tx_to_device,
                        self.hardware,
                        self.group_config,
                        param,
                        value,
                    )
                elif param == "calibration" or param.startswith("calibration."):
                    for scheme in self.schemes_by_device.values():
                        scheme.update_param(param, value)
                elif param == "samp_rate":
                    self.samp_rate = int(value)
                    self.group_config["samp_rate"] = value
                elif param == "hardware":
                    self.hardware = dict(value)
                    self.group_config["hardware"] = self.hardware
            return

        if self.generator_worker is None:
            return
        for param, value in (params or {}).items():
            if param == "signal_freq":
                self.generator_worker.signal_freq = float(value)
            elif param == "amplitude":
                self.generator_worker.amplitude = float(value)
            elif param == "noise_std":
                self.generator_worker.noise_std = float(value)

    def _disconnect(self):
        if self.generator_worker is not None:
            self.generator_worker.stop()
            self.generator_worker = None
        if self.rf_worker is not None:
            self.rf_worker.stop()
            self.rf_worker = None
        if self.process_worker is not None:
            self.process_worker.stop()
            self.process_worker = None
        if self.display_worker is not None:
            self.display_worker.stop()
        if self.save_worker is not None:
            self.save_worker.stop()

        self.rx_data_queue = {}
        self.status = DeviceStatus.DISCONNECTED
        return True
