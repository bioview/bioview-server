"""A fake device backend, for tests only.

This is a test double, not a shipped device: nothing under ``bioview_server``
imports it, and it reaches the server only through the registration in this
package's ``__init__``. It exists so the connect -> stream -> display -> save
path can be exercised on a machine with no hardware attached.

Sine mode synthesizes phase-shifted sine waves. RF mode reuses the USRP
signal-scheme / ProcessWorker path against a virtual MIMO channel model, so
calibration and DPIC balance are covered too.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import numpy as np
from bioview_common import (
    DataSource,
    DeviceStatus,
    PausableWorker,
    log_print,
    put_drop_oldest,
    put_or_drop,
)
from bioview_common.datatypes.configuration.hardware_params import (
    GLOBAL_TX_PARAMS,
    apply_global_tx_param_to_schemes,
    build_global_mapping,
)
from bioview_common.datatypes.configuration.usrp_channel_map import (
    components_from_config,
    resolve_channel_map,
)
from bioview_common.signal_schemes import (
    DpicChannel,
    scheme_from_config,
)

from bioview_server.common import balance_outcome, build_balancer
from bioview_server.datatypes import Backend

from .rf_simulation import MimoChannelModel
from .rf_worker import FakeRfWorker


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
        save_queue: mp.Queue = None,
        save_ds: int = 1,
        logger=None,
    ):
        super().__init__(logger=logger)
        self.samp_rate = float(samp_rate)
        self.num_channels = int(num_channels)
        self.signal_freq = float(signal_freq)
        self.amplitude = float(amplitude)
        self.noise_std = float(noise_std)
        self.display_queue = display_queue
        self.save_queue = save_queue
        self.save_ds = max(1, int(save_ds))
        self._save_samples_emitted = 0

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

        # Save path carries the undecimated-by-disp_ds stream, decimated only
        # by save_ds, and is tagged so the recorder can detect dropped chunks.
        if self.save_queue is not None:
            save_chunk = chunk
            if self.save_ds > 1:
                n_windows = save_chunk.shape[1] // self.save_ds
                if n_windows:
                    usable = n_windows * self.save_ds
                    save_chunk = (
                        save_chunk[:, :usable]
                        .reshape(save_chunk.shape[0], n_windows, self.save_ds)
                        .mean(axis=2)
                    )
                else:
                    save_chunk = save_chunk[:, :0]
            if save_chunk.shape[1]:
                item = {
                    "data": np.ascontiguousarray(save_chunk, dtype=np.float32),
                    "sample_idx": self._save_samples_emitted,
                    "t_wall": time.time(),
                }
                self._save_samples_emitted += save_chunk.shape[1]
                if not put_or_drop(self.save_queue, item, timeout=0.5):
                    log_print(
                        self.logger, "error", "[FAKE] Save queue full; dropping chunk"
                    )

        if not put_drop_oldest(
            self.display_queue, np.ascontiguousarray(chunk, dtype=np.float32)
        ):
            log_print(
                self.logger, "warning", "[FAKE] Display queue full; dropping chunk"
            )

        self._next_emit += self.chunk_duration
        sleep_for = self._next_emit - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._next_emit = time.monotonic()


class FakeBackend(Backend):
    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None,
        save_output_queue: mp.Queue = None,
        group_config: dict | None = None,
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
            save_output_queue=save_output_queue,
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
        self.stream_components = ["amplitude"]
        self.cal_ref_sources = []
        self.dpic_pairs = []
        self._cal_enabled = False
        self.registry = None
        self.schemes_by_device = {}
        self.global_tx_to_device = {}
        self.global_tx_offsets = {}
        self.rx_device_order = []
        self.rx_data_queue = {}
        self.channel_ifs = []
        self.if_filter_bw = []
        self.channel_model = None
        self.display_ds = int(self.group_config.get("disp_ds", 10))
        self.display_imaginary = bool(self.group_config.get("display_imaginary", False))
        self.save_ds = int(self.group_config.get("save_ds", 100))
        self.save_iq = bool(self.group_config.get("save_iq", False))
        self.save_imaginary = bool(self.group_config.get("save_imaginary", True))

        self.populate_data_sources()

    def get_save_freq(self) -> float:
        """Saved rate: acquisition rate decimated by ``save_ds`` only."""
        return float(self.samp_rate) / max(1, int(self.save_ds))

    def get_display_frequency(self) -> float:
        """Rate (Hz) at which this device emits display samples.

        The non-RF sine path forwards every sample, so it is the sample rate.
        The RF path runs the USRP ProcessWorker, which averages ``save_ds``
        samples per point and then drops by ``display_ds``.
        """
        if not self.rf_mode:
            return float(self.samp_rate)
        divisor = max(1, int(self.save_ds)) * max(1, int(self.display_ds))
        return float(self.samp_rate) / divisor

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
        self.stream_components = components_from_config(self.group_config)
        self.mimo_sources, self.registry, self.dpic_pairs = resolve_channel_map(
            self.group_id,
            channel_map,
            self.hardware,
            disp_freq=self.get_display_frequency(),
            components=self.stream_components,
        )
        self.data_sources = set(self.mimo_sources)

        (
            self.global_tx_to_device,
            self.global_tx_offsets,
            _tx_gains,
        ) = build_global_mapping(self.hardware, "tx")

        self.channel_ifs = list(self.registry.tx_if_freq)
        self.if_filter_bw = list(self.registry.tx_filter_bw)

        cal_cfg = self.group_config.get("calibration", {})
        self._cal_enabled = bool(cal_cfg.get("enabled", False))
        self.cal_ref_sources = []
        if cal_cfg.get("record_reference", True):
            inject = cal_cfg.get("inject_channels", [0])
            ch_base = len(self.mimo_sources)
            for i, tx_idx in enumerate(inject):
                label = f"CalRef_Tx{tx_idx + 1}"
                source = DataSource(
                    group_id=self.group_id,
                    channel=ch_base + i,
                    label=label,
                    disp_freq=self.get_display_frequency(),
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
                display_ds=self.display_ds,
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
            f"[FAKE] Initialized {self.group_id} "
            f"({'RF MIMO' if self.rf_mode else f'{self.num_channels} ch sine'})",
        )
        self.status = DeviceStatus.CONNECTED
        return True

    def _setup_saving(self, save_config: dict):
        super()._setup_saving(save_config)
        if self.process_worker:
            self.process_worker.save_imaginary = self.save_imaginary
            self.process_worker.save_iq = self.save_iq
            self.process_worker.save_ds = self.save_ds
            self.process_worker.save_queue = self.save_queue
            # save_ds also sets the display rate; keep disp_freq in step.
            disp_freq = self.get_display_frequency()
            for source in list(self.mimo_sources) + list(self.cal_ref_sources):
                source.disp_freq = disp_freq
            if self.display_worker is not None:
                self.display_worker.set_display_sources(self._display_sources())

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
                save_queue=self.save_queue,
                save_ds=self.save_ds,
                logger=self.logger,
            )

        if not self.generator_worker.is_alive():
            self.generator_worker.start()
        self.generator_worker.resume()

        self.status = DeviceStatus.STREAMING
        return True

    def _start_rf_streaming(self):
        if self.rf_worker is None:
            self.rf_worker = FakeRfWorker(
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

        if self.process_worker and not self.process_worker.is_alive():
            self.process_worker.start()

        self.rf_worker.resume()
        if self.process_worker:
            self.process_worker.resume()

        time.sleep(0.35)

        # The current calibration state, not the config's start-up value: the
        # overlay is toggled at runtime and would be reset on every Start.
        self._set_calibration_enabled(self._cal_enabled)

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

    def _post_start_streaming(self):
        dpic_cfg = self.group_config.get("dpic_balance", {})
        if dpic_cfg.get("auto_on_start") and self.dpic_pairs:
            # On its own thread, like a requested balance: this runs after the
            # START_STREAMING reply but still on the command loop, so calling
            # the search inline here left Stop unservable for its duration.
            self._start_balance_thread()

    def _set_calibration_enabled(self, enabled: bool):
        enabled = bool(enabled)
        for scheme in self.schemes_by_device.values():
            scheme.set_calibration_enabled(enabled)
        self._cal_enabled = enabled

    def _apply_channel_if(self, global_tx: int, freq: float):
        """Move one simulated Tx onto ``freq`` everywhere the IF is held."""
        freq = float(freq)
        self.channel_ifs[global_tx] = freq
        if global_tx < len(self.registry.tx_if_freq):
            self.registry.tx_if_freq[global_tx] = freq

        dev_name, local = self.global_tx_to_device[global_tx]
        hw = self.hardware.get(dev_name)
        if hw is not None:
            if_freqs = list(hw.get("if_freq", []) or [])
            while len(if_freqs) <= local:
                if_freqs.append(freq)
            if_freqs[local] = freq
            hw["if_freq"] = if_freqs
            self.group_config["hardware"] = self.hardware

        scheme = self.schemes_by_device[dev_name]
        local_ifs = [
            float(f)
            for idx, f in enumerate(self.channel_ifs)
            if self.global_tx_to_device.get(idx, (None, None))[0] == dev_name
        ]
        scheme.update_param("if_freq", local_ifs)

        model = self.channel_model
        if model is not None and global_tx < len(model.if_freq):
            model.if_freq[global_tx] = freq
        if self.process_worker is not None:
            self.process_worker.set_channel_if(global_tx, freq)

    def _coerce_dpic_inject_frequencies(self):
        """Put every inject Tx on its measure Tx's IF before balancing.

        Same rule as the USRP backend: the Rx band-passes around the measure
        Tx's IF, so an injection anywhere else cannot cancel the direct path.
        """
        num_tx = len(self.channel_ifs)
        for pair in self.dpic_pairs:
            if pair.inject_tx >= num_tx or pair.measure_tx >= num_tx:
                continue
            target_if = self.channel_ifs[pair.measure_tx]
            inject_if = self.channel_ifs[pair.inject_tx]
            if abs(inject_if - target_if) <= 1e-6:
                continue
            log_print(
                self.logger,
                "info",
                f"[DPIC] Coercing inject Tx{pair.inject_tx} from "
                f"{inject_if:.0f} Hz to {target_if:.0f} Hz to match measure "
                f"Tx{pair.measure_tx}",
            )
            self._apply_channel_if(pair.inject_tx, target_if)

    def _run_dpic_balance(self):
        if not self.rf_mode:
            message = "DPIC balance needs RF simulation mode (a 'hardware' block)."
            log_print(self.logger, "error", f"[DPIC] {message}")
            return {"ok": False, "message": message, "results": []}
        if not self.dpic_pairs:
            message = (
                "No DPIC pairs are configured for this device group; nothing to balance."
            )
            log_print(self.logger, "error", f"[DPIC] {message}")
            return {"ok": False, "message": message, "results": []}
        if self.process_worker is None or not self.process_worker.is_running:
            message = (
                "Balance requires the processing worker to be running; "
                "start streaming first."
            )
            log_print(self.logger, "error", f"[DPIC] {message}")
            return {"ok": False, "message": message, "results": []}

        self._coerce_dpic_inject_frequencies()

        dpic_cfg = self.group_config.get("dpic_balance", {})
        settle_s = float(dpic_cfg.get("settle_time_s", 0.02))
        prev_cal = self._cal_enabled
        if prev_cal:
            self._set_calibration_enabled(False)

        balancer = build_balancer(
            dpic_cfg,
            should_abort=self.balance_aborted,
            on_progress=self.publish_balance_progress,
        )

        def _make_channel(pair):
            dev_name, local = self.global_tx_to_device[pair.inject_tx]
            scheme = self.schemes_by_device[dev_name]
            measure_rx = pair.target_rx
            read_timeout = max(1.0, settle_s * 8)

            def set_phase(v):
                scheme.tx_phase_deg[local] = float(v)

            def set_amplitude(v):
                scheme.tx_amplitude[local] = float(v)

            return DpicChannel(
                inject_tx=pair.inject_tx,
                measure_tx=pair.measure_tx,
                measure_rx=measure_rx,
                set_phase=set_phase,
                set_amplitude=set_amplitude,
                read_metric=lambda: self.process_worker.wait_for_metric(
                    pair.measure_tx, measure_rx, min_new=2, timeout=read_timeout
                ),
                wait_settle=time.sleep,
                start_phase_deg=float(scheme.tx_phase_deg[local]),
                start_amplitude=float(scheme.get_tx_amplitude(local)),
            )

        results = balancer.balance_all([_make_channel(p) for p in self.dpic_pairs])
        outcome = balance_outcome(self.logger, results)

        if "dpic_balance" not in self.group_config:
            self.group_config["dpic_balance"] = {}
        self.group_config["dpic_balance"]["last_results"] = outcome["results"]

        if prev_cal:
            self._set_calibration_enabled(True)

        return outcome

    def get_data_sources(self):
        if self.rf_mode:
            return set(self.mimo_sources) | set(self.cal_ref_sources)
        return self.data_sources

    def _display_sources(self):
        """Rows the ProcessWorker emits, in channel order."""
        if not self.rf_mode:
            return super()._display_sources()
        sources = list(self.mimo_sources)
        if self.process_worker is None or self.process_worker.record_cal_ref:
            sources += list(self.cal_ref_sources)
        return sorted(sources, key=lambda s: s.channel)

    def _reload_channel_map(self, channel_map):
        """Rebuild everything the channel map decides, in place.

        Same path as the USRP backend: DPIC pairs are specified in the channel
        map, which is edited in the settings panel, so an edit that stops here
        is an edit the balance never sees.
        """
        if self._streaming.is_set():
            log_print(
                self.logger,
                "error",
                "[FAKE] Channel map changed while streaming; it takes "
                "effect on the next Start.",
            )
            return False

        self.group_config["channel_map"] = channel_map

        # The rf worker and the channel model hold these scheme objects, so
        # they must survive the rebuild; schemes depend on `hardware`, which a
        # channel-map edit never touches.
        preserved_schemes = dict(self.schemes_by_device)
        cal_enabled = self._cal_enabled

        self.populate_data_sources()

        for device_name, scheme in preserved_schemes.items():
            self.schemes_by_device[device_name] = scheme
        self._cal_enabled = cal_enabled

        if self.channel_model is not None:
            self.channel_model.dpic_pairs = list(self.dpic_pairs)
        if self.process_worker is not None:
            self.process_worker.set_sources(
                self.mimo_sources,
                self.cal_ref_sources,
                self.channel_ifs,
                self.if_filter_bw,
            )
        if self.display_worker is not None:
            self.display_worker.set_display_sources(self._display_sources())

        log_print(
            self.logger,
            "info",
            f"[FAKE] Channel map reloaded: {len(self.mimo_sources)} "
            f"measurement source(s), {len(self.dpic_pairs)} DPIC pair(s)",
        )
        return True

    def _apply_param_update_local(self, params):
        """Parent-side mirror; see the USRP backend's copy for why."""
        channel_map = (params or {}).get("channel_map")
        if isinstance(channel_map, dict):
            self.group_config["channel_map"] = channel_map
            self.populate_data_sources()

    def _queue_param_update(self, params):
        if self.rf_mode:
            for param, value in (params or {}).items():
                if param == "channel_map" and isinstance(value, dict):
                    self._reload_channel_map(value)
                    continue

                if param == "calibration.enabled":
                    self.group_config.setdefault("calibration", {})["enabled"] = bool(
                        value
                    )
                    self._set_calibration_enabled(value)
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
                    if param == "calibration" and isinstance(value, dict):
                        self.group_config["calibration"] = dict(value)
                        self._cal_enabled = bool(value.get("enabled", self._cal_enabled))
                    else:
                        key = param.split(".", 1)[1]
                        self.group_config.setdefault("calibration", {})[key] = value
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
