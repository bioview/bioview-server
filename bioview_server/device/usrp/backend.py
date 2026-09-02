import math
import multiprocessing as mp
import os
import queue
import time
from typing import Dict


os.environ["UHD_LOG_LEVEL"] = "error"

import uhd
from bioview_common import (
    DATA_OUTPUT_QUEUE_DEPTH,
    RX_QUEUE_DEPTH,
    DataSource,
    DeviceStatus,
    USRPConfiguration,
    drain,
    log_print,
)
from bioview_common.datatypes.configuration.hardware_params import (
    GLOBAL_RX_PARAMS,
    GLOBAL_TX_PARAMS,
    apply_global_rx_values_to_hardware,
    apply_global_tx_values_to_hardware,
    build_global_mapping,
)
from bioview_common.datatypes.configuration.usrp_channel_map import (
    build_hardware_dict,
    resolve_channel_map,
    resolve_device_serial,
)
from bioview_common.signal_schemes import (
    DpicBalancer,
    DpicChannel,
    scheme_from_config,
)

from bioview_server.datatypes import Backend

from .process import ProcessWorker
from .receive import ReceiveWorker
from .transmit import TX_PARAMS, TransmitWorker
from .utils import (
    check_channels,
    discover_devices,
    get_usrp_address,
    setup_pps,
    setup_ref,
)


SETTLING_TIME = 0.3
FILLING_TIME = 0.35
RX_TX_PARAMS = {"rx_gain"}


def initialize_usrp_device(
    serial,
    tx_subdev,
    rx_subdev,
    clock,
    pps,
    rx_channels,
    tx_channels,
    samp_rate,
    carrier_freq,
    rx_gain,
    tx_gain,
    cpu_format,
    wire_format,
    logger=None,
):
    if not hasattr(uhd, "usrp"):
        raise RuntimeError(
            "UHD Python bindings are incomplete (uhd.usrp missing). "
            "Reinstall UHD from Ettus for your Python version."
        )
    usrp = uhd.usrp.MultiUSRP(f"serial={serial},num_recv_frames=1024")

    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(rx_subdev))
    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(tx_subdev))

    if not setup_ref(usrp, clock, usrp.get_num_mboards()):
        log_print(logger, "error", "Unable to lock reference clock")
        return None

    if not setup_pps(usrp, pps, usrp.get_num_mboards()):
        log_print(logger, "error", "Unable to lock timing source")
        return None

    rx_channels, tx_channels = check_channels(usrp, rx_channels, tx_channels, logger)
    if not rx_channels and not tx_channels:
        log_print(
            logger,
            "error",
            "Mismatch between specified channels and available channels",
        )
        return None

    def _tune_request(freq_hz):
        """UHD 4.10+ Python API expects a tune_request object."""
        try:
            if hasattr(uhd, "types") and hasattr(uhd.types, "TuneRequest"):
                return uhd.types.TuneRequest(float(freq_hz))
        except Exception:
            pass
        try:
            lib_types = getattr(getattr(uhd, "libpyuhd", None), "types", None)
            if lib_types and hasattr(lib_types, "tune_request"):
                return lib_types.tune_request(float(freq_hz))
        except Exception:
            pass
        return float(freq_hz)

    tune = _tune_request(carrier_freq)

    for idx, chan in enumerate(rx_channels):
        usrp.set_rx_rate(samp_rate, chan)
        usrp.set_rx_freq(tune, chan)
        usrp.set_rx_gain(rx_gain[idx], chan)
        usrp.set_rx_antenna("RX2", chan)

    for idx, chan in enumerate(tx_channels):
        usrp.set_tx_rate(samp_rate, chan)
        usrp.set_tx_freq(tune, chan)
        usrp.set_tx_gain(tx_gain[idx], chan)
        usrp.set_tx_antenna("TX1", chan)

    stream_args = uhd.usrp.StreamArgs(cpu_format, wire_format)
    stream_args.channels = tx_channels
    tx_streamer = usrp.get_tx_stream(stream_args)

    stream_args.channels = rx_channels
    rx_streamer = usrp.get_rx_stream(stream_args)

    return {"usrp": usrp, "tx_streamer": tx_streamer, "rx_streamer": rx_streamer}


class USRPBackend(Backend):
    def __init__(
        self,
        group_id: str,
        samp_rate: int,
        devices: Dict,
        group_config: dict,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None,
        display_ds: int = 10,
        display_imaginary: bool = False,
        save_ds: int = 10,
        save_iq: bool = False,
        save_imaginary: bool = True,
        discovered_devices: Dict = None,
    ):
        super().__init__(
            group_id=group_id,
            response_queue=response_queue,
            data_output_queue=data_output_queue,
        )
        self.samp_rate = samp_rate
        self.group_config = group_config or {}
        self.rx_data_queue = {}
        self.display_ds = display_ds
        self.display_imaginary = display_imaginary
        self.save_ds = save_ds
        self.save_iq = save_iq
        self.save_imaginary = save_imaginary

        self.usrp_configs = {}
        self.usrp_handlers = {}
        self.usrp_states = {}
        self.transmit_workers = {}
        self.tx_command_queue = {}
        self.receive_workers = {}
        self.rx_command_queue = {}
        self.schemes_by_device = {}
        self.global_tx_to_device = {}
        self.global_tx_offsets = {}
        self.global_rx_offsets = {}
        self.dpic_pairs = []
        self.mimo_sources = set()
        self.cal_ref_sources = []
        self.registry = None
        self.rx_device_order = []
        self._cal_enabled_at_start = False
        self._cal_enabled = False
        self.global_rx_to_device = {}
        self.rx_gains_global = []
        self.tx_gains_global = []

        self.discovered_devices = discovered_devices or {}
        self.channel_ifs = []
        self.if_filter_bw = []

        self.hardware = build_hardware_dict_from_group(group_config, devices, group_id)

        for device_name, hw_dict in self.hardware.items():
            hw_dict["device_name"] = device_name
            cfg = USRPConfiguration({**self.group_config, **hw_dict})
            self.usrp_configs[device_name] = cfg
            self.usrp_handlers[device_name] = None
            self.usrp_states[device_name] = DeviceStatus.DISCONNECTED
            self.transmit_workers[device_name] = None
            self.receive_workers[device_name] = None

        self.populate_data_sources()

    def populate_data_sources(self):
        channel_map = self.group_config.get("channel_map")
        self.mimo_sources, self.registry, self.dpic_pairs = resolve_channel_map(
            self.group_id,
            channel_map,
            self.hardware,
        )
        self.data_sources = set(self.mimo_sources)

        (
            self.global_tx_to_device,
            self.global_tx_offsets,
            self.tx_gains_global,
        ) = build_global_mapping(self.hardware, "tx")
        (
            self.global_rx_to_device,
            self.global_rx_offsets,
            self.rx_gains_global,
        ) = build_global_mapping(self.hardware, "rx")

        self.channel_ifs = list(self.registry.tx_if_freq)
        self.if_filter_bw = list(self.registry.tx_filter_bw)

        cal_cfg = self.group_config.get("calibration", {})
        self._cal_enabled_at_start = bool(cal_cfg.get("enabled", False))
        self._cal_enabled = self._cal_enabled_at_start
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

    def _resolve_serial(self, device_name: str, hw_entry: dict) -> str:
        return resolve_device_serial(
            device_name,
            hw_entry,
            self.discovered_devices,
            get_usrp_address,
        )

    def _initialize(self):
        if not self.discovered_devices:
            self.discovered_devices = discover_devices(self.logger)
        else:
            log_print(
                self.logger,
                "debug",
                "Using server-provided USRP discovery cache",
            )

        for device_name, device_config in self.usrp_configs.items():
            hw_entry = self.hardware[device_name]
            device_serial = self._resolve_serial(device_name, hw_entry)
            if not device_serial:
                log_print(
                    self.logger,
                    "error",
                    f"Unable to resolve serial for {device_name}",
                )
                # Raised, not returned: upstream turns a falsy result into a
                # generic failure message, losing which device went wrong.
                raise RuntimeError(
                    f"could not resolve the serial number for {device_name}. "
                    "Check that the radio is attached and powered on"
                )

            try:
                rx_gain = device_config.get_param("rx_gain")
                tx_gain = device_config.get_param("tx_gain")
                rx_channels = device_config.get_param("rx_channels")
                tx_channels = device_config.get_param("tx_channels")

                response = initialize_usrp_device(
                    serial=device_serial,
                    rx_subdev=device_config.get_param("rx_subdev"),
                    tx_subdev=device_config.get_param("tx_subdev"),
                    clock=device_config.get_param("clock"),
                    pps=device_config.get_param("pps"),
                    rx_channels=rx_channels,
                    tx_channels=tx_channels,
                    samp_rate=device_config.get_param("samp_rate", self.samp_rate),
                    carrier_freq=device_config.get_param("carrier_freq"),
                    rx_gain=rx_gain,
                    tx_gain=tx_gain,
                    cpu_format=device_config.get_param("cpu_format"),
                    wire_format=device_config.get_param("wire_format"),
                    logger=self.logger,
                )

                if not response:
                    self.usrp_states[device_name] = DeviceStatus.DISCONNECTED
                    raise RuntimeError(
                        f"{device_name} did not respond while being opened"
                    )

                self.usrp_handlers[device_name] = response["usrp"]
                # Bounded: this queue used to grow without limit whenever the
                # ProcessWorker was slow or (before the start-order fix) not
                # yet running at all.
                self.rx_data_queue[device_name] = queue.Queue(maxsize=RX_QUEUE_DEPTH)
                self.rx_command_queue[device_name] = queue.Queue()

                rx_offset = self.global_rx_offsets[device_name]
                local_rx_gain = list(rx_gain)
                self.receive_workers[device_name] = ReceiveWorker(
                    usrp=response["usrp"],
                    rx_gain=local_rx_gain,
                    rx_channels=rx_channels,
                    rx_streamer=response["rx_streamer"],
                    rx_queue=self.rx_data_queue[device_name],
                    cmd_queue=self.rx_command_queue[device_name],
                    global_rx_offset=rx_offset,
                    logger=self.logger,
                )

                self.tx_command_queue[device_name] = queue.Queue()
                scheme = self.schemes_by_device[device_name]
                self.transmit_workers[device_name] = TransmitWorker(
                    usrp=response["usrp"],
                    tx_gain=tx_gain,
                    tx_channels=tx_channels,
                    samp_rate=self.samp_rate,
                    tx_streamer=response["tx_streamer"],
                    scheme=scheme,
                    cmd_queue=self.tx_command_queue[device_name],
                    global_tx_offset=self.global_tx_offsets[device_name],
                    logger=self.logger,
                )

                self.usrp_states[device_name] = DeviceStatus.CONNECTED
            except Exception as e:
                log_print(
                    self.logger,
                    "error",
                    f"Unable to initialize {device_name}: {e}",
                )
                raise

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
                self.group_config.get("calibration", {}).get("record_reference", True)
            ),
            logger=self.logger,
        )
        return True

    def _setup_saving(self, save_config: Dict):
        super()._setup_saving(save_config)
        self.process_worker.save_imaginary = self.save_imaginary
        self.process_worker.save_iq = self.save_iq
        self.process_worker.save_ds = self.save_ds
        self.process_worker.save_queue = self.save_queue

    def _start_streaming(self):
        for worker in self.transmit_workers.values():
            if not worker.is_alive():
                worker.start()
            worker.resume()

        for worker in self.receive_workers.values():
            if not worker.is_alive():
                worker.start()
            worker.resume()

        # Start this before anything else: it drains the Rx queues, and DPIC
        # balance has no metrics to read until it has produced some.
        if self.process_worker:
            if not self.process_worker.is_alive():
                self.process_worker.start()
            self.process_worker.resume()

        time.sleep(FILLING_TIME)

        self._set_calibration_enabled(self._cal_enabled_at_start)

        if self.save_worker:
            if not self.save_worker.is_alive():
                self.save_worker.start()
            self.save_worker.resume()

        if self.display_worker:
            if not self.display_worker.is_alive():
                self.display_worker.start()
            self.display_worker.resume()

        return True

    def _stop_streaming(self):
        if self.display_worker:
            self.display_worker.pause()
        if self.save_worker:
            self.save_worker.pause()
        if self.process_worker:
            self.process_worker.pause()
        for worker in self.transmit_workers.values():
            worker.pause()
        for worker in self.receive_workers.values():
            worker.pause()
        return True

    def _post_start_streaming(self):
        dpic_cfg = self.group_config.get("dpic_balance", {})
        if dpic_cfg.get("auto_on_start") and self.dpic_pairs:
            self._run_dpic_balance()

    def _set_calibration_enabled(self, enabled: bool):
        """Toggle the calibration overlay through the Tx command queues.

        Mutating ``scheme`` from this thread would race the transmit threads and
        would also leave ``TransmitWorker._use_cyclic`` stale -- with a stale
        flag the worker keeps replaying its pre-built cyclic buffer and the
        calibration bursts never reach the air.
        """
        enabled = bool(enabled)
        for q in self.tx_command_queue.values():
            q.put({"param": "calibration.enabled", "value": enabled})
        self._cal_enabled = enabled

    def _validate_dpic_pairs(self):
        """Warn about pairs that cannot physically null the direct path."""
        num_tx = len(self.channel_ifs)
        measurable = {(s.tx_idx, s.rx_idx) for s in self.mimo_sources}
        for pair in self.dpic_pairs:
            rx_idx = pair.target_rx
            if pair.inject_tx >= num_tx or pair.measure_tx >= num_tx:
                log_print(
                    self.logger,
                    "error",
                    f"[DPIC] Pair inject={pair.inject_tx} measure={pair.measure_tx} "
                    f"references a Tx outside the {num_tx}-channel group",
                )
                continue
            inject_if = self.channel_ifs[pair.inject_tx]
            measure_if = self.channel_ifs[pair.measure_tx]
            if abs(inject_if - measure_if) > 1e-6:
                log_print(
                    self.logger,
                    "error",
                    f"[DPIC] Inject Tx{pair.inject_tx} is at IF {inject_if:.0f} Hz but "
                    f"measure Tx{pair.measure_tx} is at {measure_if:.0f} Hz. The Rx "
                    "band-pass rejects the injected tone, so no phase/amplitude "
                    "setting can cancel the direct path. Put both Tx on the same IF.",
                )
            if (pair.measure_tx, rx_idx) not in measurable:
                log_print(
                    self.logger,
                    "error",
                    f"[DPIC] No measurement source for Tx{pair.measure_tx}/"
                    f"Rx{rx_idx}; set 'measure_rx' on the pair to an Rx index that "
                    "is part of the channel map.",
                )

    def _rx_gain_range(self, device_name: str):
        usrp = self.usrp_handlers.get(device_name)
        try:
            rng = usrp.get_rx_gain_range()
            return (float(rng.start()), float(rng.stop()))
        except Exception:
            return (0.0, 76.0)

    def _auto_gain_rx(
        self,
        global_rx: int,
        measure_tx: int,
        target: float,
        settle_s: float,
        max_steps: int = 4,
        tolerance_db: float = 1.0,
    ):
        """Bring the measured level on ``global_rx`` up to ``target``.

        This is the 'boost Rx power to a usable level' step: a null search is
        meaningless if the direct path sits in the noise floor.

        The correction is proportional -- ``20*log10(target/level)`` dB in one
        move -- rather than a fixed 3 dB ladder. A ladder needs up to ~25 steps
        to cross a B2xx's gain range, and at roughly 100 ms per measurement that
        alone would consume most of the balance time budget.
        """
        entry = self.global_rx_to_device.get(global_rx)
        if entry is None or global_rx >= len(self.rx_gains_global):
            return
        dev_name, _local = entry
        min_gain, max_gain = self._rx_gain_range(dev_name)
        read_timeout = max(1.0, settle_s * 8)

        for _ in range(max(int(max_steps), 1)):
            level = self.process_worker.wait_for_metric(
                measure_tx, global_rx, min_new=2, timeout=read_timeout
            )
            if level is None:
                return
            if level <= 0:
                delta_db = max_gain - self.rx_gains_global[global_rx]
            else:
                delta_db = 20.0 * math.log10(target / level)
            if abs(delta_db) <= tolerance_db:
                break

            new_gain = min(
                max(self.rx_gains_global[global_rx] + delta_db, min_gain), max_gain
            )
            if abs(new_gain - self.rx_gains_global[global_rx]) < 1e-6:
                break
            self.rx_gains_global[global_rx] = new_gain
            self.rx_command_queue[dev_name].put(
                {"param": "rx_gain", "value": list(self.rx_gains_global)}
            )
            time.sleep(settle_s)

        log_print(
            self.logger,
            "debug",
            f"[DPIC] Rx{global_rx} gain set to "
            f"{self.rx_gains_global[global_rx]:.1f} dB (target {target})",
        )

    def _tx_gain_range(self, device_name: str):
        """Analog Tx gain limits for a device, with a safe B2xx default."""
        usrp = self.usrp_handlers.get(device_name)
        try:
            rng = usrp.get_tx_gain_range()
            return (float(rng.start()), float(rng.stop()))
        except Exception:
            return (0.0, 89.75)

    def _build_dpic_channel(self, pair, dpic_cfg) -> DpicChannel:
        settle_s = float(dpic_cfg.get("settle_time_s", 0.02))
        gain_settle_s = float(dpic_cfg.get("gain_settle_time_s", 0.05))
        amp_target = float(dpic_cfg.get("amp_target", 0.5))
        measure_rx = pair.target_rx

        dev_name, _ = self.global_tx_to_device[pair.inject_tx]
        worker = self.transmit_workers[dev_name]
        # Metric freshness timeout: two chunks plus slack. Long enough that a
        # slow chunk does not read as "no data", short enough that a genuinely
        # silent path fails fast instead of eating the time budget.
        read_timeout = max(1.0, settle_s * 8)

        return DpicChannel(
            inject_tx=pair.inject_tx,
            measure_tx=pair.measure_tx,
            measure_rx=measure_rx,
            set_phase=lambda v: worker.set_global_tx_param(pair.inject_tx, "phase", v),
            set_amplitude=lambda v: worker.set_global_tx_param(
                pair.inject_tx, "amplitude", v
            ),
            set_gain=lambda v: self._set_inject_gain(pair.inject_tx, v),
            get_gain=lambda: self._get_inject_gain(pair.inject_tx),
            gain_range=self._tx_gain_range(dev_name),
            # Both readers wait for chunks captured *after* the change, so the
            # search is never biased by the Rx buffering latency.
            read_metric=lambda: self.process_worker.wait_for_metric(
                pair.measure_tx, measure_rx, min_new=2, timeout=read_timeout
            ),
            read_complex=lambda: self.process_worker.wait_for_metric_complex(
                pair.measure_tx, measure_rx, min_new=2, timeout=read_timeout
            ),
            wait_settle=lambda: time.sleep(settle_s),
            wait_gain_settle=lambda: time.sleep(gain_settle_s),
            auto_gain_rx=lambda: self._auto_gain_rx(
                measure_rx, pair.measure_tx, amp_target, settle_s
            ),
            start_phase_deg=worker.get_global_tx_param(pair.inject_tx, "phase") or 0.0,
            start_amplitude=(
                worker.get_global_tx_param(pair.inject_tx, "amplitude") or 0.0
            ),
        )

    def _get_inject_gain(self, global_tx: int) -> float:
        if global_tx < len(self.tx_gains_global):
            return float(self.tx_gains_global[global_tx])
        dev_name, _ = self.global_tx_to_device[global_tx]
        val = self.transmit_workers[dev_name].get_global_tx_param(global_tx, "gain")
        return 0.0 if val is None else float(val)

    def _set_inject_gain(self, global_tx: int, gain_db: float):
        dev_name, _ = self.global_tx_to_device[global_tx]
        self.transmit_workers[dev_name].set_global_tx_param(
            global_tx, "gain", float(gain_db)
        )
        # Mirror locally so get_gain reflects the intent immediately rather than
        # lagging by however long the Tx thread takes to drain its queue.
        if global_tx < len(self.tx_gains_global):
            self.tx_gains_global[global_tx] = float(gain_db)

    def _run_dpic_balance(self):
        if not self.dpic_pairs:
            return None
        if self.process_worker is None or not self.process_worker.is_running:
            log_print(
                self.logger,
                "error",
                "[DPIC] Balance requires the processing worker to be running; "
                "start streaming first.",
            )
            return None

        self._validate_dpic_pairs()

        dpic_cfg = self.group_config.get("dpic_balance", {})
        settle_s = float(dpic_cfg.get("settle_time_s", 0.02))

        # The calibration AM overlay modulates the very amplitude the search
        # minimizes, so it must be off for the duration -- and restored to the
        # state it was actually in, not to the config's start-up value.
        prev_cal = self._cal_enabled
        if prev_cal:
            self._set_calibration_enabled(False)
            time.sleep(settle_s)

        balancer = DpicBalancer(
            phase_step_deg=dpic_cfg.get("phase_step_deg", 0.1),
            amp_step=dpic_cfg.get("amp_step", 0.05),
            coarse_phase_step_deg=dpic_cfg.get("coarse_phase_step_deg", 10.0),
            coarse_amp_step=dpic_cfg.get("coarse_amp_step", 0.1),
            max_amplitude=dpic_cfg.get("max_amplitude", 1.0),
            amp_target=dpic_cfg.get("amp_target", 0.5),
            settle_time_s=settle_s,
            gain_settle_time_s=dpic_cfg.get("gain_settle_time_s", 0.05),
            time_budget_s=dpic_cfg.get("time_budget_s", 25.0),
            probe_amplitude=dpic_cfg.get("probe_amplitude", 0.5),
            target_weight=dpic_cfg.get("target_weight", 0.5),
            min_weight=dpic_cfg.get("min_weight", 0.15),
            refine_iterations=dpic_cfg.get("refine_iterations", 3),
        )

        channels = [self._build_dpic_channel(p, dpic_cfg) for p in self.dpic_pairs]
        results = balancer.balance_all(channels)

        for r in results:
            if not r.converged:
                log_print(
                    self.logger,
                    "error",
                    f"[DPIC] Tx{r.inject_tx}->Tx{r.measure_tx}/Rx{r.measure_rx}: no "
                    "metric was readable; previous phase/amplitude restored.",
                )
            else:
                log_print(
                    self.logger,
                    "info",
                    f"[DPIC] Tx{r.inject_tx}->Tx{r.measure_tx}/Rx{r.measure_rx} "
                    f"[{r.method}]: phase={r.best_phase_deg:.2f} deg "
                    f"amp={r.best_amplitude:.3f} gain={r.inject_gain_db:.1f} dB "
                    f"null={r.null_depth_db:.1f} dB "
                    f"({r.num_measurements} reads in {r.elapsed_s:.1f} s)",
                )

        if "dpic_balance" not in self.group_config:
            self.group_config["dpic_balance"] = {}
        self.group_config["dpic_balance"]["last_results"] = [
            {
                "inject_tx": r.inject_tx,
                "measure_tx": r.measure_tx,
                "measure_rx": r.measure_rx,
                "best_phase_deg": r.best_phase_deg,
                "best_amplitude": r.best_amplitude,
                "inject_gain_db": r.inject_gain_db,
                "min_metric": r.min_metric,
                "start_metric": r.start_metric,
                "null_depth_db": r.null_depth_db,
                "method": r.method,
                "elapsed_s": r.elapsed_s,
                "converged": r.converged,
            }
            for r in results
        ]

        if prev_cal:
            self._set_calibration_enabled(True)

        return results

    def get_data_sources(self):
        return set(self.mimo_sources) | set(self.cal_ref_sources)

    def _setup_display(self, display_config=None):
        if not self.data_output_queue:
            self.data_output_queue = mp.Queue(maxsize=DATA_OUTPUT_QUEUE_DEPTH)
        else:
            drain(self.data_output_queue)

        from bioview_server.common import DisplayWorker

        self.display_worker = DisplayWorker(
            display_sources=list(self.mimo_sources),
            data_input_queue=self.display_queue,
            data_output_queue=self.data_output_queue,
            logger=self.logger,
        )

    def _queue_param_update(self, params):
        for param, value in (params or {}).items():
            if param == "calibration.enabled":
                self._cal_enabled = bool(value)
            elif param == "calibration" and isinstance(value, dict):
                self._cal_enabled = bool(value.get("enabled", self._cal_enabled))
            elif param == "rx_gain" and isinstance(value, (list, tuple)):
                self.rx_gains_global = [float(v) for v in value]
            elif param == "tx_gain" and isinstance(value, (list, tuple)):
                self.tx_gains_global = [float(v) for v in value]

            if param in GLOBAL_TX_PARAMS and self.hardware:
                apply_global_tx_values_to_hardware(
                    self.hardware, param, value, self.group_config
                )
                self.group_config["hardware"] = self.hardware
            elif param in GLOBAL_RX_PARAMS and self.hardware:
                apply_global_rx_values_to_hardware(
                    self.hardware, param, value, self.group_config
                )
                self.group_config["hardware"] = self.hardware
            elif param == "hardware":
                self.hardware = dict(value)
                self.group_config["hardware"] = self.hardware

            is_tx = param in TX_PARAMS or param.startswith("calibration.")
            queues = (
                self.tx_command_queue.items() if is_tx else self.rx_command_queue.items()
            )
            for _device_key, q in queues:
                q.put({"param": param, "value": value})

    def _disconnect(self):
        self.stop_streaming()
        for device_key in list(self.usrp_handlers.keys()):
            self.usrp_handlers[device_key] = None
            self.usrp_states[device_key] = DeviceStatus.DISCONNECTED
            self.transmit_workers[device_key] = None
            self.receive_workers[device_key] = None
        self.rx_data_queue = {}
        self.rx_command_queue = {}
        drain(self.display_queue)
        drain(self.save_queue)
        return True


def build_hardware_dict_from_group(group_config, devices: Dict, group_id: str) -> Dict:
    """Build hardware dict from group config or legacy devices argument."""
    if group_config.get("hardware"):
        return dict(group_config["hardware"])
    if len(devices) == 1:
        device_name = list(devices.keys())[0]
        hw = devices[device_name]
        if isinstance(hw, dict):
            return {device_name: hw}
    result = {}
    for name, hw in devices.items():
        if isinstance(hw, dict):
            result[name] = hw
    return (
        result
        if result
        else build_hardware_dict(
            USRPConfiguration({**group_config, **list(devices.values())[0]}),
            group_id,
        )
    )
