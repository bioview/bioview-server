import queue
import multiprocessing as mp
import time

from typing import Dict, List

import os

os.environ["UHD_LOG_LEVEL"] = "error"

import uhd
from bioview_common import (
    DataSource,
    USRPConfiguration,
    log_print,
    DeviceStatus,
)
from bioview_common.datatypes.configuration.usrp_channel_map import (
    build_hardware_dict,
    resolve_channel_map,
    resolve_device_serial,
)
from bioview_common.datatypes.configuration.hardware_params import (
    GLOBAL_RX_PARAMS,
    GLOBAL_TX_PARAMS,
    apply_global_rx_values_to_hardware,
    apply_global_tx_values_to_hardware,
)
from bioview_common.signal_schemes import DpicBalancer, scheme_from_config

from bioview_server.datatypes import Backend

from .process import ProcessWorker
from .receive import ReceiveWorker
from .transmit import TransmitWorker, TX_PARAMS
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

        self.global_tx_to_device = {}
        self.global_rx_offsets = {}
        tx_offset = 0
        rx_offset = 0
        for device_name, hw in self.hardware.items():
            n_tx = len(hw.get("tx_channels", [0]))
            n_rx = len(hw.get("rx_channels", [0]))
            self.global_tx_offsets[device_name] = tx_offset
            self.global_rx_offsets[device_name] = rx_offset
            for local in range(n_tx):
                self.global_tx_to_device[tx_offset + local] = (device_name, local)
            tx_offset += n_tx
            rx_offset += n_rx

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
                return False

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
                    return False

                self.usrp_handlers[device_name] = response["usrp"]
                self.rx_data_queue[device_name] = queue.Queue()
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
                log_print(self.logger, "error", f"Unable to initialize device: {e}")
                return False

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

        time.sleep(FILLING_TIME)

        dpic_cfg = self.group_config.get("dpic_balance", {})
        if dpic_cfg.get("auto_on_start") and self.dpic_pairs:
            self._run_dpic_balance()

        if self._cal_enabled_at_start:
            for scheme in self.schemes_by_device.values():
                scheme.set_calibration_enabled(True)
        else:
            for scheme in self.schemes_by_device.values():
                scheme.set_calibration_enabled(False)

        if self.process_worker:
            if not self.process_worker.is_alive():
                self.process_worker.start()
            self.process_worker.resume()

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

    def _run_dpic_balance(self):
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
            self.transmit_workers[dev_name].set_global_tx_param(
                global_tx, "phase", phase_deg
            )

        def set_amplitude(global_tx, amp):
            dev_name, local = self.global_tx_to_device[global_tx]
            self.transmit_workers[dev_name].set_global_tx_param(
                global_tx, "amplitude", amp
            )

        def wait_settle():
            time.sleep(dpic_cfg.get("settle_time_s", 0.5))

        def read_metric(inject_tx, measure_tx):
            target_rx = measure_tx
            return self.process_worker.get_metric(measure_tx, target_rx)

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
        return set(self.mimo_sources) | set(self.cal_ref_sources)

    def _setup_display(self, display_config=None):
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

    def _queue_param_update(self, params):
        for param, value in (params or {}).items():
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
                self.tx_command_queue.items()
                if is_tx
                else self.rx_command_queue.items()
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
        if self.display_queue:
            while not self.display_queue.empty():
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    break
        if self.save_queue:
            while not self.save_queue.empty():
                try:
                    self.save_queue.get_nowait()
                except queue.Empty:
                    break
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
    return result if result else build_hardware_dict(
        USRPConfiguration({**group_config, **list(devices.values())[0]}),
        group_id,
    )
