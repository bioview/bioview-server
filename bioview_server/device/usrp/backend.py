import multiprocessing as mp
import os
import queue
import threading
import time


os.environ["UHD_LOG_LEVEL"] = "error"

# See the note in receive.py: the backend module must import without UHD so
# its pure-logic paths stay reachable (and testable) on machines and CI
# runners with no driver. Every uhd use below is inside a function.
try:
    import uhd
except ImportError:  # pragma: no cover - no USRP driver installed
    uhd = None
from bioview_common import (
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
    apply_global_values_to_hardware,
    build_global_mapping,
)
from bioview_common.datatypes.configuration.usrp_channel_map import (
    build_hardware_dict,
    components_from_config,
    resolve_channel_map,
    resolve_device_serial,
)
from bioview_common.signal_schemes import (
    DpicChannel,
    scheme_from_config,
)

from bioview_server.common import balance_outcome, build_balancer
from bioview_server.datatypes import Backend

from .process import ProcessWorker
from .receive import ReceiveWorker
from .transmit import TX_PARAMS, TransmitWorker


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
    # Deferred: .utils imports uhd unconditionally (device/__init__ uses that
    # import to decide whether the USRP backend is available at all).
    from .utils import check_channels, setup_pps, setup_ref

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

    tune = uhd.types.TuneRequest(float(carrier_freq))

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
        devices: dict,
        group_config: dict,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None,
        save_output_queue: mp.Queue = None,
        display_ds: int = 10,
        display_imaginary: bool = False,
        save_ds: int = 10,
        save_iq: bool = False,
        save_imaginary: bool = True,
        discovered_devices: dict = None,
    ):
        super().__init__(
            group_id=group_id,
            response_queue=response_queue,
            data_output_queue=data_output_queue,
            save_output_queue=save_output_queue,
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
        # Built by _initialize(); _display_sources() already reads it as
        # possibly-None, so it must exist before the device is connected.
        self.process_worker = None
        self.schemes_by_device = {}
        self.global_tx_to_device = {}
        self.global_tx_offsets = {}
        self.global_rx_offsets = {}
        self.dpic_pairs = []
        self.mimo_sources = set()
        self.stream_components = ["amplitude"]
        self.cal_ref_sources = []
        self.registry = None
        self.rx_device_order = []
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

    #: Rebuilt in the child rather than shipped to it; see _init_local_state.
    _LOCAL_STATE_KEYS = ("_gain_lock",)

    def _init_local_state(self):
        super()._init_local_state()
        # The gain lists are group-wide but written per channel, and a
        # parallel balance has one thread per radio writing them at once. The
        # lock covers the read-modify-write and the hardware apply that
        # follows it, which re-sends the *whole* list. A lock cannot be
        # pickled, so it is created on each side of the spawn rather than
        # shipped: without this the whole handler failed to pickle and the
        # group came up Unavailable.
        self._gain_lock = threading.Lock()

    def get_display_frequency(self) -> float:
        """Rate (Hz) at which the pipeline emits display samples.

        ProcessWorker averages ``save_ds`` raw samples per output point and the
        display path drops that again by ``display_ds``. The client sizes its
        plot ring buffer from this number, so anything else makes the trace
        scroll at the wrong speed.
        """
        divisor = max(1, int(self.save_ds)) * max(1, int(self.display_ds))
        return float(self.samp_rate) / divisor

    def get_save_freq(self) -> float:
        """Saved rate: ProcessWorker averages ``save_ds`` raw samples per point.

        The display path decimates this again by ``display_ds``; the recording
        keeps the undecimated save stream.
        """
        return float(self.samp_rate) / max(1, int(self.save_ds))

    def populate_data_sources(self):
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

    def _resolve_serial(self, device_name: str, hw_entry: dict) -> str:
        from .utils import get_usrp_address

        return resolve_device_serial(
            device_name,
            hw_entry,
            self.discovered_devices,
            get_usrp_address,
        )

    def _initialize(self):
        from .utils import discover_devices

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
                # Bounded: an unbounded queue grows without limit whenever the
                # ProcessWorker falls behind.
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
            display_ds=self.display_ds,
            record_cal_ref=bool(
                self.group_config.get("calibration", {}).get("record_reference", True)
            ),
            logger=self.logger,
        )
        return True

    def _setup_saving(self, save_config: dict):
        super()._setup_saving(save_config)
        self.process_worker.save_imaginary = self.save_imaginary
        self.process_worker.save_iq = self.save_iq
        self.process_worker.save_ds = self.save_ds
        self.process_worker.save_queue = self.save_queue
        # save_ds sets the display rate too, so the advertised disp_freq has to
        # follow it or the client keeps sizing buffers for the old rate.
        self._refresh_display_frequency()

    def _refresh_display_frequency(self):
        disp_freq = self.get_display_frequency()
        for source in list(self.mimo_sources) + list(self.cal_ref_sources):
            source.disp_freq = disp_freq
        if self.display_worker is not None:
            self.display_worker.set_display_sources(self._display_sources())

    def _workers_in_start_order(self):
        """Every worker thread this device runs, paired with a name for logs."""
        workers = [
            (f"transmit worker {name}", worker)
            for name, worker in self.transmit_workers.items()
        ]
        workers += [
            (f"receive worker {name}", worker)
            for name, worker in self.receive_workers.items()
        ]
        workers += [
            (label, worker)
            for label, worker in (
                ("process worker", self.process_worker),
                ("save worker", self.save_worker),
                ("display worker", self.display_worker),
            )
            if worker is not None
        ]
        return workers

    def _start_streaming(self):
        # Every thread is created before *any* of them is resumed, and the two
        # halves are not interleaved.
        #
        # `Thread.start()` does not return until the new thread has been
        # scheduled and has run far enough to set its started event, and it
        # waits for that without a timeout. Starting a thread after the
        # transmit and receive threads are already spinning inside UHD means
        # queueing for the GIL behind two tight native loops, and the start
        # does not complete: the backend never answers START_STREAMING, the
        # server times out after 90 s and tears down every other device in the
        # session, so a working BIOPAC plots nothing either.
        #
        # Workers are constructed paused (``PausableWorker(running=False)``),
        # so bringing them all up first costs nothing and leaves `resume()` as
        # a bare `Event.set()` with no scheduling to wait on. The steps are
        # traced because a stall here names no line by itself.
        step = self._StepTimer(self.logger, "start_streaming")

        for label, worker in self._workers_in_start_order():
            if not worker.is_alive():
                worker.start()
            step.mark(f"{label} thread up")

        # Resume order, unlike start order, is load-bearing: the process worker
        # drains the Rx queues, and DPIC balance has no metrics to read until it
        # has produced some.
        for worker in self.transmit_workers.values():
            worker.resume()
        for worker in self.receive_workers.values():
            worker.resume()
        if self.process_worker:
            self.process_worker.resume()
        step.mark("radio running")

        time.sleep(FILLING_TIME)
        step.mark("filling delay")

        # The current calibration state, not the config's start-up value: the
        # overlay is toggled at runtime and would be reset on every Start.
        self._set_calibration_enabled(self._cal_enabled)
        step.mark("calibration state")

        if self.save_worker:
            self.save_worker.resume()
        if self.display_worker:
            self.display_worker.resume()
        step.mark("consumers running")

        step.done()

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
            # On its own thread, like a requested balance: this runs after the
            # START_STREAMING reply but still on the command loop, so calling
            # the search inline here left Stop unservable for its duration.
            self._start_balance_thread()

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

    def _apply_channel_if(self, global_tx: int, freq: float):
        """Move one Tx onto ``freq``, everywhere the IF is held.

        Four places, all of which must agree or the tone is generated at one
        frequency and looked for at another: the backend's own list (shared by
        reference with the ProcessWorker), the ProcessWorker's band-pass, the
        transmit workers' waveform generators, and the config the client reads
        back.
        """
        freq = float(freq)
        self.channel_ifs[global_tx] = freq
        if global_tx < len(self.registry.tx_if_freq):
            self.registry.tx_if_freq[global_tx] = freq

        # Config, so the settings panel and a saved config show the IF the
        # radio is actually driven at.
        dev_name, local = self.global_tx_to_device[global_tx]
        hw = self.hardware.get(dev_name)
        if hw is not None:
            if_freqs = list(hw.get("if_freq", []) or [])
            while len(if_freqs) <= local:
                if_freqs.append(freq)
            if_freqs[local] = freq
            hw["if_freq"] = if_freqs
            self.group_config["hardware"] = self.hardware

        if self.process_worker is not None:
            self.process_worker.set_channel_if(global_tx, freq)

        # Whole-group list: each transmit worker slices its own window out of
        # it, so every device has to be told.
        for q in self.tx_command_queue.values():
            q.put({"param": "if_freq", "value": list(self.channel_ifs)})

    def _coerce_dpic_inject_frequencies(self):
        """Put every inject Tx on its measure Tx's IF before balancing.

        The receive chain band-passes around the measure Tx's IF, so an inject
        Tx anywhere else is rejected by that filter and no phase/amplitude can
        cancel the direct path (see the rule in ENGINEERING_NOTES 3.5). A
        mismatch used to be reported as an error and the balance ran anyway,
        sweeping 241 points against a tone the Rx could not see; the frequency
        is a consequence of which pair is being cancelled, not an independent
        setting, so it is coerced rather than complained about.
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
                # Only reachable if the coercion above could not run.
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

    def _tx_gain_range(self, device_name: str):
        usrp = self.usrp_handlers.get(device_name)
        try:
            rng = usrp.get_tx_gain_range()
            return (float(rng.start()), float(rng.stop()))
        except Exception:
            return (0.0, 89.75)

    def _set_global_rx_gain(self, global_rx: int, value: float):
        """Apply one Rx channel's analog gain, keeping the group list in step.

        The command carries the whole global list because that is what the
        ReceiveWorker's ``rx_gain`` handler consumes.
        """
        entry = self.global_rx_to_device.get(global_rx)
        if entry is None or global_rx >= len(self.rx_gains_global):
            return
        dev_name, _local = entry
        with self._gain_lock:
            self.rx_gains_global[global_rx] = float(value)
            gains = list(self.rx_gains_global)
            apply_global_values_to_hardware(
                self.hardware, "rx_gain", gains, self.group_config, kind="rx"
            )
            self.rx_command_queue[dev_name].put({"param": "rx_gain", "value": gains})

    def _set_global_tx_gain(self, global_tx: int, value: float):
        """Apply one Tx channel's analog gain through its transmit worker."""
        if global_tx >= len(self.tx_gains_global):
            return
        dev_name, _local = self.global_tx_to_device[global_tx]
        with self._gain_lock:
            self.tx_gains_global[global_tx] = float(value)
            apply_global_values_to_hardware(
                self.hardware,
                "tx_gain",
                list(self.tx_gains_global),
                self.group_config,
                kind="tx",
            )
        self.transmit_workers[dev_name].set_global_tx_param(global_tx, "gain", value)

    def _build_dpic_channel(self, pair, dpic_cfg) -> DpicChannel:
        settle_s = float(dpic_cfg.get("settle_time_s", 0.02))
        measure_rx = pair.target_rx
        measure_tx = pair.measure_tx

        dev_name, _ = self.global_tx_to_device[pair.inject_tx]
        worker = self.transmit_workers[dev_name]
        # Metric freshness timeout: the longest dwell the balancer can ask for,
        # plus slack, so a silent path fails fast without a slow chunk reading
        # as "no data".
        read_timeout = max(2.0, settle_s * 8, float(dpic_cfg.get("read_timeout_s", 0)))

        rx_entry = self.global_rx_to_device.get(measure_rx)
        rx_device = rx_entry[0] if rx_entry else dev_name
        tx_device = self.global_tx_to_device[measure_tx][0]

        return DpicChannel(
            inject_tx=pair.inject_tx,
            measure_tx=measure_tx,
            measure_rx=measure_rx,
            # The radio the loop lives on, so ``balance_all`` can run one lane
            # per radio at the same time. The inject Tx names it: that is the
            # channel the search actually drives.
            device=dev_name,
            set_phase=lambda v: worker.set_global_tx_param(pair.inject_tx, "phase", v),
            set_amplitude=lambda v: worker.set_global_tx_param(
                pair.inject_tx, "amplitude", v
            ),
            get_gain=lambda: self._get_inject_gain(pair.inject_tx),
            # Waits for chunks captured *after* the change, so the search is
            # never biased by the Rx buffering latency.
            read_metric=lambda: self.process_worker.wait_for_metric(
                measure_tx, measure_rx, min_new=2, timeout=read_timeout
            ),
            # The balancer names the dwell; the VI's per-point waits.
            wait_settle=time.sleep,
            # The VI's gain stage moves the measure Tx and the Rx together.
            get_rx_gain=lambda: float(self.rx_gains_global[measure_rx]),
            set_rx_gain=lambda v: self._set_global_rx_gain(measure_rx, v),
            get_tx_gain=lambda: float(self.tx_gains_global[measure_tx]),
            set_tx_gain=lambda v: self._set_global_tx_gain(measure_tx, v),
            rx_gain_range=self._rx_gain_range(rx_device),
            tx_gain_range=self._tx_gain_range(tx_device),
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

    def _run_dpic_balance(self):
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

        # Before validation: the IF mismatch it warns about is the one this
        # resolves, and before the settle below, so the retuned waveform is on
        # the air by the time the first metric is read.
        self._coerce_dpic_inject_frequencies()
        self._validate_dpic_pairs()

        dpic_cfg = self.group_config.get("dpic_balance", {})
        settle_s = float(dpic_cfg.get("settle_time_s", 0.02))

        # The calibration overlay modulates the amplitude the search minimizes,
        # so it is disabled for the duration and restored to its actual state.
        prev_cal = self._cal_enabled
        if prev_cal:
            self._set_calibration_enabled(False)
        # One settle covers both the calibration overlay coming off and any IF
        # coercion above landing on the next transmit buffer.
        time.sleep(settle_s)

        balancer = build_balancer(
            dpic_cfg,
            should_abort=self.balance_aborted,
            on_progress=self.publish_balance_progress,
        )

        channels = [self._build_dpic_channel(p, dpic_cfg) for p in self.dpic_pairs]
        radios = sorted({ch.device for ch in channels if ch.device})
        log_print(
            self.logger,
            "info",
            f"[DPIC] Balancing {len(channels)} loop(s) across {len(radios)} radio(s)"
            + (
                f" in parallel: {', '.join(radios)}"
                if balancer.parallel_devices and len(radios) > 1
                else ""
            ),
        )
        results = balancer.balance_all(channels)
        outcome = balance_outcome(self.logger, results)

        if "dpic_balance" not in self.group_config:
            self.group_config["dpic_balance"] = {}
        self.group_config["dpic_balance"]["last_results"] = outcome["results"]

        if prev_cal:
            self._set_calibration_enabled(True)

        return outcome

    def get_data_sources(self):
        return set(self.mimo_sources) | set(self.cal_ref_sources)

    def _display_sources(self):
        """Rows the ProcessWorker emits, in channel order.

        Cal-ref sources are advertised by ``get_data_sources()``, so they must
        appear in the display payload too or their plots never get a row.
        """
        sources = list(self.mimo_sources)
        if self.process_worker is None or self.process_worker.record_cal_ref:
            sources += list(self.cal_ref_sources)
        return sorted(sources, key=lambda s: s.channel)

    def _reload_channel_map(self, channel_map):
        """Rebuild everything the channel map decides, in place.

        DPIC pairs are *specified* in the channel map, and the channel map is
        edited in the settings panel -- so this is the path a pair actually
        arrives by. It used to fall through `_queue_param_update` untouched:
        the value never reached `group_config`, `populate_data_sources()` was
        never re-run, and `dpic_pairs` kept whatever the config file had at
        connect time. Adding a pair in the UI and pressing Balance therefore
        reported "No DPIC pairs are configured".
        """
        if self._streaming.is_set():
            # The map decides how many rows the pipeline emits, so changing it
            # mid-recording would desynchronise the file from its own header.
            log_print(
                self.logger,
                "error",
                "[USRP] Channel map changed while streaming; it takes "
                "effect on the next Start.",
            )
            return False

        self.group_config["channel_map"] = channel_map

        # Schemes are a function of `hardware`, which a channel-map edit never
        # touches, and the live transmit workers hold these exact objects --
        # letting populate_data_sources() replace them would leave the workers
        # driving detached schemes while the ProcessWorker demodulated against
        # the new ones. Same for the runtime gains, which auto-gain moves
        # without writing back to `hardware`.
        preserved_schemes = dict(self.schemes_by_device)
        preserved_rx_gains = list(self.rx_gains_global)
        preserved_tx_gains = list(self.tx_gains_global)
        cal_enabled = self._cal_enabled

        self.populate_data_sources()

        for device_name, scheme in preserved_schemes.items():
            self.schemes_by_device[device_name] = scheme
        if len(preserved_rx_gains) == len(self.rx_gains_global):
            self.rx_gains_global = preserved_rx_gains
        if len(preserved_tx_gains) == len(self.tx_gains_global):
            self.tx_gains_global = preserved_tx_gains
        self._cal_enabled = cal_enabled

        if self.process_worker is not None:
            self.process_worker.set_sources(
                self.mimo_sources,
                self.cal_ref_sources,
                self.channel_ifs,
                self.if_filter_bw,
            )
        self._refresh_display_frequency()

        log_print(
            self.logger,
            "info",
            f"[USRP] Channel map reloaded: {len(self.mimo_sources)} "
            f"measurement source(s), {len(self.dpic_pairs)} DPIC pair(s)",
        )
        return True

    def _apply_param_update_local(self, params):
        """Parent-side mirror of the channel map.

        ``get_data_sources()`` is answered out of the parent process, and the
        parent's own `dpic_pairs` is what the balance command is dispatched
        against, so the new map has to land here as well as in the child.
        There are no workers on this side, so the plain rebuild is enough.
        """
        channel_map = (params or {}).get("channel_map")
        if isinstance(channel_map, dict):
            self.group_config["channel_map"] = channel_map
            self.populate_data_sources()

    def _queue_param_update(self, params):
        for param, value in (params or {}).items():
            if param == "channel_map" and isinstance(value, dict):
                # Not forwarded to a worker queue: nothing down there reads it,
                # and everything it decides is rebuilt here.
                self._reload_channel_map(value)
                continue

            if param == "calibration.enabled":
                self._cal_enabled = bool(value)
                self.group_config.setdefault("calibration", {})[
                    "enabled"
                ] = self._cal_enabled
            elif param == "calibration" and isinstance(value, dict):
                self._cal_enabled = bool(value.get("enabled", self._cal_enabled))
                self.group_config["calibration"] = dict(value)
            elif param.startswith("calibration."):
                key = param.split(".", 1)[1]
                self.group_config.setdefault("calibration", {})[key] = value
            elif param == "rx_gain" and isinstance(value, list | tuple):
                self.rx_gains_global = [float(v) for v in value]
            elif param == "tx_gain" and isinstance(value, list | tuple):
                self.tx_gains_global = [float(v) for v in value]

            if param in GLOBAL_TX_PARAMS and self.hardware:
                apply_global_values_to_hardware(
                    self.hardware, param, value, self.group_config, kind="tx"
                )
                self.group_config["hardware"] = self.hardware
            elif param in GLOBAL_RX_PARAMS and self.hardware:
                apply_global_values_to_hardware(
                    self.hardware, param, value, self.group_config, kind="rx"
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


def build_hardware_dict_from_group(group_config, devices: dict, group_id: str) -> dict:
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
