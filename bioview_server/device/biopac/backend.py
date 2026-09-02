import contextlib
import multiprocessing as mp
from ctypes import c_int
from typing import List, Optional

from bioview_common import DataSource, DeviceStatus, log_print

from bioview_server.datatypes import Backend

from .acquire import BiopacAcquisitionWorker
from .utils import (
    build_hardware_dict_from_group,
    configure_biopac_device,
    connect_biopac_device,
    disconnect_biopac_device,
    load_mpdev_dll,
    resolve_biopac_hardware_entry,
    start_acq_daemon,
    start_acquisition,
    stop_acquisition,
)


BIOPAC_PARAM_KEYS = {
    "channels",
    "samp_rate",
    "model",
    "mpdev_path",
    "connection_type",
    "port",
    "disp_ds",
    "save_ds",
    "device_name",
}


class BIOPACBackend(Backend):
    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None,
        group_config: Optional[dict] = None,
        discovered_devices: Optional[dict] = None,
    ):
        super().__init__(
            group_id=group_id,
            response_queue=response_queue,
            data_output_queue=data_output_queue,
        )
        self.group_config = dict(group_config or {})
        self.discovered_devices = discovered_devices or {}
        self.hardware = build_hardware_dict_from_group(self.group_config, group_id)
        self.hw_entry = resolve_biopac_hardware_entry(
            self.hardware, self.discovered_devices
        )

        self.samp_rate = int(
            self.group_config.get("samp_rate", self.hw_entry.get("samp_rate", 1000))
        )
        self.model = self.hw_entry.get("model", self.group_config.get("model", "MP36"))
        self.mpdev_path = self.group_config.get("mpdev_path")
        self.connection_type = int(
            self.hw_entry.get(
                "connection_type",
                self.group_config.get("connection_type", 10),
            )
        )
        self.port = self.hw_entry.get("port", self.group_config.get("port", "auto"))
        self.device_code = int(
            self.hw_entry.get("device_code", self._model_code(self.model))
        )
        self.disp_ds = int(self.group_config.get("disp_ds", 10))
        self.save_ds = int(self.group_config.get("save_ds", 1))

        self.mpdev_handler = None
        self.channels: List[int] = []
        self._channel_array = None
        self.acquisition_worker: Optional[BiopacAcquisitionWorker] = None

        self.populate_data_sources()

    @staticmethod
    def _model_code(model: str) -> int:
        from bioview_common.datatypes.configuration.biopac import MODEL_CODE_MAPPING

        return MODEL_CODE_MAPPING.get(model, MODEL_CODE_MAPPING["MP36"])

    def _enabled_channels(self) -> List[int]:
        raw = self.hw_entry.get(
            "channels", self.group_config.get("channels", [1, 1, 1, 1])
        )
        return list(raw)

    def _acquired_channel_indices(self) -> List[int]:
        """Zero-based indices of the channels mpdev will actually return.

        mpdev packs one value per *enabled* channel into the sample buffer, so
        the number of acquired columns is the number of set entries in the
        channel mask -- not the length of the mask. Reading len(mask) values
        appended two uninitialized doubles per sample to every chunk whenever
        fewer than four channels were enabled.
        """
        return [idx for idx, enabled in enumerate(self._enabled_channels()) if enabled]

    def _ctypes_channels(self):
        padded = self._enabled_channels() + [0] * (16 - len(self._enabled_channels()))
        return (c_int * 16)(*padded[:16])

    def populate_data_sources(self):
        self.data_sources = set()
        labels = self.hw_entry.get("labels") or []
        enabled_idx = 0
        for idx, enabled in enumerate(self._enabled_channels()):
            if not enabled:
                continue
            enabled_idx += 1
            label = (
                labels[idx] if idx < len(labels) and labels[idx] else f"Ch{enabled_idx}"
            )
            # Every acquired sample is forwarded to the display (disp_ds only sets
            # the chunk size), so the display rate *is* the sample rate. Reporting
            # the default 200 Hz instead made the plot window scroll at the wrong
            # speed for any other sample rate.
            source = DataSource(
                group_id=self.group_id,
                channel=enabled_idx - 1,
                label=label,
                disp_freq=float(self.samp_rate),
            )
            self.data_sources.add(source)

    def _initialize(self):
        self.mpdev_handler = load_mpdev_dll(self.mpdev_path)
        if self.mpdev_handler is None:
            raise ValueError("A valid reference to mpdev.dll was not found")

        self.channels = self._enabled_channels()
        self._channel_array = self._ctypes_channels()

        try:
            connect_biopac_device(
                mpdev_handler=self.mpdev_handler,
                device_code=self.device_code,
                connection_code=self.connection_type,
                port=self.port,
            )
            configure_biopac_device(
                mpdev_handler=self.mpdev_handler,
                channels=self._channel_array,
                sample_rate=self.samp_rate,
            )
            self.status = DeviceStatus.CONNECTED
            log_print(self.logger, "debug", "Successfully initialized BIOPAC device")
            return True
        except Exception as e:
            # Raise rather than returning False. A falsy result is turned into a
            # generic "Unable to initialize device" upstream, which is all the
            # user ever saw; the real cause -- which mpdev error and what to do
            # about it -- is the whole point of the message.
            log_print(self.logger, "error", f"Unable to initialize BIOPAC device: {e}")
            raise

    def _start_streaming(self):
        if self.mpdev_handler is None:
            return False

        # The daemon has to be running before acquisition starts; it is what
        # makes the buffered stream read (receiveMPData) possible. If this DLL
        # or unit cannot provide it we fall back to per-sample polling, which
        # works but cannot sustain a high sample rate.
        use_stream = False
        try:
            use_stream = start_acq_daemon(self.mpdev_handler)
        except Exception as e:
            log_print(
                self.logger,
                "warning",
                f"Falling back to per-sample polling: {e}",
            )

        start_acquisition(self.mpdev_handler)
        self.status = DeviceStatus.STREAMING

        # Without these the acquisition worker fills self.display_queue and
        # nothing ever drains it into the client's output queue, so BIOPAC data
        # never reached the plots. USRP and dummy have always done this here.
        if self.save_worker is not None:
            if not self.save_worker.is_alive():
                self.save_worker.start()
            self.save_worker.resume()

        if self.display_worker is not None:
            if not self.display_worker.is_alive():
                self.display_worker.start()
            self.display_worker.resume()

        # Chunk size is the acquisition latency: with the buffered stream read
        # the worker blocks until this many samples exist. disp_ds sets it, but
        # a low disp_ds at a high sample rate would ask for a second of data per
        # read and make the plot lag by that much, so cap it at 100 ms.
        chunk_size = int(round(self.samp_rate / max(1, self.disp_ds)))
        chunk_size = max(1, min(chunk_size, max(1, int(self.samp_rate * 0.1))))
        self.acquisition_worker = BiopacAcquisitionWorker(
            mpdev_handler=self.mpdev_handler,
            channels=self._acquired_channel_indices(),
            samp_rate=self.samp_rate,
            display_queue=self.display_queue,
            save_queue=self.save_queue,
            chunk_size=chunk_size,
            use_stream=use_stream,
            logger=self.logger,
        )
        self.acquisition_worker.start()
        self.acquisition_worker.resume()
        log_print(
            self.logger,
            "debug",
            f"BIOPAC streaming at {self.samp_rate} Hz using "
            f"{'the acquisition daemon' if use_stream else 'per-sample polling'} "
            f"({chunk_size} samples/chunk)",
        )
        return True

    def _stop_streaming(self):
        if self.display_worker is not None:
            self.display_worker.pause()
        if self.save_worker is not None:
            self.save_worker.pause()

        if self.acquisition_worker is not None:
            self.acquisition_worker.stop()
            self.acquisition_worker.join(timeout=2)
            self.acquisition_worker = None

        if self.mpdev_handler is not None:
            try:
                stop_acquisition(self.mpdev_handler)
            except Exception as e:
                log_print(self.logger, "error", f"BIOPAC stopping error: {e}")

        self.status = DeviceStatus.CONNECTED
        return True

    def _queue_param_update(self, params: dict):
        if not params:
            return

        restart_streaming = False
        for param, value in params.items():
            if param == "samp_rate":
                self.samp_rate = int(value)
                self.group_config["samp_rate"] = self.samp_rate
                self.hw_entry["samp_rate"] = self.samp_rate
                self.populate_data_sources()
                if self.display_worker is not None:
                    self.display_worker.set_display_sources(list(self.data_sources))
                restart_streaming = True
            elif param == "channels":
                self.hw_entry["channels"] = list(value)
                self.group_config["channels"] = list(value)
                self.channels = self._enabled_channels()
                self._channel_array = self._ctypes_channels()
                self.populate_data_sources()
                # The display worker labels each emitted row; leaving it with the
                # old source list would mislabel (or silently drop) every plot
                # after a channel change.
                if self.display_worker is not None:
                    self.display_worker.set_display_sources(list(self.data_sources))
                restart_streaming = True
            elif param == "model":
                self.model = str(value)
                self.device_code = self._model_code(self.model)
                self.group_config["model"] = self.model
            elif param == "mpdev_path":
                self.mpdev_path = value
                self.group_config["mpdev_path"] = value
            elif param == "connection_type":
                self.connection_type = int(value)
                restart_streaming = True
            elif param == "port":
                self.port = value
                restart_streaming = True
            elif param == "disp_ds":
                self.disp_ds = int(value)
            elif param == "save_ds":
                self.save_ds = int(value)

        if self.mpdev_handler is not None and self._channel_array is not None:
            try:
                configure_biopac_device(
                    mpdev_handler=self.mpdev_handler,
                    channels=self._channel_array,
                    sample_rate=self.samp_rate,
                )
            except Exception as e:
                log_print(self.logger, "warning", f"BIOPAC param apply failed: {e}")

        if restart_streaming and self.status == DeviceStatus.STREAMING:
            self._stop_streaming()
            self._start_streaming()

    def _apply_param_update_local(self, params: dict):
        """Mirror channel/label/rate changes on the parent side.

        ``get_data_sources()`` is answered out of the parent process, so a
        channel the user just enabled has to be reflected here as well as in the
        child -- otherwise the server keeps advertising the old channel list and
        the plot-source selector never changes.
        """
        params = dict(params or {})
        touched = False

        # Applied first: it replaces hw_entry wholesale, so a channels/labels
        # update arriving in the same call must land on the new entry.
        hardware = params.pop("hardware", None)
        if isinstance(hardware, dict) and hardware:
            self.hardware = dict(hardware)
            self.hw_entry = resolve_biopac_hardware_entry(
                self.hardware, self.discovered_devices
            )
            touched = True

        for param, value in params.items():
            if param == "channels":
                self.hw_entry["channels"] = list(value)
                self.group_config["channels"] = list(value)
                touched = True
            elif param == "labels":
                self.hw_entry["labels"] = list(value)
                touched = True
            elif param == "samp_rate":
                self.samp_rate = int(value)
                self.group_config["samp_rate"] = self.samp_rate
                self.hw_entry["samp_rate"] = self.samp_rate
                touched = True

        if touched:
            self.populate_data_sources()

    def _disconnect(self):
        self._stop_streaming()
        if self.mpdev_handler is not None:
            with contextlib.suppress(Exception):
                disconnect_biopac_device(self.mpdev_handler)
        self.mpdev_handler = None
        self.status = DeviceStatus.DISCONNECTED
        return True
