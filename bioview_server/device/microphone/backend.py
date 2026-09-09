import contextlib
import multiprocessing as mp
import queue

from bioview_common import DataSource, DeviceStatus, log_print

from bioview_server.datatypes import Backend

from .acquire import MicrophoneAcquisitionWorker
from .utils import (
    build_hardware_dict_from_group,
    describe_input,
    negotiate_samplerate,
    resolve_hardware_entry,
    resolve_input_device,
    supports_input,
)


#: Longest capture chunk, in seconds. The chunk is the acquisition latency, so
#: a large blocksize would delay every plot update by the same amount.
MAX_CHUNK_SECONDS = 0.1

#: Chunk used when nothing is configured: 10 per second, matching the BIOPAC
#: backend's default so a mixed session's chunks arrive at a similar cadence.
DEFAULT_CHUNKS_PER_SECOND = 10


class MicrophoneBackend(Backend):
    """Host audio input as an ordinary BioView device.

    One row per captured channel at ``samp_rate``, emitted through the standard
    display path -- which is also the save path -- so speech is recorded
    sample-aligned with the RF and physiological rows in the same ``.bvr``.
    """

    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None,
        save_output_queue: mp.Queue = None,
        group_config: dict | None = None,
        discovered_devices: dict | None = None,
    ):
        super().__init__(
            group_id=group_id,
            response_queue=response_queue,
            data_output_queue=data_output_queue,
            save_output_queue=save_output_queue,
        )
        self.group_config = dict(group_config or {})
        self.discovered_devices = discovered_devices or {}
        self.hardware = build_hardware_dict_from_group(self.group_config, group_id)
        self.hw_key, self.hw_entry = resolve_hardware_entry(
            self.hardware, self.discovered_devices
        )

        self.samp_rate = int(self._setting("samp_rate", 16000))
        self.channel_count = self._channel_count()
        self.device = self._setting("device", "default")
        self.blocksize = int(self._setting("blocksize", 0) or 0)
        self.gain = float(self._setting("gain", 1.0))

        self.stream = None
        self.device_index = None
        self.acquisition_worker: MicrophoneAcquisitionWorker | None = None

        # Negotiated here, in the parent, rather than when the stream opens in
        # the child: ``get_data_sources()`` is answered out of the parent, so a
        # rate settled on in the child would leave the advertised ``disp_freq``
        # -- which is the timebase written into the recording header -- stale.
        self._negotiate_samp_rate()
        self.populate_data_sources()

    # ------------------------------------------------------------- settings

    def _setting(self, key, default=None):
        """Hardware entry first, then the group block, then the default.

        The same precedence BIOPAC uses: a UI edit is written to both, and the
        nested entry is the one that describes a specific physical device.
        """
        if key in self.hw_entry and self.hw_entry[key] is not None:
            return self.hw_entry[key]
        value = self.group_config.get(key, default)
        return default if value is None else value

    def _channel_count(self) -> int:
        raw = self._setting("channels", 1)
        if isinstance(raw, list | tuple):
            # A config written against the BIOPAC enable-mask shape.
            return max(1, sum(1 for entry in raw if entry))
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 1

    def _negotiate_samp_rate(self):
        """Settle ``samp_rate`` against what the input will actually run at.

        Failures are swallowed: this runs during construction, before the
        device has been asked for anything, and a missing microphone must
        surface at Connect with its own message rather than as a constructor
        that raised.
        """
        try:
            index = resolve_input_device(self.device, self.logger)
            rate = negotiate_samplerate(
                index, self.channel_count, self.samp_rate, self.logger
            )
        except Exception as e:
            log_print(
                self.logger,
                "debug",
                f"Could not probe audio input rates ({e}); keeping the "
                f"configured {self.samp_rate} Hz",
            )
            return
        self.samp_rate = int(round(rate))

    def _frames_per_read(self) -> int:
        if self.blocksize > 0:
            frames = self.blocksize
        else:
            frames = int(round(self.samp_rate / DEFAULT_CHUNKS_PER_SECOND))
        cap = max(1, int(self.samp_rate * MAX_CHUNK_SECONDS))
        return max(1, min(frames, cap))

    def populate_data_sources(self):
        self.data_sources = set()
        labels = self._setting("labels", None) or []
        for idx in range(self.channel_count):
            label = (
                labels[idx]
                if idx < len(labels) and labels[idx]
                else (f"Audio{idx + 1}" if self.channel_count > 1 else "Audio")
            )
            # Every captured sample is emitted, so the display rate is the
            # sample rate. Saving is fed from this same stream, so decimating
            # here would decimate the recording too.
            self.data_sources.add(
                DataSource(
                    group_id=self.group_id,
                    channel=idx,
                    label=label,
                    disp_freq=float(self.samp_rate),
                )
            )

    # -------------------------------------------------------------- control

    def _open_stream(self):
        """Open the input, wired to the acquisition worker's callback.

        The worker is built first because the callback belongs to it, and it
        outlives Stop/Start: a PausableWorker is a thread, so it is started once
        and paused thereafter. The stream is opened here but *not* started -- a
        running stream with a paused worker only fills the capture queue with
        audio nobody is going to emit.
        """
        import sounddevice as sd

        self.device_index = resolve_input_device(self.device, self.logger)
        # Deliberately not sd.check_input_settings: under WDM-KS it answers
        # about the format without answering about the device, and accepts
        # settings that Pa_OpenStream then refuses outright. The open below is
        # the only check that means anything; this one exists to turn its
        # failure into a message that names what was asked for and what the
        # likely cause is.
        if not supports_input(self.device_index, self.channel_count, self.samp_rate):
            name = describe_input(self.device_index)
            raise RuntimeError(
                f"{name} will not open for {self.channel_count} channel(s) at "
                f"{self.samp_rate:.0f} Hz. On Windows an input that enumerates "
                "but refuses to open usually has nothing plugged into its jack, "
                "or is disabled in Sound settings. Name a different input with "
                '"device" in the configuration if one is attached'
            )

        self.acquisition_worker = MicrophoneAcquisitionWorker(
            channels=self.channel_count,
            samp_rate=self.samp_rate,
            display_queue=self.display_queue,
            save_queue=self.save_queue,
            gain=self.gain,
            logger=self.logger,
        )
        self.stream = sd.InputStream(
            device=self.device_index,
            channels=self.channel_count,
            samplerate=self.samp_rate,
            dtype="float32",
            blocksize=self._frames_per_read(),
            callback=self.acquisition_worker.callback,
        )

    def _initialize(self):
        try:
            self._open_stream()
            self.status = DeviceStatus.CONNECTED
            log_print(
                self.logger,
                "debug",
                f"Opened audio input {self.device_index} "
                f"({self.channel_count}ch @ {self.samp_rate} Hz)",
            )
            return True
        except Exception as e:
            # Raised, not returned falsy: upstream turns a falsy result into a
            # generic "unable to initialize" and loses the PortAudio reason.
            self.status = DeviceStatus.DISCONNECTED
            log_print(self.logger, "error", f"Unable to open audio input: {e}")
            self._teardown_capture()
            raise

    def _teardown_capture(self):
        """Close the stream and retire the worker bound to its callback."""
        if self.stream is not None:
            with contextlib.suppress(Exception):
                self.stream.stop()
            with contextlib.suppress(Exception):
                self.stream.close()
            self.stream = None

        if self.acquisition_worker is not None:
            self.acquisition_worker.stop()
            if self.acquisition_worker.is_alive():
                self.acquisition_worker.join(timeout=2)
            self.acquisition_worker = None

    def _start_streaming(self):
        if self.stream is None or self.acquisition_worker is None:
            return False

        if self.save_worker is not None:
            if not self.save_worker.is_alive():
                self.save_worker.start()
            self.save_worker.resume()

        if self.display_worker is not None:
            if not self.display_worker.is_alive():
                self.display_worker.start()
            self.display_worker.resume()

        # Anything captured before Start is stale by the time it is pressed, and
        # emitting it would offset the whole recording against the devices
        # starting alongside it.
        with contextlib.suppress(queue.Empty):
            while True:
                self.acquisition_worker.capture_queue.get_nowait()

        if not self.acquisition_worker.is_alive():
            self.acquisition_worker.start()
        self.acquisition_worker.resume()
        self.stream.start()

        self.status = DeviceStatus.STREAMING
        log_print(
            self.logger,
            "debug",
            f"Microphone streaming at {self.samp_rate} Hz "
            f"({self._frames_per_read()} frames/chunk)",
        )
        return True

    def _stop_streaming(self):
        if self.display_worker is not None:
            self.display_worker.pause()
        if self.save_worker is not None:
            self.save_worker.pause()

        # The stream is stopped rather than closed: Start/Stop is pressed many
        # times a session and reopening a PortAudio device each time is both
        # slow and a chance for another application to take it in between.
        if self.stream is not None:
            with contextlib.suppress(Exception):
                self.stream.stop()
        if self.acquisition_worker is not None:
            self.acquisition_worker.pause()

        if self.stream is not None:
            self.status = DeviceStatus.CONNECTED
        return True

    def _queue_param_update(self, params: dict):
        if not params:
            return

        reopen = False
        for param, value in params.items():
            if param == "samp_rate":
                self.samp_rate = int(value)
                self.group_config["samp_rate"] = self.samp_rate
                self.hw_entry["samp_rate"] = self.samp_rate
                self.populate_data_sources()
                reopen = True
            elif param == "channels":
                self.hw_entry["channels"] = value
                self.group_config["channels"] = value
                self.channel_count = self._channel_count()
                self.populate_data_sources()
                reopen = True
            elif param == "device":
                self.device = value
                self.group_config["device"] = value
                self.hw_entry["device"] = value
                reopen = True
            elif param == "blocksize":
                self.blocksize = int(value or 0)
                self.group_config["blocksize"] = self.blocksize
                reopen = True
            elif param == "gain":
                # Applied per chunk, so it takes effect without reopening.
                self.gain = float(value)
                self.group_config["gain"] = self.gain
                self.hw_entry["gain"] = self.gain
                if self.acquisition_worker is not None:
                    self.acquisition_worker.gain = self.gain

        if reopen:
            # A new rate or channel count may not be one this input supports.
            self._negotiate_samp_rate()
            self.populate_data_sources()
            if self.display_worker is not None:
                # The display worker labels each row; a stale source list
                # mislabels every plot after a channel or rate change.
                self.display_worker.set_display_sources(self._display_sources())

        if not reopen or self.stream is None:
            return

        # A new rate, channel count or input needs a different PortAudio
        # stream, and the worker is bound to that stream's callback -- so both
        # are rebuilt rather than adjusted.
        was_streaming = self.status == DeviceStatus.STREAMING
        if was_streaming:
            self._stop_streaming()
        self._teardown_capture()
        try:
            self._open_stream()
        except Exception as e:
            log_print(self.logger, "error", f"Could not reopen audio input: {e}")
            self.status = DeviceStatus.DISCONNECTED
            return
        if was_streaming:
            self._start_streaming()

    def _apply_param_update_local(self, params: dict):
        """Mirror channel/label/rate changes on the parent side.

        ``get_data_sources()`` is answered out of the parent process, so a
        change made in the UI has to land here as well as in the child --
        otherwise the server keeps advertising the old source list and the
        plot-source selector never changes.
        """
        params = dict(params or {})
        touched = False

        hardware = params.pop("hardware", None)
        if isinstance(hardware, dict) and hardware:
            self.hardware = {k: dict(v) for k, v in hardware.items()}
            self.hw_key, self.hw_entry = resolve_hardware_entry(
                self.hardware, self.discovered_devices
            )
            touched = True

        for param, value in params.items():
            if param == "channels":
                self.hw_entry["channels"] = value
                self.group_config["channels"] = value
                self.channel_count = self._channel_count()
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
            self._negotiate_samp_rate()
            self.populate_data_sources()

    def _disconnect(self):
        self._stop_streaming()
        self._teardown_capture()
        self.status = DeviceStatus.DISCONNECTED
        return True
