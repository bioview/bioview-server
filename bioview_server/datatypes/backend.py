import logging
import multiprocessing as mp
import queue
import time

from bioview_common import (
    DATA_OUTPUT_QUEUE_DEPTH,
    DISPLAY_QUEUE_DEPTH,
    SAVE_QUEUE_DEPTH,
    DataSource,
    DeviceError,
    DeviceStatus,
    IPCCommand,
    Response,
    drain,
    log_print,
)

from bioview_server.common import DisplayWorker, SaveWorker


# Opening a radio is the slow step: USB enumeration, FPGA/CODEC bring-up and
# clock locking, plus an OS process spawn per backend on Windows.
CONNECT_TIMEOUT = 150

# Starting a stream, by contrast, is near-instant once the device is open: every
# worker thread already exists and is merely resumed, so the whole path is a few
# Event.set() calls plus a sub-second buffer-filling delay. Measured bring-up is
# ~0.4 s for a two-channel USRP and ~0.02 s for BIOPAC. A device that has not
# answered in several seconds is wedged, not slow, and waiting longer only
# delays the error -- and, because devices are started in sequence, holds up
# every other device in the session behind it.
START_STREAMING_TIMEOUT = 5
# Stopping has real work to do: the transmit worker sends a final end-of-burst
# buffer and the recorder flushes and closes its HDF5 file.
STOP_STREAMING_TIMEOUT = 15
DISCONNECT_TIMEOUT = 15
DEFAULT_TIMEOUT = 10


class _StepTimer:
    """Traces a multi-step bring-up so a hang names the step it hung on.

    Each step is logged when it *finishes*, so the last line in the log is the
    last thing that completed and the hang is in whatever comes next. Elapsed
    time is per-step; a step that is merely slow is then distinguishable from
    one that never returned.
    """

    def __init__(self, logger, what: str):
        self.logger = logger
        self.what = what
        self.started = time.monotonic()
        self.last = self.started
        log_print(self.logger, "debug", f"[{what}] begin")

    def mark(self, step: str):
        now = time.monotonic()
        log_print(
            self.logger,
            "debug",
            f"[{self.what}] {step} ok ({(now - self.last) * 1000:.0f} ms)",
        )
        self.last = now

    def done(self):
        elapsed_ms = (time.monotonic() - self.started) * 1000
        log_print(
            self.logger,
            "debug",
            f"[{self.what}] complete ({elapsed_ms:.0f} ms)",
        )


class Backend(mp.Process):
    """Common contract shared by every device-specific backend.

    Each backend runs as its own process, driven over ``command_queue``; replies
    come back on ``response_queue``, correlated by request id.
    """

    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue = None,
        data_output_queue: mp.Queue = None,
    ):
        super().__init__()
        # Parameters
        self.group_id = group_id
        self.data_sources: set[DataSource] = set()

        # Queues
        self.command_queue = mp.Queue()
        self.save_queue = None
        # Bounded: an unbounded display queue grows without limit whenever the
        # client or the socket writer falls behind.
        self.display_queue = mp.Queue(maxsize=DISPLAY_QUEUE_DEPTH)

        self.data_output_queue = data_output_queue
        # Never shared between backends: a reply carries no sender, so two
        # devices reading one queue steal each other's answers.
        self.response_queue = (
            response_queue if response_queue is not None else mp.Queue()
        )

        self.enable_save = False

        # Common workers
        self.save_worker = None
        self.display_worker = None

        # State
        self.status = DeviceStatus.DISCONNECTED
        self._running = mp.Event()
        self._streaming = mp.Event()
        self._request_id = 0

    #: Step tracer for bring-up paths; see _StepTimer.
    _StepTimer = _StepTimer

    # Common setup
    def _setup_saving(self, save_config: dict = None):
        self.enable_save = save_config.get("enable_save", False)
        self.save_path = save_config.get("save_path", None)

        if not self.save_queue:
            self.save_queue = mp.Queue(maxsize=SAVE_QUEUE_DEPTH)
        else:
            drain(self.save_queue)

        if self.enable_save and self.save_path:
            # Stop the previous recorder before replacing it, or its thread stays
            # alive holding an unflushed HDF5 file open.
            if self.save_worker is not None:
                self.save_worker.stop()
            self.save_worker = SaveWorker(
                save_path=self.save_path,
                data_queue=self.save_queue,
                num_channels=len(self.get_data_sources()),
                logger=self.logger,
            )

    def stop_saving(self):
        if self.save_worker:
            self.save_worker.stop()

        drain(self.save_queue)

    def _display_sources(self):
        """Rows this device emits, in the order the display payload carries them.

        Overridden where the emitted rows are not simply ``data_sources`` -- the
        USRP appends calibration-reference rows.
        """
        return list(self.data_sources)

    def _setup_display(self, display_config: dict = None):
        if not self.data_output_queue:
            self.data_output_queue = mp.Queue(maxsize=DATA_OUTPUT_QUEUE_DEPTH)
        # The output queue is deliberately *not* drained here. The server hands
        # the same queue to every backend, so draining it on one device's Start
        # discards the chunks another device has already queued -- data loss
        # that only shows up once two devices stream together. It is bounded and
        # evicted oldest-first anyway, and the server's data handler drains it
        # continuously, so nothing stale can accumulate.

        # The client receives the full stream and decides what to plot, so all
        # sources are forwarded. The worker is reused across Start/Stop cycles:
        # replacing it leaks the old thread, which stays alive on the same
        # input queue.
        if self.display_worker is not None:
            self.display_worker.set_display_sources(self._display_sources())
            return

        self.display_worker = DisplayWorker(
            display_sources=self._display_sources(),
            data_input_queue=self.display_queue,
            data_output_queue=self.data_output_queue,
            logger=self.logger,
        )

    def stop_display(self):
        if self.display_worker:
            self.display_worker.stop()

        drain(self.display_queue)

    # Device control, implemented per device
    def _initialize(self):
        raise NotImplementedError

    def _queue_param_update(self):
        raise NotImplementedError

    def populate_data_sources(self):
        raise NotImplementedError

    def get_data_sources(self):
        return self.data_sources

    def _start_streaming(self):
        raise NotImplementedError

    def _stop_streaming(self):
        raise NotImplementedError

    def _disconnect(self):
        raise NotImplementedError

    def _run_dpic_balance(self):
        """Overridden by the USRP backend."""
        return None

    def _post_start_streaming(self):
        """Hook for work that must run after START_STREAMING has been answered."""
        return None

    def _apply_param_update_local(self, params: dict):
        """Parent-side mirror of the parameters that change ``data_sources``."""
        return None

    # Child process
    def run(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s: (%(levelname)s) %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )

        self._running.set()

        while self._running.is_set():
            try:
                cmd_data = self.command_queue.get(timeout=1)
                self._handle_command(cmd_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in subprocess: {e}")

    def _reply(self, request_id, payload: dict):
        self.response_queue.put({**payload, "request_id": request_id})

    def _handle_command(self, data):
        cmd = data.get("command")
        request_id = data.get("request_id")

        try:
            cmd_args = data.get("args", {})

            match cmd:
                case IPCCommand.CONNECT_DEVICES:
                    result = self._initialize()
                    if not result:
                        raise RuntimeError("Unable to initialize device")
                    self._reply(request_id, {"type": Response.SUCCESS, "result": result})

                case IPCCommand.START_STREAMING:
                    cmd_args = cmd_args or {}
                    save_cfg = cmd_args.get("save_config", {}) or {}
                    display_cfg = cmd_args.get("display_config", {}) or {}
                    step = _StepTimer(self.logger, f"{self.group_id} START_STREAMING")
                    if save_cfg.get("enable_save"):
                        self._setup_saving(save_cfg)
                        step.mark("saving set up")
                    self._setup_display(display_cfg)
                    step.mark("display set up")
                    result = self._start_streaming()
                    step.mark("device started")
                    self._reply(request_id, {"type": Response.SUCCESS, "result": result})
                    step.done()
                    self._streaming.set()
                    # Slow post-start work (an auto DPIC balance runs for a
                    # minute or more) goes after the reply, never before it.
                    if result:
                        self._post_start_streaming()

                case IPCCommand.STOP_STREAMING:
                    self._streaming.clear()
                    result = self._stop_streaming()
                    self._reply(request_id, {"type": Response.SUCCESS, "result": result})

                case IPCCommand.DISCONNECT_DEVICES:
                    result = self._disconnect()
                    self._reply(request_id, {"type": Response.SUCCESS, "result": result})

                case IPCCommand.UPDATE_RUNNING_PARAMETER:
                    self._queue_param_update(cmd_args)
                    self._reply(request_id, {"type": Response.SUCCESS, "result": None})

                case IPCCommand.RUN_DPIC_BALANCE:
                    result = self._run_dpic_balance()
                    self._reply(
                        request_id,
                        {"type": Response.SUCCESS, "result": result is not None},
                    )

                case IPCCommand.SHUTDOWN:
                    self._streaming.clear()
                    self._running.clear()

        except Exception as e:
            log_print(self.logger, "error", f"Command {cmd} failed: {e}")
            self._reply(
                request_id,
                {
                    "type": Response.ERROR,
                    "message": str(e) or f"{type(e).__name__} in {self.group_id}",
                },
            )

    # Parent-side API
    @staticmethod
    def _command_name(command) -> str:
        return getattr(command, "name", str(command))

    def _request(self, command, args: dict = None, timeout: float = DEFAULT_TIMEOUT):
        """Send a command to the child and wait for *its* reply.

        Replies are matched by request id, so a late answer to a request that
        already timed out is discarded rather than handed to the next caller.
        """
        self._request_id += 1
        request_id = self._request_id
        self.command_queue.put(
            {"command": command, "args": args or {}, "request_id": request_id}
        )

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeviceError(
                    f"{self.group_id} did not answer "
                    f"{self._command_name(command)} within {timeout:.0f}s"
                )
            try:
                response = self.response_queue.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                # A child that died -- a native crash inside a driver leaves no
                # Python traceback -- would otherwise be indistinguishable from
                # a slow one until the full timeout expired, and would then be
                # reported as "did not answer", which points at the wrong thing.
                # Checked only after an empty read, so a reply already queued by
                # a child that exited straight afterwards is still delivered.
                if self.pid is not None and not self.is_alive():
                    raise DeviceError(
                        f"{self.group_id} backend process exited while handling "
                        f"{self._command_name(command)} "
                        f"(exit code {self.exitcode})"
                    ) from None
                continue

            if not isinstance(response, dict):
                continue
            if response.get("request_id") not in (None, request_id):
                continue
            return response

    def _request_or_raise(self, command, args=None, timeout=DEFAULT_TIMEOUT):
        response = self._request(command, args, timeout)
        if response.get("type") in (Response.ERROR, Response.ERROR.name):
            message = response.get("message") or "unknown backend error"
            raise DeviceError(f"{self.group_id}: {message}")
        return response

    def initialize(self, **kwargs):
        return self._request(IPCCommand.CONNECT_DEVICES, kwargs, timeout=CONNECT_TIMEOUT)

    def start_streaming(self, cfg_dict: dict = None):
        return self._request_or_raise(
            IPCCommand.START_STREAMING, cfg_dict, timeout=START_STREAMING_TIMEOUT
        )

    def stop_streaming(self):
        return self._request_or_raise(
            IPCCommand.STOP_STREAMING, timeout=STOP_STREAMING_TIMEOUT
        )

    def queue_param_update(self, **params):
        # Fire and forget: applying this can restart the device stream, and the
        # server must not block its command thread on that.
        self._request_id += 1
        self.command_queue.put(
            {
                "command": IPCCommand.UPDATE_RUNNING_PARAMETER,
                "args": params,
                "request_id": self._request_id,
            }
        )
        # get_data_sources() is answered by the parent, so source-affecting
        # parameters have to be mirrored here as well as in the child.
        try:
            self._apply_param_update_local(params)
        except Exception as e:
            log_print(
                getattr(self, "logger", None),
                "warning",
                f"Local param mirror failed: {e}",
            )

    def run_dpic_balance(self, timeout: float = 1800):
        return self._request(IPCCommand.RUN_DPIC_BALANCE, timeout=timeout)

    def disconnect(self):
        return self._request_or_raise(
            IPCCommand.DISCONNECT_DEVICES, timeout=DISCONNECT_TIMEOUT
        )

    def shutdown(self):
        self.command_queue.put({"command": IPCCommand.SHUTDOWN})
        self.join(timeout=5)
        if self.is_alive():
            self.terminate()
