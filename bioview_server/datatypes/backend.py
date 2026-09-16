import contextlib
import logging
import multiprocessing as mp
import queue
import threading
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

from bioview_server.common import DisplayWorker, SaveForwarder


CONNECT_TIMEOUT = 150

START_STREAMING_TIMEOUT = 5
STOP_STREAMING_TIMEOUT = 15
DISCONNECT_TIMEOUT = 15
DEFAULT_TIMEOUT = 10

BALANCE_PROGRESS_QUEUE_DEPTH = 64


class _StepTimer:
    """Traces a multi-step bring-up so a hang names the step it hung on."""

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
    """Common contract shared by every device-specific backend."""

    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue = None,
        data_output_queue: mp.Queue = None,
        save_output_queue: mp.Queue = None,
    ):
        super().__init__()
        self.daemon = True

        self.group_id = group_id
        self.data_sources: set[DataSource] = set()
        self.logger = None

        self.command_queue = mp.Queue()
        self.save_queue = None
        self.display_queue = mp.Queue(maxsize=DISPLAY_QUEUE_DEPTH)

        self.progress_queue = mp.Queue(maxsize=BALANCE_PROGRESS_QUEUE_DEPTH)

        self.data_output_queue = data_output_queue
        self.save_output_queue = save_output_queue
        self.response_queue = (
            response_queue if response_queue is not None else mp.Queue()
        )

        self.enable_save = False

        self.save_worker = None
        self.display_worker = None

        self.status = DeviceStatus.DISCONNECTED
        self._running = mp.Event()
        self._streaming = mp.Event()
        self._request_id = 0
        self._init_local_state()

    def _init_local_state(self):
        """Locks, events and threads that must not cross the process boundary."""
        self._reply_lock = threading.Lock()
        self._pending_replies = {}
        self._balance_thread = None
        self._balance_abort = threading.Event()

    _LOCAL_STATE_KEYS = (
        "_reply_lock",
        "_pending_replies",
        "_balance_thread",
        "_balance_abort",
    )

    @classmethod
    def _local_state_keys(cls):
        """Every ``_LOCAL_STATE_KEYS`` entry declared along the MRO."""
        keys = []
        for klass in cls.__mro__:
            for key in klass.__dict__.get("_LOCAL_STATE_KEYS", ()):
                if key not in keys:
                    keys.append(key)
        return tuple(keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in type(self)._local_state_keys():
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_local_state()

    _StepTimer = _StepTimer

    def _setup_saving(self, save_config: dict = None):
        """Wire this device's save stream to the session recorder."""
        self.enable_save = save_config.get("enable_save", False)

        if not self.save_queue:
            self.save_queue = mp.Queue(maxsize=SAVE_QUEUE_DEPTH)
        else:
            drain(self.save_queue)

        if self.enable_save and self.save_output_queue is not None:
            if self.save_worker is not None:
                self.save_worker.stop()
            self.save_worker = SaveForwarder(
                device_id=self.group_id,
                data_input_queue=self.save_queue,
                data_output_queue=self.save_output_queue,
                logger=self.logger,
            )

    def record_param_change(self, param: str, value, t_wall: float = None):
        """Timestamp a setting this backend changed into the running recording."""
        if not self.enable_save or self.save_output_queue is None:
            return False
        with contextlib.suppress(Exception):
            self.save_output_queue.put_nowait(
                {
                    "type": "param_change",
                    "device_id": self.group_id,
                    "param": param,
                    "value": value,
                    "t_wall": float(t_wall if t_wall is not None else time.time()),
                }
            )
            return True
        return False

    def get_save_freq(self) -> float:
        """Rate (Hz) of this device's save stream, one sample per row per tick."""
        return float(getattr(self, "samp_rate", 0.0) or 0.0)

    def _save_rows(self):
        """Rows this device's save records carry, in order."""
        return self._display_sources()

    def describe_for_recording(self) -> dict:
        """This device's entry in the recording header's device table."""
        rows = list(self._save_rows())
        return {
            "device_id": self.group_id,
            "fs": self.get_save_freq(),
            "n_rows": len(rows),
            "dtype": "float32",
            "sources": [src.to_dict() for src in rows],
        }

    def stop_saving(self):
        if self.save_worker:
            self.save_worker.stop()

        drain(self.save_queue)

    def _display_sources(self):
        """Rows this device emits, in the order the display payload carries them."""
        return list(self.data_sources)

    def _setup_display(self, display_config: dict = None):
        if not self.data_output_queue:
            self.data_output_queue = mp.Queue(maxsize=DATA_OUTPUT_QUEUE_DEPTH)

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
        """Overridden by backends that support DPIC."""
        return {
            "ok": False,
            "message": f"{self.group_id} does not support DPIC balance.",
            "results": [],
        }

    def _post_start_streaming(self):
        """Hook for work that must run after START_STREAMING has been answered."""
        return None

    def publish_balance_progress(self, payload: dict):
        """Child side: offer the balancer's live state to the parent."""
        with contextlib.suppress(Exception):
            self.progress_queue.put_nowait(payload)

    def drain_balance_progress(self):
        """Parent side: the most recent progress entry, or None if there is none."""
        latest = None
        while True:
            try:
                latest = self.progress_queue.get_nowait()
            except queue.Empty:
                return latest
            except (OSError, ValueError):
                return latest

    def balance_aborted(self) -> bool:
        """True once a running balance has been asked to stop."""
        return self._balance_abort.is_set()

    def _abort_balance(self, join_timeout: float = 5.0):
        """Ask a running balance to stop and wait briefly for it to unwind."""
        thread = self._balance_thread
        if thread is None or not thread.is_alive():
            return
        log_print(
            self.logger,
            "info",
            f"[DPIC] Aborting the running balance on {self.group_id}",
        )
        self._balance_abort.set()
        thread.join(timeout=join_timeout)

    def _balance_in_progress(self) -> bool:
        return self._balance_thread is not None and self._balance_thread.is_alive()

    def _start_balance_thread(self, request_id=None):
        """Run one balance off the command loop and reply when it finishes."""

        if self._balance_in_progress():
            return False

        def _worker():
            try:
                outcome = self._run_dpic_balance()
                ok = bool(outcome.get("ok"))
                payload = {
                    "type": Response.SUCCESS if ok else Response.ERROR,
                    "message": outcome.get("message", ""),
                    "result": outcome.get("results", []),
                }
            except Exception as e:
                log_print(self.logger, "error", f"DPIC balance failed: {e}")
                payload = {
                    "type": Response.ERROR,
                    "message": str(e) or f"{type(e).__name__} in {self.group_id}",
                    "result": [],
                }
            if request_id is not None:
                self._reply(request_id, payload)

        self._balance_abort.clear()
        self._balance_thread = threading.Thread(
            target=_worker, name=f"dpic-balance-{self.group_id}", daemon=True
        )
        self._balance_thread.start()
        return True

    def _apply_param_update_local(self, params: dict):
        """Parent-side mirror of the parameters that change ``data_sources``."""
        return None

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
                    if result:
                        self._post_start_streaming()

                case IPCCommand.STOP_STREAMING:
                    self._streaming.clear()
                    self._abort_balance()
                    result = self._stop_streaming()
                    self._reply(request_id, {"type": Response.SUCCESS, "result": result})

                case IPCCommand.DISCONNECT_DEVICES:
                    self._abort_balance()
                    result = self._disconnect()
                    self._reply(request_id, {"type": Response.SUCCESS, "result": result})

                case IPCCommand.UPDATE_RUNNING_PARAMETER:
                    self._queue_param_update(cmd_args)
                    self._reply(request_id, {"type": Response.SUCCESS, "result": None})

                case IPCCommand.RUN_DPIC_BALANCE:
                    if self._balance_in_progress():
                        self._reply(
                            request_id,
                            {
                                "type": Response.ERROR,
                                "message": (
                                    f"A DPIC balance is already running on "
                                    f"{self.group_id}"
                                ),
                                "result": [],
                            },
                        )
                    else:
                        self._start_balance_thread(request_id)

                case IPCCommand.SHUTDOWN:
                    self._streaming.clear()
                    self._abort_balance()
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

    @staticmethod
    def _command_name(command) -> str:
        return getattr(command, "name", str(command))

    def _request(self, command, args: dict = None, timeout: float = DEFAULT_TIMEOUT):
        """Send a command to the child and wait for *its* reply."""
        with self._reply_lock:
            self._request_id += 1
            request_id = self._request_id
        self.command_queue.put(
            {"command": command, "args": args or {}, "request_id": request_id}
        )

        deadline = time.monotonic() + timeout
        while True:
            with self._reply_lock:
                parked = self._pending_replies.pop(request_id, None)
            if parked is not None:
                return parked

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._reply_lock:
                    self._pending_replies.pop(request_id, None)
                raise DeviceError(
                    f"{self.group_id} did not answer "
                    f"{self._command_name(command)} within {timeout:.0f}s"
                )
            try:
                response = self.response_queue.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if self.pid is not None and not self.is_alive():
                    raise DeviceError(
                        f"{self.group_id} backend process exited while handling "
                        f"{self._command_name(command)} "
                        f"(exit code {self.exitcode})"
                    ) from None
                continue

            if not isinstance(response, dict):
                continue
            reply_id = response.get("request_id")
            if reply_id in (None, request_id):
                return response
            with self._reply_lock:
                if len(self._pending_replies) >= 32:
                    self._pending_replies.pop(next(iter(self._pending_replies)))
                self._pending_replies[reply_id] = response

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
        with self._reply_lock:
            self._request_id += 1
            request_id = self._request_id
        self.command_queue.put(
            {
                "command": IPCCommand.UPDATE_RUNNING_PARAMETER,
                "args": params,
                "request_id": request_id,
            }
        )
        try:
            self._apply_param_update_local(params)
        except Exception as e:
            log_print(self.logger, "warning", f"Local param mirror failed: {e}")

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
