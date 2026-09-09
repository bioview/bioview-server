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

# One entry per measurement, drained about once a second by the client's poll.
# Deep enough to survive a slow poll, shallow enough that a client which stops
# polling costs nothing.
BALANCE_PROGRESS_QUEUE_DEPTH = 64


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
        save_output_queue: mp.Queue = None,
    ):
        super().__init__()
        # Parameters
        self.group_id = group_id
        self.data_sources: set[DataSource] = set()
        # Replaced in run(), inside the child. Defined here so any code touching
        # it from the parent gets a no-op logger rather than an AttributeError.
        self.logger = None

        # Queues
        self.command_queue = mp.Queue()
        self.save_queue = None
        # Bounded: an unbounded display queue grows without limit whenever the
        # client or the socket writer falls behind.
        self.display_queue = mp.Queue(maxsize=DISPLAY_QUEUE_DEPTH)

        # Balance progress, child -> parent. Small dicts, one per measurement;
        # bounded because a UI that stops draining must never grow it without
        # limit, and only the newest entry is worth anything anyway.
        self.progress_queue = mp.Queue(maxsize=BALANCE_PROGRESS_QUEUE_DEPTH)

        self.data_output_queue = data_output_queue
        # Shared with every other backend and drained by the server's single
        # BvrWriter; this device only ever pushes its own tagged records.
        self.save_output_queue = save_output_queue
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
        self._init_local_state()

    def _init_local_state(self):
        """Locks, events and threads that must not cross the process boundary.

        Parent-side reply routing: a balance is answered on its own thread
        while Stop is answered on the command thread, so two callers can be
        inside ``_request()`` on one response queue; whichever reads a reply
        that is not its own parks it in ``_pending_replies`` for the thread
        waiting on it. Dropping it -- what the old ``continue`` did -- lost the
        other caller's answer and timed it out.

        Child-side balance state: the search runs on its own thread so
        STOP_STREAMING and SHUTDOWN are still answered while it is in flight.

        Both sides are rebuilt after unpickling (see ``__setstate__``): spawn
        pickles this object into the child, and a lock, an event or a live
        thread cannot survive that.
        """
        self._reply_lock = threading.Lock()
        self._pending_replies = {}
        self._balance_thread = None
        self._balance_abort = threading.Event()

    #: Rebuilt in the child rather than shipped to it; see _init_local_state.
    _LOCAL_STATE_KEYS = (
        "_reply_lock",
        "_pending_replies",
        "_balance_thread",
        "_balance_abort",
    )

    @classmethod
    def _local_state_keys(cls):
        """Every ``_LOCAL_STATE_KEYS`` entry declared along the MRO.

        Subclasses add their own locks in ``_init_local_state``; collecting the
        keys here means a subclass declaring its own tuple extends the base
        list rather than shadowing it.
        """
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

    #: Step tracer for bring-up paths; see _StepTimer.
    _StepTimer = _StepTimer

    # Common setup
    def _setup_saving(self, save_config: dict = None):
        """Wire this device's save stream to the session recorder.

        The file itself is written by the server's single ``BvrWriter``: every
        backend is its own process, so each forwards tagged records onto one
        shared queue rather than opening a file of its own.
        """
        self.enable_save = save_config.get("enable_save", False)

        if not self.save_queue:
            self.save_queue = mp.Queue(maxsize=SAVE_QUEUE_DEPTH)
        else:
            drain(self.save_queue)

        if self.enable_save and self.save_output_queue is not None:
            # Stop the previous forwarder before replacing it, or its thread
            # stays alive on the same input queue.
            if self.save_worker is not None:
                self.save_worker.stop()
            self.save_worker = SaveForwarder(
                device_id=self.group_id,
                data_input_queue=self.save_queue,
                data_output_queue=self.save_output_queue,
                logger=self.logger,
            )

    def get_save_freq(self) -> float:
        """Rate (Hz) of this device's save stream, one sample per row per tick.

        Distinct from ``DataSource.disp_freq``, which describes the decimated
        display stream. Backends whose save path is decimated relative to
        acquisition override this.
        """
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
        """Overridden by backends that support DPIC.

        Returns ``{"ok": bool, "message": str, "results": list}``.
        """
        return {
            "ok": False,
            "message": f"{self.group_id} does not support DPIC balance.",
            "results": [],
        }

    def _post_start_streaming(self):
        """Hook for work that must run after START_STREAMING has been answered."""
        return None

    def publish_balance_progress(self, payload: dict):
        """Child side: offer the balancer's live state to the parent.

        Dropped rather than blocked on when the queue is full: a stalled UI must
        never slow the search down, and the next measurement supersedes this one
        anyway.
        """
        with contextlib.suppress(Exception):
            self.progress_queue.put_nowait(payload)

    def drain_balance_progress(self):
        """Parent side: the most recent progress entry, or None if there is none.

        Everything queued is drained, not just one entry, so a poll that runs
        after a burst of measurements reports where the search *is* rather than
        walking through where it has been.
        """
        latest = None
        while True:
            try:
                latest = self.progress_queue.get_nowait()
            except queue.Empty:
                return latest
            except (OSError, ValueError):
                return latest

    def balance_aborted(self) -> bool:
        """True once a running balance has been asked to stop.

        Handed to ``DpicBalancer.should_abort`` by the backends that support
        DPIC, so an abort takes effect at the next sweep point instead of after
        the whole time budget.
        """
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
        """Run one balance off the command loop and reply when it finishes.

        ``request_id`` is None for the auto-balance that follows START_STREAMING:
        nobody is waiting on a reply for it, and putting one on the queue with a
        null id would let an unrelated caller claim it.
        """

        # Safety net for the caller that does not check first (the
        # auto-balance): two searches driving one radio would fight.
        if self._balance_in_progress():
            return False

        def _worker():
            try:
                # Always a dict: a balance that bails out early carries the
                # reason, so it can never reach the client as a success.
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
                    # A balance measures a live stream; stopping the radio out
                    # from under it would leave it waiting on metrics that can
                    # never arrive until its read timeout expires, once per
                    # sweep point.
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
                    # Answered from the balance thread, not from here: the
                    # search takes a minute or more and the command loop has to
                    # keep serving Stop and Shutdown throughout.
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

    # Parent-side API
    @staticmethod
    def _command_name(command) -> str:
        return getattr(command, "name", str(command))

    def _request(self, command, args: dict = None, timeout: float = DEFAULT_TIMEOUT):
        """Send a command to the child and wait for *its* reply.

        Replies are matched by request id, so a late answer to a request that
        already timed out is discarded rather than handed to the next caller.
        """
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
            reply_id = response.get("request_id")
            if reply_id in (None, request_id):
                return response
            with self._reply_lock:
                # Bounded: a reply to a request that already timed out is never
                # claimed, and this map must not grow for the life of the run.
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
        # Fire and forget: applying this can restart the device stream, and the
        # server must not block its command thread on that.
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
        # get_data_sources() is answered by the parent, so source-affecting
        # parameters have to be mirrored here as well as in the child.
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
