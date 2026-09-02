"""
A common spec shared by all device specific Backend subclasses, in order to provide
a semblance of sanity for every future stage of the codebase. While each device
will have their own specific implementation, every backend is expected to provide a
shared set of functionality, listed below.

Queues:
- Display Data Queue:
- Response Queue:


Functions:
1. Device Control
- initialize()
- queue_param_update()
- start_streaming()
- stop_streaming()
- disconnect()

2. Parameter Handling
- get_param()
- set_param()


Signals:
- data_ready(DATA)


Properties:
- group_id: str
- status: DeviceStatus
- enable_save: bool
- save_path: str
"""
import logging
import multiprocessing as mp
import queue
from typing import Dict, Set

from bioview_common import (
    DATA_OUTPUT_QUEUE_DEPTH,
    DISPLAY_QUEUE_DEPTH,
    SAVE_QUEUE_DEPTH,
    DataSource,
    DeviceStatus,
    IPCCommand,
    Response,
    drain,
    log_print,
)

from bioview_server.common import DisplayWorker, SaveWorker


class Backend(mp.Process):
    def __init__(
        self,
        group_id: str,
        response_queue: mp.Queue,
        data_output_queue: mp.Queue = None,
    ):
        super().__init__()
        # Parameters
        self.group_id = group_id
        self.data_sources: Set[DataSource] = set()

        # Queues
        self.command_queue = mp.Queue()
        self.save_queue = None
        # Bounded: an unbounded display queue grows without limit whenever the
        # client or the socket writer falls behind.
        self.display_queue = mp.Queue(maxsize=DISPLAY_QUEUE_DEPTH)

        self.data_output_queue = data_output_queue
        self.response_queue = response_queue  # Queue for responses to client commands

        self.enable_save = False

        # Common workers
        self.save_worker = None
        self.display_worker = None

        # State
        self.status = DeviceStatus.DISCONNECTED
        self._running = mp.Event()
        self._streaming = mp.Event()

    # Internal Implementations
    # Common setup
    def _setup_saving(self, save_config: Dict = None):
        """
        Sets up workers to save data in a common format
        """
        self.enable_save = save_config.get("enable_save", False)
        self.save_path = save_config.get("save_path", None)

        if not self.save_queue:
            self.save_queue = mp.Queue(maxsize=SAVE_QUEUE_DEPTH)
        else:
            drain(self.save_queue)

        if self.enable_save and self.save_path:
            self.save_worker = SaveWorker(
                save_path=self.save_path,
                data_queue=self.save_queue,
                num_channels=len(self.get_data_sources()),
                logger=self.logger,
            )

        # Any other specific functionality can be implemented by subclasses

    def stop_saving(self):
        if self.save_worker:
            self.save_worker.stop()

        # Drain, not a single get: the old code removed exactly one item and
        # left the rest queued for the next session.
        drain(self.save_queue)

    def _setup_display(self, display_config: Dict = None):
        """
        Sets up workers to save data in a common format
        """
        if not self.data_output_queue:
            self.data_output_queue = mp.Queue(maxsize=DATA_OUTPUT_QUEUE_DEPTH)
        else:
            drain(self.data_output_queue)

        # The client receives the full stream (so it can save everything) and then
        # decides which sources to plot. Therefore we forward all data sources.
        self.display_worker = DisplayWorker(
            display_sources=list(self.data_sources),
            data_input_queue=self.display_queue,
            data_output_queue=self.data_output_queue,
            logger=self.logger,
        )

    def stop_display(self):
        if self.display_worker:
            self.display_worker.stop()

        drain(self.display_queue)

    # Device Control
    def _initialize(self):
        raise NotImplementedError

    def _queue_param_update(self):
        """
        Backends that implement this function will be able to handle
        real-time update of parameters by implementing multiprocessing
        queues internally
        """
        raise NotImplementedError

    def populate_data_sources(self):
        raise NotImplementedError

    def get_data_sources(self):
        """
        Broadcasts available data sources to server handler which can
        then choose to enable/disable on a per-device basis, as specified
        by the client handler. Since the handler only needs to know labels,
        we can deal accordingly.
        """
        return self.data_sources

    def _start_streaming(self):
        raise NotImplementedError

    def _stop_streaming(self):
        raise NotImplementedError

    def _disconnect(self):
        raise NotImplementedError

    # Parameter handling
    def get_param(self, param, default_value):
        try:
            value = getattr(self, param)
        except AttributeError:
            value = default_value
        return value

    def set_param(self, param, value):
        current_type = type(getattr(self, param, None))
        if current_type is not None:
            setattr(self, param, current_type(value))
        else:
            setattr(self, param, value)

    # Handle multiprocessing
    def run(self):
        # Create a new logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s: (%(levelname)s) %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )

        self._running.set()

        # Start thread to handle display
        while self._running.is_set():
            try:
                # Process commands from parent
                cmd_data = self.command_queue.get(timeout=1)
                self._handle_command(cmd_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in subprocess: {e}")

    def _handle_command(self, data):
        try:
            cmd = data["command"]
            cmd_args = data.get("args", {})

            match cmd:
                case IPCCommand.CONNECT_DEVICES:
                    result = self._initialize()
                    if not result:
                        raise RuntimeError("Unable to initialize device")
                    self.response_queue.put({"type": Response.SUCCESS, "result": result})

                case IPCCommand.START_STREAMING:
                    cmd_args = cmd_args or {}
                    save_cfg = cmd_args.get("save_config", {}) or {}
                    display_cfg = cmd_args.get("display_config", {}) or {}
                    if save_cfg.get("enable_save"):
                        self._setup_saving(save_cfg)
                    self._setup_display(display_cfg)
                    result = self._start_streaming()
                    self.response_queue.put({"type": Response.SUCCESS, "result": result})
                    self._streaming.set()
                    # Anything slow that must happen once streaming is live (an
                    # auto DPIC balance can run for a minute or more) goes here,
                    # *after* the response, so start_streaming() never blocks the
                    # caller past its timeout.
                    if result:
                        self._post_start_streaming()

                case IPCCommand.STOP_STREAMING:
                    self._streaming.clear()
                    result = self._stop_streaming()
                    self.response_queue.put({"type": Response.SUCCESS, "result": result})

                case IPCCommand.DISCONNECT_DEVICES:
                    result = self._disconnect()
                    self.response_queue.put({"type": Response.SUCCESS, "result": result})

                case IPCCommand.UPDATE_RUNNING_PARAMETER:
                    self._queue_param_update(cmd_args)
                    self.response_queue.put({"type": Response.SUCCESS, "result": None})

                case IPCCommand.RUN_DPIC_BALANCE:
                    result = self._run_dpic_balance()
                    self.response_queue.put(
                        {"type": Response.SUCCESS, "result": result is not None}
                    )

                case IPCCommand.SHUTDOWN:
                    self._streaming.clear()
                    self._running.clear()

        except Exception as e:
            log_print(self.logger, "error", f"Command {cmd} failed: {e}")
            self.response_queue.put({"type": Response.ERROR, "message": str(e)})

    def _run_dpic_balance(self):
        """Override in USRP backend for DPIC balancing."""
        return None

    def _post_start_streaming(self):
        """Hook for work that must run after START_STREAMING has been answered."""
        return None

    # Public API for non-blocking calls
    def initialize(self, **kwargs):
        self.command_queue.put({"command": IPCCommand.CONNECT_DEVICES, "args": kwargs})
        response = self.response_queue.get(timeout=150)
        return response

    def start_streaming(self, cfg_dict: Dict = None):
        self.command_queue.put({"command": IPCCommand.START_STREAMING, "args": cfg_dict})
        response = self.response_queue.get(timeout=10)
        return response

    def stop_streaming(self):
        self.command_queue.put({"command": IPCCommand.STOP_STREAMING})
        response = self.response_queue.get(timeout=10)
        return response

    def queue_param_update(self, **params):
        self.command_queue.put(
            {"command": IPCCommand.UPDATE_RUNNING_PARAMETER, "args": params}
        )
        # The command above is applied in the child process, which owns its own
        # copy of this object. get_data_sources() is answered by the *parent*, so
        # without mirroring the source-affecting parameters here the server would
        # keep advertising the channel list the device started with.
        try:
            self._apply_param_update_local(params)
        except Exception as e:
            # self.logger only exists inside the child; this runs in the parent.
            log_print(
                getattr(self, "logger", None),
                "warning",
                f"Local param mirror failed: {e}",
            )

    def _apply_param_update_local(self, params: Dict):
        """Parent-side mirror of the parameters that change ``data_sources``.

        Subclasses whose data sources depend on configuration override this;
        the default is a no-op because most parameters do not change the set of
        streams a device produces.
        """
        return None

    def run_dpic_balance(self, timeout: float = 1800):
        # A full balance is bounded by (number of pairs) x (search points) x
        # settle_time_s, which at conservative settings runs into minutes.
        self.command_queue.put(
            {
                "command": IPCCommand.RUN_DPIC_BALANCE,
                "args": {},
            }
        )
        response = self.response_queue.get(timeout=timeout)
        return response

    def disconnect(self):
        self.command_queue.put({"command": IPCCommand.DISCONNECT_DEVICES})
        response = self.response_queue.get(timeout=5)
        return response

    def shutdown(self):
        self.command_queue.put({"command": IPCCommand.SHUTDOWN})
        self.join(timeout=5)
        if self.is_alive():
            self.terminate()
