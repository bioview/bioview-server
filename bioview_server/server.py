""" BioView Server

The server exposes a flexible way for connecting to devices by forwarding
client commands to the appropriate handlers.

Several clients may be connected at once -- typically a Monitor and a
Configurator window sharing the one server on the machine. Every client gets
the same level of access: there is a single set of attached hardware, so the
device configuration, device status and streaming state are server-wide and
shared, and a device operation started by one client is visible to the others
through GET_DEVICE_STATUS. Each client has its own control connection and
command thread, and every command is answered on the connection it arrived on.
Acquired data is fanned out to every connected client.
"""

import argparse
import contextlib
import logging
import multiprocessing as mp
import queue
import signal
import socket
import time
from threading import Lock, Thread, local

from bioview_common import (
    APP_VERSION,
    CONTROL_PORT,
    DATA_OUTPUT_QUEUE_DEPTH,
    DATA_PORT,
    Command,
    DeviceError,
    DeviceStatus,
    DeviceType,
    Response,
    ValidationError,
    generate_challenge,
    get_app_info,
    get_local_addresses,
    is_local_request,
    log_print,
    parse_and_validate_command,
    recv_message,
    send_datachunk,
    send_response,
    set_exclusive_bind,
    validate_token,
)

from bioview_server.device import (
    AVAILABLE_BACKENDS,
    UNAVAILABLE_BACKENDS,
    get_device_handler,
)


SLEEP_DURATION = 0.001  # Confirm CPU load with varying this value


def _handler_init_succeeded(resp: dict) -> bool:
    """True only when a backend subprocess reports a successful connect."""
    if not resp or not isinstance(resp, dict):
        return False
    resp_type = resp.get("type")
    if resp_type in (Response.ERROR, Response.ERROR.name):
        return False
    if resp_type not in (Response.SUCCESS, Response.SUCCESS.name):
        return False
    if resp.get("result") is False:
        return False
    return True


class ClientSession:
    """One connected client: its control connection, data connection and info.

    A session owns the command thread that serves it. Commands are answered on
    ``control_conn``; acquired data is pushed to ``data_conn`` by the server's
    single data thread, which fans each chunk out to every live session.
    """

    def __init__(self, control_conn, data_conn, info=None):
        self.control_conn = control_conn
        self.data_conn = data_conn
        self.info = info or {}
        self.active = True
        self.thread = None
        # Serializes writes to control_conn: replies come from this session's
        # command thread, but a device op started elsewhere can also report in.
        self.send_lock = Lock()

    @property
    def name(self):
        return self.info.get("hostname") or self.info.get("name") or "client"

    def close(self):
        self.active = False
        for conn in (self.control_conn, self.data_conn):
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()


class Server:
    def __init__(
        self,
        local_only: bool,
        control_port: int,
        data_port: int,
        logger=None,
        exit_when_idle: float = 0,
    ):
        # Keep track of local PC information for client communication
        self.info = get_app_info()
        self.token = 42  # TODO: Load using secrets

        # Network info
        self.control_port = control_port
        self.data_port = data_port

        # Server status
        self.running = False

        # A server started by a GUI window outlives that window when another
        # window is still using it, so it retires itself once it has sat with no
        # clients for this long. Zero (the default, and what a manually started
        # server gets) means never.
        self.exit_when_idle = exit_when_idle
        self._idle_since = time.monotonic()

        # Connected client sessions (see ClientSession). Guarded by its lock
        # because the accept loop adds to it while command threads remove
        # themselves and the data thread iterates over it.
        self.sessions = []
        self._sessions_lock = Lock()

        # The session whose command a thread is currently handling. Replies go
        # to the client that asked, so each command thread records itself here
        # and self.client_control_conn resolves against it.
        self._thread_session = local()

        # Client handling
        self.local_only = local_only
        self.discovered_clients = {}
        self.connected_client_info = {}

        # Device handling
        self.device_group_states = {}
        self.device_group_handlers = {}
        #: Why a group is not usable, keyed by device/group id. The reason used
        #: to be logged on the server only -- and a server spawned by the GUI has
        #: its output detached -- so a failed device gave the user nothing but
        #: "initialization failed" with no cause anywhere.
        self.device_group_errors = {}
        self.config = None
        self.data_sources = set()  # set(source: DataSource)
        self.discovered_devices_cache = {}
        self._device_op_lock = Lock()
        self._device_op_in_progress = False
        self._device_op_thread = None

        # Sockets
        self.data_socket = None
        self.control_socket = None

        # Server-wide data fan-out worker
        self.data_thread = None

        # Queue for overall logging
        self.response_queue = mp.Queue()

        # Queue for data output
        self.data_queue = mp.Queue(maxsize=DATA_OUTPUT_QUEUE_DEPTH)

        # Message logging
        if not logger:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s %(name)s: (%(levelname)s) %(message)s",
                datefmt="%m/%d %H:%M:%S",
            )
        else:
            self.logger = logger

    def start(self):
        log_print(self.logger, "info", "Starting server")

        # Setup sockets
        self._create_sockets()

        # Mark the server status as running
        self.running = True

        # One data thread for the whole server: it drains the acquisition queue
        # and fans each chunk out to whichever clients are connected at the time.
        self.data_thread = Thread(target=self._data_handler, daemon=True)
        self.data_thread.start()

        while self.running:
            control_conn = None
            self._check_idle_exit()
            try:
                # Wait for client connection.
                try:
                    self.control_socket.settimeout(1.0)
                    control_conn, addr = self.control_socket.accept()
                    log_print(
                        self.logger, "debug", f"Control connection initiated from {addr}"
                    )
                except socket.timeout:
                    # Timeout just means no one connected yet. Loop back and
                    # check self.running.
                    continue
                except OSError:
                    # Socket closed or error
                    break

                control_conn.settimeout(5.0)

                # If remote clients are not allowed, close the connection
                if self.local_only and not self._is_local_client(addr):
                    control_conn.close()
                    continue

                # Now that we have a connection, we will validate the payload
                auth_data = recv_message(control_conn, self.logger)
                if not auth_data:
                    control_conn.close()
                    continue

                cmd_type, payload = parse_and_validate_command(auth_data)
                if cmd_type == Command.DISCOVER_SERVERS.name:
                    send_response(
                        sock=control_conn,
                        response=Response.SUCCESS,
                        # The live client count lets a closing window tell
                        # whether another one still needs this server.
                        params={**self.info, "clients": len(self.sessions)},
                        logger=self.logger,
                    )
                    control_conn.close()
                    continue
                elif cmd_type == Command.CONNECT_SERVER.name:
                    hostname = payload.get("client_info", {}).get("hostname", None)
                    if hostname:
                        log_print(
                            self.logger, "info", f"Incoming connection from: {hostname}"
                        )

                    # Send challenge
                    challenge = generate_challenge()
                    (
                        send_response(
                            sock=control_conn,
                            response=Response.SERVER_CHALLENGE,
                            params={"challenge": challenge, "timestamp": time.time()},
                            logger=self.logger,
                        ),
                    )

                    challenge_response = recv_message(control_conn, self.logger)
                    client_cmd, client_payload = parse_and_validate_command(
                        challenge_response
                    )

                    if client_cmd != Command.AUTHENTICATE_CLIENT.name:
                        # This is an invalid connection attempt.
                        # TODO: log it.
                        control_conn.close()
                        continue

                    auth_token = client_payload.get("token", None)
                    if auth_token and validate_token(challenge, auth_token):
                        send_response(
                            sock=control_conn,
                            response=Response.AUTHENTICATION_SUCCESS,
                            params={"server_info": self.info, "timestamp": time.time()},
                            logger=self.logger,
                        )
                    else:
                        control_conn.close()
                        continue

                    # This client's details, kept with its own session. Several
                    # clients may be connecting at once, so it is a local rather
                    # than shared server state.
                    client_info = payload.get("client_info") or {}
                    session_info = {
                        "ip": client_info.get("ip", ""),
                        "hostname": client_info.get("hostname", ""),
                        "name": client_info.get("name", ""),
                        "version": client_info.get("version", ""),
                    }
                    self.connected_client_info = session_info
                else:
                    # Invalid command, just keep searching
                    control_conn.close()
                    continue

                """
                Since we are here only when the client has been successfully
                authenticated, it makes sense to initiate the data connection.
                It is also useful to do it
                at this stage since if the data connection cannot be made, the program is
                pretty much useless and we should just try to restart the client
                """
                try:
                    data_conn, _ = self.data_socket.accept()
                    log_print(self.logger, "debug", "Data connection accepted.")
                except socket.timeout:
                    log_print(
                        self.logger,
                        "error",
                        "Client failed to connect data socket in time.",
                    )
                    control_conn.close()
                    continue

                # Client is fully connected. It is served on its own thread so
                # this loop stays free: a second window (Monitor alongside
                # Configurator) must be able to connect while the first is
                # still using the server.
                self.handle_client_session(control_conn, data_conn, session_info)

            except Exception as e:
                # One client's failure must not disconnect the others, so only
                # this connection attempt is abandoned.
                log_print(self.logger, "error", f"Error in main loop: {e}")
                if control_conn is not None:
                    with contextlib.suppress(Exception):
                        control_conn.close()

    def _create_sockets(self):
        """
        Since the client can shutdown at any time or have an error, we want to ensure
        that the server only binds to sockets at launch and closes them when the server
        shuts down.
        """
        # Create control socket
        try:
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Exclusive: a second BioView server must fail to bind rather than
            # start alongside this one and quietly split the incoming clients.
            set_exclusive_bind(self.control_socket)
            self.control_socket.bind(("0.0.0.0", self.control_port))
            # A LAN discovery scan opens many short-lived probe connections at
            # once; a generous backlog keeps them from being refused/reset.
            self.control_socket.listen(socket.SOMAXCONN)
            self.control_socket.settimeout(1)  # Make sure that accept is non-blocking
            log_print(self.logger, "debug", "Control socket created")
        except OSError as e:
            # A port already in use almost always means another BioView server is
            # already running; surface it so the caller can exit cleanly rather
            # than spin on an unbound socket.
            log_print(self.logger, "error", f"Unable to create control socket: {e}")
            raise

        # Create data socket
        try:
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            set_exclusive_bind(self.data_socket)
            self.data_socket.bind(("0.0.0.0", self.data_port))
            self.data_socket.listen(8)
            self.data_socket.settimeout(5)
            log_print(self.logger, "debug", "Data socket connected")
        except OSError as e:
            log_print(self.logger, "error", f"Unable to create data socket: {e}")
            raise

    @property
    def client_control_conn(self):
        """The control connection of the client this thread is serving.

        Every reply is addressed to the client that asked, and each command is
        handled on its own session's thread -- long device operations are
        acknowledged synchronously and then reported through GET_DEVICE_STATUS
        polling, so no handler answers from a foreign thread. Resolving the
        connection per thread therefore routes each reply correctly without
        passing a session argument through every handler.
        """
        session = getattr(self._thread_session, "session", None)
        return session.control_conn if session is not None else None

    @property
    def client_session_active(self):
        """True while at least one client is connected."""
        return bool(self._live_sessions())

    def _check_idle_exit(self):
        """Shut down once no client has been connected for exit_when_idle seconds.

        This is what lets the Monitor and the Configurator share one server: the
        window that started it does not have to be the last one standing, and no
        server is left behind after every window has closed.
        """
        if not self.exit_when_idle or self._idle_since is None:
            return
        if self.sessions:
            return
        if time.monotonic() - self._idle_since < self.exit_when_idle:
            return

        log_print(
            self.logger,
            "info",
            f"No clients for {self.exit_when_idle:g}s. Shutting down server...",
        )
        self.running = False

    def _live_sessions(self):
        """A snapshot of the connected sessions, safe to iterate outside the lock."""
        with self._sessions_lock:
            return [session for session in self.sessions if session.active]

    def handle_client_session(self, control_conn, data_conn, info=None):
        """Register a newly authenticated client and start serving it.

        Returns immediately: the client is served on its own thread so the
        accept loop stays free for the next window that wants to connect.
        """
        session = ClientSession(control_conn, data_conn, info)

        with self._sessions_lock:
            self.sessions.append(session)
            client_count = len(self.sessions)
            self._idle_since = None

        session.thread = Thread(
            target=self._serve_client,
            args=(session,),
            daemon=True,
        )
        session.thread.start()

        log_print(
            self.logger,
            "info",
            f"{session.name} connected ({client_count} client(s) connected)",
        )
        return session

    def _serve_client(self, session):
        """Run one client's command loop, then retire its session."""
        self._thread_session.session = session
        try:
            self._command_handler()
        finally:
            self._thread_session.session = None
            self._end_session(session)

    def _end_session(self, session):
        """Drop a session and close its connections. Safe to call twice."""
        with self._sessions_lock:
            if session in self.sessions:
                self.sessions.remove(session)
            remaining = len(self.sessions)
            if not remaining and self._idle_since is None:
                self._idle_since = time.monotonic()

        was_active = session.active
        session.close()

        if was_active:
            log_print(
                self.logger,
                "debug",
                f"{session.name} disconnected ({remaining} client(s) remaining)",
            )

    def close_client_connections(self):
        """Disconnect every client (server shutdown, or an unrecoverable error)."""
        log_print(self.logger, "debug", "Closing client connections")

        for session in self._live_sessions():
            self._end_session(session)

    def _data_handler(self):
        """Drain acquired data and fan each chunk out to every connected client.

        One thread serves the whole server rather than one per client: the data
        queue can only be drained once, so a chunk is read here and then written
        to each live session. A client whose socket has gone is dropped and the
        rest keep streaming.
        """
        while self.running:
            try:
                # Use a real (short) timeout so the loop periodically re-checks
                # self.running and the thread can exit cleanly when the server
                # stops or while streaming is paused (no data queued).
                buff = self.data_queue.get(timeout=1.0)

                # Backends push {'data': ndarray, 'sources': [source dicts]}.
                # The source list is forwarded as chunk metadata so the client
                # can route each row to the correct plot/save column.
                if isinstance(buff, dict) and "data" in buff:
                    data, meta = buff["data"], {"sources": buff.get("sources")}
                else:
                    data, meta = buff, None

                for session in self._live_sessions():
                    try:
                        with session.send_lock:
                            send_datachunk(session.data_conn, data, meta=meta)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        log_print(
                            self.logger,
                            "error",
                            f"{session.name} disconnected during data transmission.",
                        )
                        self._end_session(session)
            except queue.Empty:
                continue  # continue execution if no data arrived
            except Exception as e:
                # This thread serves every client for the whole life of the
                # server, so one bad chunk must not end streaming for good.
                # The queue.get() timeout keeps a persistent fault from
                # spinning here.
                log_print(self.logger, "error", f"Unexpected data handler error: {e}")
                continue

    def _command_handler(self):
        """Serve commands from the client bound to this thread until it goes away."""
        session = self._thread_session.session
        while self.running and session.active:
            try:
                # Receive commands (but we block while waiting)
                self.client_control_conn.settimeout(1.0)
                try:
                    data = recv_message(self.client_control_conn, self.logger)
                except socket.timeout:
                    continue  # ensure timeouts do not kill this thread
                except (OSError, ConnectionResetError) as e:
                    log_print(self.logger, "error", f"Connection reset by host: {e}")
                    break

                if not data:
                    break  # Control connection is closed

                # Parse received command and appropriately call background function
                cmd_type, payload = parse_and_validate_command(data)
                log_print(self.logger, "debug", f"Received {cmd_type} with {payload}")

                match cmd_type:
                    case Command.DISCONNECT_SERVER.name:
                        # A break is enough to close this thread and the
                        # client connection.
                        break

                    # Device commands
                    case Command.DISCOVER_DEVICES.name:
                        self._start_discover_devices_async(payload)
                    case Command.LIST_DEVICES.name:
                        self._handle_list_devices(payload)
                    case Command.SET_DEVICE_CONFIG.name:
                        self._handle_set_device_config(payload)
                    case Command.INITIALIZE_DEVICES.name:
                        self._start_initialize_devices_async(payload)
                    case Command.GET_DEVICE_STATUS.name:
                        self._handle_get_device_status()
                    case Command.DISCONNECT_DEVICES.name:
                        self._disconnect_devices()

                    # Streaming
                    case Command.START_STREAMING.name:
                        # Specify streaming parameters, typically pertaining
                        # to saving/display
                        self._start_streaming(payload)
                    case Command.STOP_STREAMING.name:
                        self._stop_streaming()
                    case Command.UPDATE_RUNNING_PARAMETER.name:
                        self._update_running_parameter(payload)
                    case Command.RUN_DPIC_BALANCE.name:
                        self._run_dpic_balance(payload)

            except ValidationError as e:
                log_print(self.logger, "debug", f"Invalid command {cmd_type} sent: {e}")
                continue  # Invalid command should not close connection

    def _is_local_client(self, address):
        """True when a peer address belongs on this machine or its LAN.

        Delegates to bioview_common.is_local_request rather than repeating the
        check: that version also accepts loopback and 0.0.0.0 explicitly, and
        returns False for a malformed address instead of raising.

        A same-machine client can arrive on one of this host's own NIC addresses
        rather than loopback (the client dials the address the server advertised
        in its discovery info). On a network that hands out public addresses that
        NIC address is not in any private range, so it is checked against this
        host's own addresses too -- otherwise a --local server refuses the very
        client that launched it.
        """
        if not isinstance(address, (list, tuple)) or not address:
            # Python sockets hand back an (ip, port) tuple.
            return False
        peer = address[0]
        return is_local_request(peer) or peer in get_local_addresses()

    # Device command handling callbacks

    def _config_from_payload(self, payload):
        from bioview_common import Configuration

        return Configuration.from_dict(payload.get("device_groups", payload))

    def _connecting_states_for_config(self, config):
        return {device_id: DeviceStatus.CONNECTING.value for device_id in config.devices}

    def _handle_get_device_status(self):
        send_response(
            sock=self.client_control_conn,
            response=Response.SUCCESS,
            params={
                "pending": self._device_op_in_progress,
                "device_status": self.device_group_states,
                "device_errors": dict(self.device_group_errors),
                "data_sources": [src.to_dict() for src in self.data_sources],
            },
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # Configurator support
    # ------------------------------------------------------------------

    def _enumerate_devices(self, include_virtual=False):
        """Every attached device across all loaded backends, config-free.

        Unlike DISCOVER_DEVICES this does not consult (or require) a loaded
        configuration -- the Configurator runs before any experiment config
        exists. A backend that raises is reported as unavailable rather than
        failing the whole listing, so one missing driver cannot hide the
        devices belonging to every other backend.

        Virtual (dummy) devices are left out unless asked for: the Configurator
        lists hardware that is actually attached, and a simulated device sitting
        among the real ones is misleading. Tests and development opt back in.

        A backend that failed to load is reported too, with the reason, so a
        missing driver or Python dependency shows up in the Configurator instead
        of the hardware silently never appearing.
        """
        devices = []
        backends = {
            backend_type: {
                "editable_properties": {},
                "available": False,
                "error": reason,
            }
            for backend_type, reason in UNAVAILABLE_BACKENDS.items()
            if include_virtual or backend_type != DeviceType.DUMMY.value
        }

        for backend_type, backend in AVAILABLE_BACKENDS.items():
            if backend_type == DeviceType.DUMMY.value and not include_virtual:
                continue
            schema = getattr(backend, "EDITABLE_PROPERTIES", {}) or {}
            entry = {"editable_properties": schema, "available": True}
            try:
                found = backend.discover_devices()
            except Exception as e:
                entry["available"] = False
                entry["error"] = str(e)
                backends[backend_type] = entry
                log_print(
                    self.logger,
                    "warning",
                    f"Listing devices failed for {backend_type}: {e}",
                )
                continue

            if isinstance(found, dict):
                found = list(found.values())
            for info in found or []:
                if not isinstance(info, dict):
                    info = {"name": str(info)}
                info = dict(info)
                info.setdefault("device_type", backend_type)
                info.setdefault("name", info.get("serial", "Unnamed Device"))
                info["editable"] = bool(schema)
                devices.append(info)

            backends[backend_type] = entry

        return devices, backends

    def _handle_list_devices(self, payload=None):
        include_virtual = bool((payload or {}).get("include_virtual", False))
        try:
            devices, backends = self._enumerate_devices(include_virtual)
        except Exception as e:
            log_print(self.logger, "error", f"Device listing failed: {e}")
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": str(e)},
                logger=self.logger,
            )
            return

        for info in devices:
            name = info.get("name")
            if name:
                self.discovered_devices_cache[name] = info

        log_print(self.logger, "info", f"Listed {len(devices)} attached device(s)")
        send_response(
            sock=self.client_control_conn,
            response=Response.DEVICE_LIST,
            params={"devices": devices, "backends": backends},
            logger=self.logger,
        )

    def _handle_set_device_config(self, payload):
        payload = payload or {}
        device_info = payload.get("device_info") or {}
        new_config = payload.get("config") or {}
        device_type = device_info.get("device_type")

        backend = AVAILABLE_BACKENDS.get(device_type)
        if backend is None:
            msg = f"No backend loaded for device type {device_type!r}"
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": msg},
                logger=self.logger,
            )
            return

        setter = getattr(backend, "set_device_config", None)
        if setter is None:
            msg = f"{device_type} devices have no editable properties"
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": msg},
                logger=self.logger,
            )
            return

        try:
            ok, message = setter(device_info, new_config, logger=self.logger)
        except Exception as e:
            ok, message = False, str(e)
            log_print(self.logger, "error", f"Device config update failed: {e}")

        if ok:
            log_print(
                self.logger,
                "info",
                f"Updated {device_info.get('name')}: {message}",
            )
            send_response(
                self.client_control_conn,
                Response.DEVICE_CONFIG_UPDATED,
                params={
                    "device_info": device_info,
                    "config": new_config,
                    "message": message,
                },
                logger=self.logger,
            )
        else:
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": message},
                logger=self.logger,
            )

    def _reject_if_device_op_running(self):
        with self._device_op_lock:
            if self._device_op_in_progress:
                send_response(
                    self.client_control_conn,
                    Response.ERROR,
                    params={"message": "Device operation already in progress"},
                    logger=self.logger,
                )
                return True
        return False

    def _ack_device_operation_start(self, payload):
        config = self._config_from_payload(payload)
        self.config = config
        self.device_group_states = self._connecting_states_for_config(config)
        send_response(
            sock=self.client_control_conn,
            response=Response.DEVICE_CONNECTING,
            params={
                "pending": True,
                "device_status": dict(self.device_group_states),
            },
            logger=self.logger,
        )

    def _start_discover_devices_async(self, payload):
        if self._reject_if_device_op_running():
            return

        self._ack_device_operation_start(payload)

        with self._device_op_lock:
            self._device_op_in_progress = True

        def _worker():
            try:
                self._discover_devices(payload)
            except Exception as e:
                log_print(
                    self.logger,
                    "error",
                    f"Background device discovery failed: {e}",
                )
            finally:
                with self._device_op_lock:
                    self._device_op_in_progress = False

        self._device_op_thread = Thread(target=_worker, daemon=True)
        self._device_op_thread.start()

    def _start_initialize_devices_async(self, payload):
        if self._reject_if_device_op_running():
            return

        self._ack_device_operation_start(payload)

        with self._device_op_lock:
            self._device_op_in_progress = True

        def _worker():
            try:
                self._initialize_devices_work(payload)
            except Exception as e:
                log_print(
                    self.logger,
                    "error",
                    f"Background device initialization failed: {e}",
                )
            finally:
                with self._device_op_lock:
                    self._device_op_in_progress = False

        self._device_op_thread = Thread(target=_worker, daemon=True)
        self._device_op_thread.start()

    def _discover_devices(self, payload):
        log_print(self.logger, "info", "Discovering connected devices")

        discovered_names = set()
        # Which backend each device came from. Availability for a backend must
        # not be decided from the combined pool: every backend's results land in
        # one cache, so a machine with only a virtual device would otherwise
        # look like it had BIOPAC hardware attached.
        discovered_by_backend = {}
        for backend_type, backend in AVAILABLE_BACKENDS.items():
            backend_names = set()
            try:
                found = backend.discover_devices()
                if isinstance(found, dict):
                    backend_names.update(found.keys())
                    self.discovered_devices_cache.update(found)
                elif isinstance(found, list):
                    for entry in found:
                        if isinstance(entry, dict):
                            name = entry.get("name", "")
                            backend_names.add(name)
                            if name:
                                self.discovered_devices_cache[name] = entry
                        else:
                            backend_names.add(str(entry))
            except Exception as e:
                msg = (
                    f"Device discovery failed for devices of type "
                    f"{backend_type} with error: {e}"
                )
                log_print(self.logger, "warning", msg)

            backend_names.discard("")
            discovered_by_backend[backend_type] = backend_names
            discovered_names.update(backend_names)

        discovered_names.discard("")
        log_print(self.logger, "debug", f"Found {sorted(discovered_names)}")

        if not self.config:
            self.config = self._config_from_payload(payload)

        self.device_group_states = {}
        self.device_group_errors = {}

        for device_id, device_cfg in self.config.devices.items():
            device_type = device_cfg.get_param("device_type")
            if device_type == DeviceType.DUMMY.value:
                self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                continue

            if device_type == DeviceType.BIOPAC.value:
                # A BIOPAC group's hardware keys are labels the user chose (say
                # "BIOPAC_MP36"), not identifiers the unit reports: discovery
                # names an MP unit after its Windows device name ("BIOPAC MP36
                # USB Data Acquisition Unit"), so the two almost never match and
                # requiring them to would mark attached hardware unavailable.
                # One MP unit is driven per group, so the group is available
                # when a BIOPAC unit was found at all.
                biopac_discovered = discovered_by_backend.get(
                    DeviceType.BIOPAC.value, set()
                )
                if biopac_discovered:
                    self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                else:
                    self.device_group_states[device_id] = DeviceStatus.UNAVAILABLE.value
                    self.device_group_errors[device_id] = (
                        "no BIOPAC unit was found. Check that it is powered on and "
                        "connected, and that its driver is installed"
                        if DeviceType.BIOPAC.value in AVAILABLE_BACKENDS
                        else "the BIOPAC backend is not available on this server: "
                        + UNAVAILABLE_BACKENDS.get(
                            DeviceType.BIOPAC.value, "unknown reason"
                        )
                    )
                continue

            if device_id in discovered_names:
                self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                continue

            hardware = device_cfg.get_param("hardware") or {}
            hw_names = set(hardware.keys()) if isinstance(hardware, dict) else set()
            # Available when any of the group's hardware entries was discovered.
            # (A full match was tested separately before; it is just the case
            # where the intersection happens to be all of hw_names.)
            if hw_names & discovered_names:
                self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
            else:
                self.device_group_states[device_id] = DeviceStatus.UNAVAILABLE.value
                backend_names = discovered_by_backend.get(device_type) or set()
                if device_type not in AVAILABLE_BACKENDS:
                    reason = (
                        f"the {device_type} backend is not available on this server: "
                        + UNAVAILABLE_BACKENDS.get(device_type, "unknown reason")
                    )
                elif not backend_names:
                    reason = f"no {device_type} hardware was found"
                else:
                    reason = (
                        f"none of the configured devices {sorted(hw_names)} were "
                        f"found. Attached: {sorted(backend_names)}"
                    )
                self.device_group_errors[device_id] = reason

        log_print(self.logger, "info", "Device discovery completed successfully")

    def _explain_device_failure(self, message: str) -> str:
        """Add machine-level context to a backend's failure message.

        A backend subprocess reports what the device told it; only the server
        can look at how this machine is configured. Kept here rather than in the
        backend because the queries involved have hung inside a subprocess, and
        a diagnostic must never stall the operation it is explaining.
        """
        if "MPDRVERR" not in message:
            return message

        try:
            from bioview_server.device.biopac.utils import driver_failure_hint

            return message + driver_failure_hint()
        except Exception as e:
            log_print(self.logger, "debug", f"Could not check driver context: {e}")
            return message

    def _active_device_handlers(self):
        return {
            device_id: handler
            for device_id, handler in self.device_group_handlers.items()
            if handler is not None
        }

    def _initialize_devices_work(self, payload):
        self.config = self._config_from_payload(payload)
        self._discover_devices(payload)

        if self.device_group_states == {}:
            log_print(self.logger, "error", "Invalid configuration provided")
            return

        log_print(self.logger, "info", "Initializing devices")

        self.device_group_handlers = {}
        self.device_group_errors = {}
        uninit_groups = []

        for device_id, device_cfg in self.config.devices.items():
            self.device_group_handlers[device_id] = None
            self.device_group_states[device_id] = DeviceStatus.CONNECTING.value
            handler = None

            try:
                handler = get_device_handler(
                    device_id,
                    device_cfg,
                    self.response_queue,
                    self.data_queue,
                    self.logger,
                    discovered_devices=self.discovered_devices_cache,
                )
                if not handler:
                    raise DeviceError(f"Unable to create handler for {device_id}")

                handler.start()

                resp = handler.initialize()
                if not _handler_init_succeeded(resp):
                    message = (resp or {}).get("message", "Unknown initialization error")
                    raise DeviceError(message)

                self.device_group_states[device_id] = DeviceStatus.CONNECTED.value
                self.data_sources.update(handler.get_data_sources())
                self.device_group_handlers[device_id] = handler
            except Exception as e:
                reason = self._explain_device_failure(str(e))
                msg = f"Unable to initialize device: {device_id}. Error: {reason}"
                log_print(self.logger, "error", msg)
                self.device_group_errors[device_id] = reason
                self.device_group_states[device_id] = DeviceStatus.UNAVAILABLE.value
                self.device_group_handlers[device_id] = None
                uninit_groups.append(device_id)
                if handler is not None:
                    with contextlib.suppress(Exception):
                        handler.shutdown()

        if len(uninit_groups) > 0:
            log_print(
                self.logger,
                "warning",
                f"Device initialization failed for groups: {uninit_groups}",
            )
        else:
            log_print(self.logger, "info", "All devices successfully initialized")

    def _disconnect_devices(self):
        active_handlers = self._active_device_handlers()
        if not active_handlers:
            msg = "Server has no initialized devices"
            log_print(self.logger, "warning", msg)
            send_response(
                self.client_control_conn,
                Response.SUCCESS,
                params={"message": msg},
                logger=self.logger,
            )
            return

        try:
            for handler in active_handlers.values():
                handler.disconnect()

            msg = "Devices disconnected successfully"
            log_print(self.logger, "info", msg)
            send_response(
                self.client_control_conn,
                Response.SUCCESS,
                params={"message": msg},
                logger=self.logger,
            )
        except Exception as e:
            msg = f"Failed to disconnect devices: {e}"
            log_print(self.logger, "error", msg)
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": msg},
                logger=self.logger,
            )

    # Handle streaming
    def _start_streaming(self, payload):
        active_handlers = self._active_device_handlers()
        if not active_handlers:
            msg = "Server has no initialized devices"
            log_print(self.logger, "error", msg)
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": msg},
                logger=self.logger,
            )
            return

        # Build a structured streaming config from the experiment configuration.
        # Saving happens on the client (fast disk), so server-side saving is off;
        # the display path is the live stream to the client and is always enabled.
        experiment_cfg = payload.get("Experiment", payload.get("experiment", {})) or {}
        stream_cfg = {
            "save_config": {"enable_save": False},
            "display_config": {
                "display_sources": experiment_cfg.get("display_sources", []),
            },
        }

        # Do not overwrite backend runtime state on every Start.
        # UI parameter edits are already propagated live via UPDATE_RUNNING_PARAMETER,
        # so resuming streaming should continue from the last runtime values.

        # Ask all backends to start
        try:
            log_print(self.logger, "info", "Attempting to start data streaming")

            # Start your existing receive/transmit workers
            for handler in active_handlers.values():
                handler.start_streaming(stream_cfg)

            msg = "Data streaming started successfully"
            log_print(self.logger, "info", msg)
            send_response(
                self.client_control_conn,
                Response.SUCCESS,
                params={"message": msg},
                logger=self.logger,
            )
        except Exception as e:
            msg = f"Failed to start streaming: {e}"
            log_print(self.logger, "error", msg)
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": msg},
                logger=self.logger,
            )

    def _stop_streaming(self):
        active_handlers = self._active_device_handlers()
        if not active_handlers:
            msg = "Server has no initialized devices"
            log_print(self.logger, "warning", msg)
            send_response(
                self.client_control_conn,
                Response.SUCCESS,
                params={"message": msg},
                logger=self.logger,
            )
            return

        try:
            log_print(self.logger, "info", "Attempting to stop data streaming")

            for handler in active_handlers.values():
                handler.stop_streaming()

            msg = "Data streaming stopped successfully"
            log_print(self.logger, "info", msg)
            send_response(
                self.client_control_conn, Response.SUCCESS, params={"message": msg}
            )
        except Exception as e:
            msg = f"Failed to stop streaming: {e}"
            log_print(self.logger, "error", msg)
            send_response(
                self.client_control_conn, Response.ERROR, params={"message": msg}
            )

    def _sync_device_params_from_payload(self, payload):
        """Apply latest client device configuration to live backends before streaming."""
        if not payload:
            return

        from bioview_common.datatypes.configuration.hardware_params import (
            GLOBAL_RX_PARAMS,
            GLOBAL_TX_PARAMS,
        )

        skip = {
            "type",
            "device_type",
            "cfg_type",
            "device_name",
            "absolute_channel_nums",
        }
        sync_keys = (
            GLOBAL_TX_PARAMS
            | GLOBAL_RX_PARAMS
            | {
                "calibration",
                "samp_rate",
                "signal_scheme",
                "signal_freq",
                "amplitude",
                "noise_std",
                "chunk_duration",
                "hardware",
                "channel_map",
                "channels",
                "model",
                "mpdev_path",
                "connection_type",
                "port",
            }
        )

        for device_id, handler in self.device_group_handlers.items():
            if handler is None:
                continue
            device_payload = payload.get(device_id)
            if not isinstance(device_payload, dict):
                continue

            if self.config:
                for param, value in device_payload.items():
                    if param in skip:
                        continue
                    self.config.update_device_param(device_id, param, value)

            sync_params = {
                k: v
                for k, v in device_payload.items()
                if k in sync_keys or str(k).startswith("calibration.")
            }
            if sync_params:
                handler.queue_param_update(**sync_params)

    def _update_running_parameter(self, payload):
        device_id = payload.get("id")
        config = payload.get("config")

        if not device_id or not config:
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": "Invalid payload"},
                logger=self.logger,
            )
            return

        log_print(self.logger, "info", f"Updating parameter for device {device_id}")

        # Update internal config
        if self.config:
            for param, value in config.items():
                self.config.update_device_param(device_id, param, value)

        # Find the handler managing this device. device_id is typically the group_id.
        handler = self.device_group_handlers.get(device_id)

        if handler is None:
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": "Device handler not found"},
                logger=self.logger,
            )
            return

        try:
            handler.queue_param_update(**config)
            # A parameter such as the BIOPAC channel mask changes which streams
            # the device produces, so the client is told the new source list in
            # the same reply rather than being left with the old one until the
            # next connect.
            self._refresh_data_sources()
            send_response(
                self.client_control_conn,
                Response.SUCCESS,
                params={
                    "message": "Parameter updated",
                    "data_sources": [src.to_dict() for src in self.data_sources],
                },
                logger=self.logger,
            )
        except Exception as e:
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": str(e)},
                logger=self.logger,
            )

    def _refresh_data_sources(self):
        """Rebuild the advertised source list from every live device handler.

        Rebuilt rather than merged: a channel that has just been disabled has to
        disappear, which a set union can never do.
        """
        sources = set()
        for handler in self.device_group_handlers.values():
            if handler is None:
                continue
            with contextlib.suppress(Exception):
                sources.update(handler.get_data_sources())
        self.data_sources = sources
        return self.data_sources

    def _run_dpic_balance(self, payload):
        device_id = payload.get("id") if payload else None
        if not device_id and self.device_group_handlers:
            device_id = next(iter(self.device_group_handlers))

        handler = self.device_group_handlers.get(device_id)
        if handler is None:
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": "Device handler not found"},
                logger=self.logger,
            )
            return

        try:
            response = handler.run_dpic_balance()
            if response.get("type") in (
                Response.SUCCESS,
                Response.SUCCESS.name,
                Response.SUCCESS.value,
            ):
                send_response(
                    self.client_control_conn,
                    Response.SUCCESS,
                    params={"message": "DPIC balance complete"},
                    logger=self.logger,
                )
            else:
                send_response(
                    self.client_control_conn,
                    Response.ERROR,
                    params={"message": response.get("message", "DPIC failed")},
                    logger=self.logger,
                )
        except Exception as e:
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": str(e)},
                logger=self.logger,
            )

    def stop(self):
        log_print(self.logger, "debug", "Attempting to shutdown server")

        # Drop the flag first: the data thread and every client's command loop
        # watch it, so they wind down while the connections are being closed.
        self.running = False

        # Close any active client connections
        self.close_client_connections()

        # Close sockets
        if self.control_socket:
            self.control_socket.close()
        self.control_socket = None

        if self.data_socket:
            self.data_socket.close()
        self.data_socket = None

        if self.data_thread is not None:
            self.data_thread.join(timeout=2.0)
            self.data_thread = None

        log_print(self.logger, "debug", "Server shut down successfully")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Launch BioView Backend Server")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Flag to make server restricteed only to local clients",
    )
    parser.add_argument(
        "--exit-when-idle",
        type=float,
        default=0,
        help=(
            "Shut down after this many seconds with no client connected. Used by "
            "the GUI launcher so a shared server retires itself once every "
            "BioView window has closed. 0 (default) means never."
        ),
    )
    parser.add_argument(
        "--control-port",
        type=int,
        help=f"Port number to use for control connections. Default: {CONTROL_PORT}",
        required=False,
        default=CONTROL_PORT,
    )
    parser.add_argument(
        "--data-port",
        type=int,
        help=f"Port number to use for data connections. Default: {DATA_PORT}",
        required=False,
        default=DATA_PORT,
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s: (%(levelname)s) %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    log_print(logger, "info", f"BioView Device Server, Version: {APP_VERSION}")

    args = parser.parse_args(argv)

    server = Server(
        local_only=args.local,
        exit_when_idle=args.exit_when_idle,
        control_port=args.control_port,
        data_port=args.data_port,
        logger=logger,
    )

    # Stop cleanly when the launcher (or the OS) asks us to terminate, so sockets
    # are released promptly when the GUI that spawned us closes.
    def _handle_termination(signum, frame):
        log_print(logger, "info", f"Received signal {signum}. Shutting down server...")
        server.running = False

    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, _handle_termination)

    exit_code = 0
    try:
        server.start()
    except KeyboardInterrupt:
        log_print(
            logger, "warning", "Keyboard interrupt received. Shutting down server..."
        )
    except OSError as e:
        # Most likely the control/data port is already bound by another server.
        log_print(logger, "error", f"Unable to bind server sockets ({e}). Exiting...")
        exit_code = 1
    except Exception:
        log_print(logger, "error", "Server error. Shutting down server...")
        exit_code = 1
    finally:
        try:
            server.stop()
        except Exception:
            log_print(logger, "error", "Unable to shut down server. Exiting...")

    return exit_code


if __name__ == "__main__":
    import sys

    sys.exit(main())
