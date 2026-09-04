"""BioView server: forwards client commands to device backends.

Several clients share one server and one set of hardware.
See bioview-docs/architecture/server.md.
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
    """One connected client: its control connection, data connection and info."""

    def __init__(self, control_conn, data_conn, info=None):
        self.control_conn = control_conn
        self.data_conn = data_conn
        self.info = info or {}
        self.active = True
        self.thread = None
        # Serializes writes to control_conn against out-of-band device reports.
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
        self.info = get_app_info()
        self.token = 42  # TODO: Load using secrets

        self.control_port = control_port
        self.data_port = data_port

        self.running = False

        # Seconds with no client before the server retires itself; 0 = never.
        self.exit_when_idle = exit_when_idle
        self._idle_since = time.monotonic()

        # Guarded: the accept loop adds, command threads remove, the data
        # thread iterates.
        self.sessions = []
        self._sessions_lock = Lock()

        # Thread-local: the session whose command this thread is handling, so
        # a reply goes back to the client that asked.
        self._thread_session = local()

        self.local_only = local_only
        self.discovered_clients = {}
        self.connected_client_info = {}

        self.device_group_states = {}
        self.device_group_handlers = {}
        # Why a group is not usable, keyed by group id, and reported to clients.
        self.device_group_errors = {}
        self.config = None
        self.data_sources = set()  # set(source: DataSource)
        self.discovered_devices_cache = {}
        self._device_op_lock = Lock()
        self._device_op_in_progress = False

        # Start/Stop is served on the session's own command thread, so two
        # clients -- or one client whose Start button stayed live through a
        # minute-long start -- can drive a device through START_STREAMING while
        # it is already streaming. See _start_streaming().
        self._streaming_lock = Lock()
        self._streaming_active = False
        self._device_op_thread = None

        self.data_socket = None
        self.control_socket = None

        self.data_thread = None

        # One response queue per device backend, created on handler creation.
        # A queue shared by all of them lets one device consume another's reply.
        self.response_queues = {}

        self.data_queue = mp.Queue(maxsize=DATA_OUTPUT_QUEUE_DEPTH)

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

        self._create_sockets()

        self.running = True

        # One data thread for the whole server, fanning out to every client.
        self.data_thread = Thread(target=self._data_handler, daemon=True)
        self.data_thread.start()

        while self.running:
            control_conn = None
            self._check_idle_exit()
            try:
                try:
                    self.control_socket.settimeout(1.0)
                    control_conn, addr = self.control_socket.accept()
                    log_print(
                        self.logger, "debug", f"Control connection initiated from {addr}"
                    )
                except socket.timeout:
                    # No one connected yet; loop back and re-check self.running.
                    continue
                except OSError:
                    break

                control_conn.settimeout(5.0)

                if self.local_only and not self._is_local_client(addr):
                    control_conn.close()
                    continue

                auth_data = recv_message(control_conn, self.logger)
                if not auth_data:
                    control_conn.close()
                    continue

                cmd_type, payload = parse_and_validate_command(auth_data)
                if cmd_type == Command.DISCOVER_SERVERS.name:
                    send_response(
                        sock=control_conn,
                        response=Response.SUCCESS,
                        # Lets a closing window tell whether another still needs us.
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
                        # TODO: log the invalid connection attempt.
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

                    # Local, not shared: several clients may connect at once.
                    client_info = payload.get("client_info") or {}
                    session_info = {
                        "ip": client_info.get("ip", ""),
                        "hostname": client_info.get("hostname", ""),
                        "name": client_info.get("name", ""),
                        "version": client_info.get("version", ""),
                    }
                    self.connected_client_info = session_info
                else:
                    control_conn.close()
                    continue

                """Accept the data connection for a client that just authenticated."""
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

                # Served on its own thread so this loop stays free for the
                # next window to connect.
                self.handle_client_session(control_conn, data_conn, session_info)

            except Exception as e:
                # Abandon only this connection attempt, never the other clients.
                log_print(self.logger, "error", f"Error in main loop: {e}")
                if control_conn is not None:
                    with contextlib.suppress(Exception):
                        control_conn.close()

    def _create_sockets(self):
        """Bind the control and data listeners. Done once, at launch."""
        try:
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Exclusive: a second server must fail to bind rather than run
            # alongside this one and split the clients between them.
            set_exclusive_bind(self.control_socket)
            self.control_socket.bind(("0.0.0.0", self.control_port))
            # A LAN discovery scan opens many short-lived probes at once.
            self.control_socket.listen(socket.SOMAXCONN)
            self.control_socket.settimeout(1)  # Make sure that accept is non-blocking
            log_print(self.logger, "debug", "Control socket created")
        except OSError as e:
            # Almost always another BioView server; surface it so the caller
            # can exit rather than spin on an unbound socket.
            log_print(self.logger, "error", f"Unable to create control socket: {e}")
            raise

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

        Resolving it per thread routes every reply to the client that asked
        without threading a session argument through every handler.
        """
        session = getattr(self._thread_session, "session", None)
        return session.control_conn if session is not None else None

    @property
    def client_session_active(self):
        """True while at least one client is connected."""
        return bool(self._live_sessions())

    def _check_idle_exit(self):
        """Shut down once no client has connected for ``exit_when_idle`` seconds.

        This is what lets the window that started the server not be the last
        one standing, without leaving a server behind.
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
        """Register a newly authenticated client and serve it on its own thread."""
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

        One thread, not one per client: the data queue can only be drained once.
        """
        while self.running:
            try:
                # A short timeout lets the loop re-check self.running and exit.
                buff = self.data_queue.get(timeout=1.0)

                # Backends push {'data': ndarray, 'sources': [...]}; the source
                # list is what routes each row on the client.
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
                # One bad chunk must not end streaming for every client.
                log_print(self.logger, "error", f"Unexpected data handler error: {e}")
                continue

    def _command_handler(self):
        """Serve commands from the client bound to this thread until it goes away."""
        session = self._thread_session.session
        while self.running and session.active:
            try:
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

                cmd_type, payload = parse_and_validate_command(data)
                log_print(self.logger, "debug", f"Received {cmd_type} with {payload}")

                match cmd_type:
                    case Command.DISCONNECT_SERVER.name:
                        break

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

                    case Command.START_STREAMING.name:
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

        This host's own NIC addresses count: a same-machine client dials the
        address the server advertised, which need not be loopback or private.
        """
        if not isinstance(address, list | tuple) or not address:
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

    # Configurator support

    def _enumerate_devices(self, include_virtual=False):
        """Every attached device across all loaded backends, config-free.

        A backend that raises is reported as unavailable rather than failing the
        whole listing. Virtual devices are excluded unless explicitly asked for.
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
            # The listing cache is keyed on device name, so a rename must
            # evict the old entry.
            self.discovered_devices_cache.pop(device_info.get("name"), None)
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
        # Every backend's results land in one cache, so availability has to be
        # decided per backend rather than from the combined pool.
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
                # Hardware keys are user-chosen labels, not the device names
                # discovery reports, and one MP unit is driven per group.
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
            # Available when any of the group's hardware entries was found.
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

        Kept out of the backend subprocess: these queries have been seen to hang.
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
                self.response_queues[device_id] = mp.Queue()
                handler = get_device_handler(
                    device_id,
                    device_cfg,
                    self.response_queues[device_id],
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

    def _start_streaming(self, payload):
        with self._streaming_lock:
            if self._streaming_active:
                # Idempotent rather than an error: the session *is* streaming,
                # which is what the client asked for, and answering ERROR would
                # drop a working client into a failed state. Restarting the
                # devices under a running session is the harmful reading.
                msg = "Data streaming already in progress"
                log_print(self.logger, "warning", msg)
                send_response(
                    self.client_control_conn,
                    Response.SUCCESS,
                    params={"message": msg},
                    logger=self.logger,
                )
                return

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

        # Saving happens on the client, so server-side saving stays off. Runtime
        # parameters are not replayed: UI edits already reach backends live.
        experiment_cfg = payload.get("Experiment", payload.get("experiment", {})) or {}
        stream_cfg = {
            "save_config": {"enable_save": False},
            "display_config": {
                "display_sources": experiment_cfg.get("display_sources", []),
            },
        }

        log_print(self.logger, "info", "Attempting to start data streaming")

        started = []
        failures = []
        for device_id, handler in active_handlers.items():
            try:
                handler.start_streaming(stream_cfg)
                started.append((device_id, handler))
            except Exception as e:
                reason = str(e) or type(e).__name__
                log_print(self.logger, "error", f"{device_id} failed to start: {reason}")
                failures.append(f"{device_id}: {reason}")

        if failures:
            # Never leave half the rig running: a partially started session
            # writes data that cannot be aligned across devices.
            for _device_id, handler in started:
                with contextlib.suppress(Exception):
                    handler.stop_streaming()

            msg = "Failed to start streaming -- " + "; ".join(failures)
            log_print(self.logger, "error", msg)
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": msg},
                logger=self.logger,
            )
            return

        with self._streaming_lock:
            self._streaming_active = True

        msg = "Data streaming started successfully"
        log_print(self.logger, "info", msg)
        send_response(
            self.client_control_conn,
            Response.SUCCESS,
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

        log_print(self.logger, "info", "Attempting to stop data streaming")

        with self._streaming_lock:
            self._streaming_active = False

        # One device refusing to stop must not leave the others running.
        failures = []
        for device_id, handler in active_handlers.items():
            try:
                handler.stop_streaming()
            except Exception as e:
                failures.append(f"{device_id}: {str(e) or type(e).__name__}")

        if failures:
            msg = "Failed to stop streaming -- " + "; ".join(failures)
            log_print(self.logger, "error", msg)
            send_response(
                self.client_control_conn, Response.ERROR, params={"message": msg}
            )
            return

        msg = "Data streaming stopped successfully"
        log_print(self.logger, "info", msg)
        send_response(
            self.client_control_conn, Response.SUCCESS, params={"message": msg}
        )

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

        if self.config:
            for param, value in config.items():
                self.config.update_device_param(device_id, param, value)

        # device_id is the group_id.
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
            # exist, so the new source list rides back in the same reply.
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

        Rebuilt, not merged: a set union can never drop a disabled channel.
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

        # Dropped first: the data thread and every command loop watch it.
        self.running = False

        self.close_client_connections()

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

    # Release the sockets promptly when the GUI that spawned us closes.
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
