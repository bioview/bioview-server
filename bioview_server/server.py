"""BioView server: forwards client commands to device backends."""

import argparse
import contextlib
import logging
import multiprocessing as mp
import queue
import signal
import socket
import time
from pathlib import Path
from threading import Lock, Thread, local

from bioview_common import (
    APP_VERSION,
    CONTROL_PORT,
    DATA_OUTPUT_QUEUE_DEPTH,
    DATA_PORT,
    SAVE_OUTPUT_QUEUE_DEPTH,
    Command,
    DeviceError,
    DeviceStatus,
    DeviceType,
    Response,
    ValidationError,
    drain,
    generate_challenge,
    get_app_info,
    get_local_addresses,
    get_unique_path,
    is_local_request,
    log_print,
    parse_and_validate_command,
    recv_message,
    send_datachunk,
    send_response,
    set_exclusive_bind,
    validate_token,
)

from bioview_server.admission import (
    DEFAULT_POLICY,
    POLICIES,
    AdmissionController,
    AdmissionError,
    resolve_hostnames,
)
from bioview_server.common import BvrWriter
from bioview_server.device import (
    AVAILABLE_BACKENDS,
    UNAVAILABLE_BACKENDS,
    backend_report,
    get_device_handler,
)


SLEEP_DURATION = 0.001

RECORDING_CLOSE_TIMEOUT_S = 15

WINDOW_CLAIM_LIFETIME = 30.0


def _handler_init_succeeded(resp: dict) -> bool:
    """True only when a backend subprocess reports a successful connect."""
    if not resp or not isinstance(resp, dict):
        return False
    resp_type = resp.get("type")
    if resp_type in (Response.ERROR, Response.ERROR.name):
        return False
    if resp_type not in (Response.SUCCESS, Response.SUCCESS.name):
        return False
    return resp.get("result") is not False


class ClientSession:
    """One connected client: its control connection, data connection and info."""

    def __init__(self, control_conn, data_conn, info=None):
        self.control_conn = control_conn
        self.data_conn = data_conn
        self.info = info or {}
        self.active = True
        self.thread = None
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
        control_port: int,
        data_port: int,
        local_only: bool = None,
        allow: str = None,
        trusted=None,
        logger=None,
        exit_when_idle: float = 0,
    ):
        """``allow`` is the admission policy; see ``bioview_server.admission``."""
        self.info = get_app_info()
        self.info["backends"] = backend_report()
        self.token = 42

        self.control_port = control_port
        self.data_port = data_port

        self.info["control_port"] = control_port
        self.info["data_port"] = data_port

        self.running = False

        self.exit_when_idle = exit_when_idle
        self._idle_since = time.monotonic()

        self._windows = {}
        self._windows_lock = Lock()

        self.sessions = []
        self._sessions_lock = Lock()

        self._handshake_lock = Lock()

        self._thread_session = local()

        if allow is None:
            allow = "local" if local_only else DEFAULT_POLICY
        self._admission = AdmissionController(
            policy=allow, trusted=trusted, logger=logger
        )

        self.discovered_clients = {}
        self.connected_client_info = {}

        self.device_group_states = {}
        self.device_group_handlers = {}
        self.device_group_errors = {}
        self.config = None
        self.data_sources = set()
        self.discovered_devices_cache = {}
        self._device_op_lock = Lock()
        self._device_op_in_progress = False

        self._streaming_lock = Lock()
        self._streaming_active = False
        self._device_op_thread = None

        self._dpic_lock = Lock()
        self._dpic_threads = {}
        self._dpic_states = {}
        self._dpic_last_device = None

        self.data_socket = None
        self.control_socket = None

        self.data_thread = None

        self.response_queues = {}

        self.data_queue = mp.Queue(maxsize=DATA_OUTPUT_QUEUE_DEPTH)

        self.save_data_queue = mp.Queue(maxsize=SAVE_OUTPUT_QUEUE_DEPTH)
        self.bvr_writer = None

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

        self.data_thread = Thread(target=self._data_handler, daemon=True)
        self.data_thread.start()

        while self.running:
            control_conn = None
            self._check_abandoned_recording()
            self._check_idle_exit()
            try:
                try:
                    self.control_socket.settimeout(1.0)
                    control_conn, addr = self.control_socket.accept()
                    log_print(
                        self.logger, "debug", f"Control connection initiated from {addr}"
                    )
                except TimeoutError:
                    continue
                except OSError:
                    break

                control_conn.settimeout(5.0)
                self._dispatch_connection(control_conn, addr)
                control_conn = None

            except Exception as e:
                log_print(self.logger, "error", f"Error in main loop: {e}")
                if control_conn is not None:
                    with contextlib.suppress(Exception):
                        control_conn.close()

    def _peer_address(self, addr) -> str:
        """The peer's IP out of an ``accept()`` address tuple, or ""."""
        if isinstance(addr, list | tuple) and addr:
            return str(addr[0])
        return ""

    def _dispatch_connection(self, control_conn, addr):
        """Read one connection's opening message and route it."""
        peer = self._peer_address(addr)

        try:
            auth_data = recv_message(control_conn, self.logger)
        except Exception:
            auth_data = None
        if not auth_data:
            control_conn.close()
            return

        cmd_type, payload = parse_and_validate_command(auth_data)

        if cmd_type == Command.DISCOVER_SERVERS.name:
            if not self._admission.may_probe(peer):
                control_conn.close()
                return
            self._update_window_claim(payload)
            send_response(
                sock=control_conn,
                response=Response.SUCCESS,
                params={
                    **self.info,
                    "clients": len(self._live_sessions()),
                    "windows": self._live_window_count(),
                },
                logger=self.logger,
            )
            control_conn.close()
            return

        if cmd_type == Command.SHUTDOWN_SERVER.name:
            self._handle_shutdown_request(control_conn, peer)
            return

        if cmd_type != Command.CONNECT_SERVER.name:
            control_conn.close()
            return

        Thread(
            target=self._complete_connection,
            args=(control_conn, peer, payload or {}),
            name=f"bioview-connect-{peer or 'unknown'}",
            daemon=True,
        ).start()

    def _handle_shutdown_request(self, control_conn, peer):
        """Retire the server on request, but only for its own machine."""
        try:
            if not self._admission.is_own_machine(peer):
                log_print(
                    self.logger,
                    "warning",
                    f"Ignoring a shutdown request from {peer}: not this machine",
                )
                send_response(
                    sock=control_conn,
                    response=Response.ERROR,
                    params={"message": "Only this machine may shut this server down"},
                    logger=self.logger,
                )
                return

            log_print(self.logger, "info", "Shutdown requested; retiring server")
            send_response(
                sock=control_conn,
                response=Response.SUCCESS,
                params={"message": "Shutting down"},
                logger=self.logger,
            )
            self.running = False
        finally:
            with contextlib.suppress(Exception):
                control_conn.close()

    def _complete_connection(self, control_conn, peer, payload):
        """Admit, authenticate and register one client. Runs on its own thread."""
        client_info = payload.get("client_info") or {}
        try:
            allowed, reason = self._admission.decide(peer, client_info)
            if not allowed:
                log_print(
                    self.logger,
                    "warning",
                    f"Refused a connection from {peer}: {reason}",
                )
                with contextlib.suppress(Exception):
                    send_response(
                        sock=control_conn,
                        response=Response.ERROR,
                        params={"message": f"Connection refused: {reason}"},
                        logger=self.logger,
                    )
                control_conn.close()
                return

            hostname = client_info.get("hostname") or peer
            log_print(self.logger, "info", f"Incoming connection from: {hostname}")

            with self._handshake_lock:
                self._authenticate_and_register(control_conn, peer, client_info)
        except Exception as e:
            log_print(self.logger, "error", f"Connection from {peer} failed: {e}")
            with contextlib.suppress(Exception):
                control_conn.close()

    def _authenticate_and_register(self, control_conn, peer, client_info):
        """Challenge, verify, take the data connection and start the session."""
        challenge = generate_challenge()
        send_response(
            sock=control_conn,
            response=Response.SERVER_CHALLENGE,
            params={"challenge": challenge, "timestamp": time.time()},
            logger=self.logger,
        )

        challenge_response = recv_message(control_conn, self.logger)
        client_cmd, client_payload = parse_and_validate_command(challenge_response)

        if client_cmd != Command.AUTHENTICATE_CLIENT.name:
            log_print(
                self.logger,
                "warning",
                f"{peer} did not answer the challenge; dropping the connection",
            )
            control_conn.close()
            return

        auth_token = (client_payload or {}).get("token", None)
        if not (auth_token and validate_token(challenge, auth_token)):
            log_print(
                self.logger,
                "warning",
                f"{peer} failed authentication; dropping the connection",
            )
            control_conn.close()
            return

        send_response(
            sock=control_conn,
            response=Response.AUTHENTICATION_SUCCESS,
            params={"server_info": self.info, "timestamp": time.time()},
            logger=self.logger,
        )

        session_info = {
            "ip": client_info.get("ip", "") or peer,
            "hostname": client_info.get("hostname", ""),
            "name": client_info.get("name", ""),
            "version": client_info.get("version", ""),
        }
        self.connected_client_info = session_info

        try:
            data_conn, _ = self.data_socket.accept()
            log_print(self.logger, "debug", "Data connection accepted.")
        except TimeoutError:
            log_print(
                self.logger,
                "error",
                "Client failed to connect data socket in time.",
            )
            control_conn.close()
            return

        self.handle_client_session(control_conn, data_conn, session_info)

    def _create_sockets(self):
        """Bind the control and data listeners. Done once, at launch."""
        try:
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            set_exclusive_bind(self.control_socket)
            self.control_socket.bind(("0.0.0.0", self.control_port))
            self.control_socket.listen(socket.SOMAXCONN)
            self.control_socket.settimeout(1)
            log_print(self.logger, "debug", "Control socket created")
        except OSError as e:
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
    def local_only(self) -> bool:
        """True when this server will not serve an arbitrary remote machine."""
        return self._admission.policy in ("loopback", "local")

    @property
    def admission_policy(self) -> str:
        return self._admission.policy

    @property
    def client_control_conn(self):
        """The control connection of the client this thread is serving."""
        session = getattr(self._thread_session, "session", None)
        return session.control_conn if session is not None else None

    @property
    def client_session_active(self):
        """True while at least one client is connected."""
        return bool(self._live_sessions())

    def _update_window_claim(self, payload):
        """Record, refresh or withdraw one window's claim on this server."""
        if not isinstance(payload, dict):
            return
        token = payload.get("window")
        if not token or not isinstance(token, str):
            return

        role = payload.get("role") or "window"

        with self._windows_lock:
            if payload.get("leaving"):
                if self._windows.pop(token, None) is not None:
                    log_print(
                        self.logger,
                        "debug",
                        f"{role} window released this server "
                        f"({len(self._windows)} window(s) still holding it)",
                    )
                return

            try:
                interval = float(payload.get("heartbeat") or 0)
            except (TypeError, ValueError):
                interval = 0
            lifetime = 3 * interval if interval > 0 else WINDOW_CLAIM_LIFETIME
            first_seen = token not in self._windows
            self._windows[token] = (time.monotonic() + lifetime, role)

        if first_seen:
            log_print(self.logger, "debug", f"{role} window claimed this server")

    def _live_window_count(self) -> int:
        """How many windows are currently claiming this server, lapsed ones dropped."""
        now = time.monotonic()
        with self._windows_lock:
            lapsed = [t for t, (expiry, _) in self._windows.items() if expiry <= now]
            for token in lapsed:
                role = self._windows.pop(token)[1]
                log_print(
                    self.logger,
                    "debug",
                    f"{role} window stopped answering; its claim has lapsed",
                )
            return len(self._windows)

    def _in_use(self) -> bool:
        """True while anyone is connected, or any window says it still wants this."""
        with self._sessions_lock:
            if self.sessions:
                return True
        return self._live_window_count() > 0

    def _check_abandoned_recording(self):
        """Second line of defence against a recording nobody is watching."""
        if self.bvr_writer is None and not self._streaming_active:
            return
        if self._live_sessions():
            return
        self._halt_orphaned_streaming("no client is connected")
        if self.bvr_writer is not None and not self._live_sessions():
            log_print(
                self.logger,
                "warning",
                "Closing a recording that no client is left to stop",
            )
            self._close_recording()

    def _check_idle_exit(self):
        """Shut down once nothing has wanted this server for ``exit_when_idle``."""
        if not self.exit_when_idle:
            return

        if self._in_use():
            self._idle_since = None
            return

        if self._idle_since is None:
            self._idle_since = time.monotonic()
            return

        if time.monotonic() - self._idle_since < self.exit_when_idle:
            return

        log_print(
            self.logger,
            "info",
            f"Nothing has needed this server for {self.exit_when_idle:g}s. "
            "Shutting down...",
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

        was_active = session.active
        session.close()

        if was_active:
            log_print(
                self.logger,
                "debug",
                f"{session.name} disconnected ({remaining} client(s) remaining)",
            )

        if remaining == 0:
            self._halt_orphaned_streaming(
                "the last client disconnected while data was streaming"
            )

    def _halt_orphaned_streaming(self, why: str):
        """Stop devices and finalize the recording when no client is left."""
        with self._streaming_lock:
            if not self._streaming_active:
                return
            self._streaming_active = False

        if self._live_sessions():
            with self._streaming_lock:
                self._streaming_active = True
            return

        log_print(self.logger, "warning", f"Stopping data streaming: {why}")

        for device_id, handler in self._active_device_handlers().items():
            try:
                handler.stop_streaming()
            except Exception as e:
                log_print(
                    self.logger,
                    "error",
                    f"{device_id} did not stop cleanly: {e}",
                )

        self._close_recording()

    def close_client_connections(self):
        """Disconnect every client (server shutdown, or an unrecoverable error)."""
        log_print(self.logger, "debug", "Closing client connections")

        for session in self._live_sessions():
            self._end_session(session)

    def _data_handler(self):
        """Drain acquired data and fan each chunk out to every connected client."""
        while self.running:
            try:
                buff = self.data_queue.get(timeout=1.0)

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
                continue
            except Exception as e:
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
                except TimeoutError:
                    continue
                except (OSError, ConnectionResetError) as e:
                    log_print(self.logger, "error", f"Connection reset by host: {e}")
                    break

                if not data:
                    break

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
                    case Command.MARK_EVENT.name:
                        self._mark_event(payload)

            except ValidationError as e:
                log_print(self.logger, "debug", f"Invalid command {cmd_type} sent: {e}")
                continue

    def _is_local_client(self, address):
        """True when a peer address belongs on this machine or its LAN."""
        if not isinstance(address, list | tuple) or not address:
            return False
        peer = address[0]
        return is_local_request(peer) or peer in get_local_addresses()

    def _config_from_payload(self, payload):
        from bioview_common import Configuration

        return Configuration.from_dict(payload.get("device_groups", payload))

    def _connecting_states_for_config(self, config):
        return {device_id: DeviceStatus.CONNECTING.value for device_id in config.devices}

    IDLE_DPIC_STATE = {
        "pending": False,
        "ok": None,
        "message": "",
        "results": [],
        "device_id": None,
        "progress": None,
    }

    def _drain_dpic_progress(self):
        """Fold each group's newest progress entry into its balance state."""
        with self._dpic_lock:
            device_ids = list(self._dpic_states)
        for device_id in device_ids:
            handler = self.device_group_handlers.get(device_id)
            if handler is None or not hasattr(handler, "drain_balance_progress"):
                continue
            progress = handler.drain_balance_progress()
            if progress is None:
                continue
            with self._dpic_lock:
                state = self._dpic_states.get(device_id)
                if state is not None:
                    state["progress"] = progress

    def _dpic_status_payload(self):
        """The per-group balance states, plus the one-group view for old clients."""
        with self._dpic_lock:
            states = {
                device_id: dict(state) for device_id, state in self._dpic_states.items()
            }
            last = self._dpic_last_device
        legacy = states.get(last) or dict(self.IDLE_DPIC_STATE)
        return states, legacy

    def _handle_get_device_status(self):
        self._drain_dpic_progress()
        dpic_states, dpic_state = self._dpic_status_payload()
        send_response(
            sock=self.client_control_conn,
            response=Response.SUCCESS,
            params={
                "pending": self._device_op_in_progress,
                "device_status": self.device_group_states,
                "device_errors": dict(self.device_group_errors),
                "data_sources": [src.to_dict() for src in self.data_sources],
                "dpic_balances": dpic_states,
                "dpic_balance": dpic_state,
            },
            logger=self.logger,
        )

    def _enumerate_devices(self):
        """Every attached device across all loaded backends, config-free."""
        devices = []
        backends = {
            backend_type: {
                "editable_properties": {},
                "available": False,
                "error": reason,
            }
            for backend_type, reason in UNAVAILABLE_BACKENDS.items()
        }

        for backend_type, backend in AVAILABLE_BACKENDS.items():
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
        try:
            devices, backends = self._enumerate_devices()
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

            if device_type == DeviceType.BIOPAC.value:
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

            if device_type == DeviceType.MICROPHONE.value:
                mic_discovered = discovered_by_backend.get(
                    DeviceType.MICROPHONE.value, set()
                )
                if mic_discovered:
                    self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                else:
                    self.device_group_states[device_id] = DeviceStatus.UNAVAILABLE.value
                    self.device_group_errors[device_id] = (
                        "no audio input device was found. Check that a microphone "
                        "is attached and enabled in the sound settings"
                        if DeviceType.MICROPHONE.value in AVAILABLE_BACKENDS
                        else "the microphone backend is not available on this "
                        "server: "
                        + UNAVAILABLE_BACKENDS.get(
                            DeviceType.MICROPHONE.value, "unknown reason"
                        )
                    )
                continue

            if device_id in discovered_names:
                self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                continue

            hardware = device_cfg.get_param("hardware") or {}
            hw_names = set(hardware.keys()) if isinstance(hardware, dict) else set()
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
        """Add machine-level context to a backend's failure message."""
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
                    save_output_queue=self.save_data_queue,
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
            for device_id, handler in active_handlers.items():
                handler.disconnect()
                try:
                    handler.shutdown()
                except Exception as e:
                    log_print(self.logger, "debug", f"{device_id} shutdown failed: {e}")
                self.device_group_handlers[device_id] = None
                self.device_group_states[device_id] = DeviceStatus.DISCONNECTED.value

            self.data_sources = set()

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

        experiment_cfg = payload.get("Experiment", payload.get("experiment", {})) or {}
        enable_save = self._open_recording(experiment_cfg, active_handlers)
        stream_cfg = {
            "save_config": {"enable_save": enable_save},
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
            for _device_id, handler in started:
                with contextlib.suppress(Exception):
                    handler.stop_streaming()
            self._close_recording()

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

    def _open_recording(self, experiment_cfg, active_handlers) -> bool:
        """Open this session's .bvr file. Returns whether saving is on."""
        self._close_recording()

        file_name = (experiment_cfg.get("file_name") or "").strip()
        save_dir = experiment_cfg.get("save_dir") or ""
        if not file_name:
            log_print(self.logger, "debug", "No file name given; not recording")
            return False

        directory = Path(save_dir) if save_dir else Path.cwd()
        if not directory.is_dir():
            fallback = Path.cwd()
            log_print(
                self.logger,
                "warning",
                f"Save directory {directory} does not exist on the server; "
                f"recording to {fallback} instead",
            )
            directory = fallback

        base = Path(file_name).stem or "bioview_recording"
        label = (experiment_cfg.get("save_label") or "").strip()
        if label:
            base = f"{base}_{label}"

        devices = []
        for device_id, handler in active_handlers.items():
            try:
                devices.append(handler.describe_for_recording())
            except Exception as e:
                log_print(
                    self.logger,
                    "error",
                    f"{device_id} could not describe itself for the recording ({e}); "
                    "its data will not be saved",
                )
        if not devices:
            return False

        sample_limits = self._sample_limits(experiment_cfg, devices)

        try:
            save_path = get_unique_path(str(directory), f"{base}.bvr")
            drain(self.save_data_queue)
            self.bvr_writer = BvrWriter(
                save_path=save_path,
                data_queue=self.save_data_queue,
                devices=devices,
                device_config={
                    device_id: cfg.to_dict()
                    for device_id, cfg in (self.config.devices or {}).items()
                }
                if self.config is not None
                else {},
                sample_limits=sample_limits,
                logger=self.logger,
            )
            self.bvr_writer.open()
            self.bvr_writer.start()
            self.bvr_writer.resume()
        except Exception as e:
            log_print(self.logger, "error", f"Unable to start recording: {e}")
            self.bvr_writer = None
            return False
        return True

    def _sample_limits(self, experiment_cfg, devices) -> dict:
        """Exact per-device sample budget for a fixed-length run, if one was asked
        for.

        A routine that says 255 s should produce the same number of samples
        every time it runs, so the client sends the duration and the recorder
        stops at ``duration * fs`` rather than wherever the stop command lands.
        """
        duration = experiment_cfg.get("record_duration_s")
        if duration is None:
            return {}
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            return {}
        if duration <= 0:
            return {}

        limits = {}
        for dev in devices:
            fs = dev.get("fs")
            if not fs:
                log_print(
                    self.logger,
                    "warning",
                    f"{dev.get('device_id')} reports no sample rate; its rows "
                    "cannot be held to the routine length",
                )
                continue
            limits[dev["device_id"]] = int(round(duration * float(fs)))

        if limits:
            log_print(
                self.logger,
                "info",
                f"[Save] Recording capped at {duration:g}s: "
                + ", ".join(f"{k} {v} samples" for k, v in limits.items()),
            )
        return limits

    def _close_recording(self):
        """Stop the recorder and finalize the file, if one is open."""
        writer, self.bvr_writer = self.bvr_writer, None
        if writer is None:
            return
        try:
            writer.stop()
            writer.join(timeout=RECORDING_CLOSE_TIMEOUT_S)
            if writer.is_alive():
                log_print(
                    self.logger,
                    "error",
                    "Recorder did not finish writing within "
                    f"{RECORDING_CLOSE_TIMEOUT_S}s; the file may be incomplete",
                )
        except Exception as e:
            log_print(self.logger, "error", f"Error closing recording: {e}")

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

        failures = []
        for device_id, handler in active_handlers.items():
            try:
                handler.stop_streaming()
            except Exception as e:
                failures.append(f"{device_id}: {str(e) or type(e).__name__}")

        self._close_recording()

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

    def _mark_event(self, payload):
        """Record an annotation against the running recording."""
        text = (payload or {}).get("text", "")
        writer = self.bvr_writer
        if writer is None:
            send_response(
                self.client_control_conn,
                Response.ERROR,
                params={"message": "No recording is active"},
                logger=self.logger,
            )
            return

        entry = writer.record_annotation(text)
        log_print(self.logger, "info", f"Marked event: {text}")
        send_response(
            self.client_control_conn,
            Response.SUCCESS,
            params={"annotation": entry},
            logger=self.logger,
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

        writer = self.bvr_writer
        if writer is not None:
            for param, value in config.items():
                with contextlib.suppress(Exception):
                    writer.record_change(device_id, param, value)

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
        """Rebuild the advertised source list from every live device handler."""
        sources = set()
        for device_id, handler in self.device_group_handlers.items():
            if handler is None:
                continue
            try:
                sources.update(handler.get_data_sources())
            except Exception as e:
                log_print(
                    self.logger,
                    "error",
                    f"{device_id} could not report its data sources ({e}); "
                    "its channels will be missing from this session.",
                )
        self.data_sources = sources
        return self.data_sources

    def _run_dpic_balance(self, payload):
        """Start a balance and answer at once; the result arrives by polling."""
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

        with self._dpic_lock:
            existing = self._dpic_states.get(device_id)
            if existing is not None and existing["pending"]:
                send_response(
                    self.client_control_conn,
                    Response.ERROR,
                    params={
                        "message": f"A DPIC balance is already running on {device_id}"
                    },
                    logger=self.logger,
                )
                return
            self._dpic_states[device_id] = {
                "pending": True,
                "ok": None,
                "message": f"DPIC balance running on {device_id}",
                "results": [],
                "device_id": device_id,
                "progress": None,
            }
            self._dpic_last_device = device_id

        send_response(
            self.client_control_conn,
            Response.SUCCESS,
            params={
                "pending": True,
                "message": f"DPIC balance started on {device_id}",
                "results": [],
            },
            logger=self.logger,
        )

        thread = Thread(
            target=self._dpic_balance_work,
            args=(device_id, handler),
            name=f"dpic-balance-{device_id}",
            daemon=True,
        )
        with self._dpic_lock:
            self._dpic_threads[device_id] = thread
        thread.start()

    def _dpic_balance_work(self, device_id, handler):
        """Drive one balance to completion and record its outcome for polling."""
        try:
            response = handler.run_dpic_balance()
            ok = response.get("type") in (
                Response.SUCCESS,
                Response.SUCCESS.name,
                Response.SUCCESS.value,
            )
            message = response.get("message") or (
                "DPIC balance complete" if ok else "DPIC balance failed"
            )
            results = response.get("result", []) or []
        except Exception as e:
            ok, message, results = False, str(e), []

        log_print(
            self.logger,
            "info" if ok else "error",
            f"[DPIC] {device_id}: {message}",
        )
        with self._dpic_lock:
            previous = self._dpic_states.get(device_id) or {}
            self._dpic_states[device_id] = {
                "pending": False,
                "ok": ok,
                "message": message,
                "results": results,
                "device_id": device_id,
                "progress": previous.get("progress"),
            }

    def _shutdown_devices(self):
        """Stop every backend, finalize the recording and reap the subprocesses."""
        handlers = self._active_device_handlers()

        with self._streaming_lock:
            was_streaming = self._streaming_active
            self._streaming_active = False

        for device_id, handler in handlers.items():
            if not was_streaming:
                break
            try:
                handler.stop_streaming()
            except Exception as e:
                log_print(self.logger, "debug", f"{device_id} stop failed: {e}")

        self._close_recording()

        for device_id, handler in handlers.items():
            try:
                handler.shutdown()
            except Exception as e:
                log_print(self.logger, "debug", f"{device_id} shutdown failed: {e}")

        self.device_group_handlers = {}
        self.data_sources = set()

    def stop(self):
        log_print(self.logger, "debug", "Attempting to shutdown server")

        self.running = False

        self._shutdown_devices()

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
        help=(
            "Shorthand for --allow local: serve this machine and private "
            "addresses only, with no prompt"
        ),
    )
    parser.add_argument(
        "--allow",
        choices=POLICIES,
        default=None,
        help=(
            "Who may connect. 'loopback': this machine only. 'local': this "
            "machine and private addresses. 'ask' (default): this machine "
            "silently, anyone else only if the prompt at this server is "
            "answered yes. 'any': everyone, unasked."
        ),
    )
    parser.add_argument(
        "--trust",
        action="append",
        default=[],
        metavar="HOST",
        help=(
            "Address or hostname admitted without a prompt. Repeatable, and "
            "accepts a comma-separated list."
        ),
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

    trusted = resolve_hostnames(
        entry for value in args.trust for entry in str(value).split(",")
    )

    try:
        server = Server(
            local_only=args.local,
            allow=args.allow,
            trusted=trusted,
            exit_when_idle=args.exit_when_idle,
            control_port=args.control_port,
            data_port=args.data_port,
            logger=logger,
        )
    except AdmissionError as e:
        log_print(logger, "error", str(e))
        return 2

    log_print(
        logger,
        "info",
        f"Connection policy: {server.admission_policy}"
        + (f", trusting {sorted(trusted)}" if trusted else ""),
    )

    def _handle_termination(signum, frame):
        log_print(logger, "info", f"Received signal {signum}. Shutting down server...")
        server.running = False

    for signal_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        handled = getattr(signal, signal_name, None)
        if handled is None:
            continue
        with contextlib.suppress(Exception):
            signal.signal(handled, _handle_termination)

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
