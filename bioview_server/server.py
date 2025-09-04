""" BioView Server

The server exposes a flexible way for connecting to devices by forwarding
client commands to the appropriate handlers.

Note that, as of now, the server assumes a single connected client since
there is ambiguity regarding control from multiple clients. Specifically,
it is not yet clear whether we require a main client and other observer
clients or whether every client needs to be provided the same level of
access or whether each device should be considered as a different client
connection. Subsequent experimentation and discussion will be pertinent
for expanding functionality to handle the case for multiple clients.
"""
import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
import secrets
import socket
import time
import traceback
from threading import Lock, Thread
from typing import Dict, List

from bioview_common import (
    APP_VERSION,
    AUTH_TIMEOUT,
    CONTROL_PORT,
    DATA_PORT,
    MAX_BUFFER_SIZE,
    RESPONSE_TIMEOUT,
    AuthenticationError,
    Command,
    Configuration,
    Response,
    ServerStatus,
    get_app_info,
    get_ip,
)

from bioview_server.datatypes import Backend
from bioview_server.device import AVAILABLE_BACKENDS


try:
    from bioview_client.utils.zeroconf_discovery import register_service
except Exception:
    register_service = None
logger = logging.getLogger(__name__)
logger.info("Available Backends: %s", list(AVAILABLE_BACKENDS.keys()))


class Server:
    def __init__(
        self,
        control_port: int = CONTROL_PORT,
        data_port: int = DATA_PORT,
        # By default, run server using local-only mode for safety.
        discoverable: bool = False,
        auth_timeout: int = AUTH_TIMEOUT,
        resp_timeout: int = RESPONSE_TIMEOUT,
    ):
        # Server network information
        self.address = get_ip()
        # Fetch app info and sanitize to JSON-serializable primitives only
        raw_info = get_app_info() or {}
        sanitized_info = {}
        for k in ("hostname", "app_name", "app_version"):
            v = raw_info.get(k)
            try:
                # allow simple serializable values
                json.dumps({k: v})
                sanitized_info[k] = v
            except Exception:
                sanitized_info[k] = str(v) if v is not None else ""
        # include network address for convenience
        sanitized_info["address"] = self.address
        self.server_info = sanitized_info

        # Server connection preference
        self.auth_timeout = auth_timeout
        self.resp_timeout = resp_timeout
        self.discoverable = discoverable
        # Ports
        self.data_port = data_port
        self.control_port = control_port

        # Sockets
        self.data_socket = None
        self.control_socket = None

        # Clients
        self.data_clients = []  # List of connected data clients
        self.data_lock = Lock()

        # Server state
        self.status = ServerStatus.DEFAULT

        # Device state
        self.discovered_devices = {}
        self.backends: List[Backend] = []
        self.display_data_queue = mp.Queue()
        # Application token used for challenge-response authentication.
        # For now use a deterministic value (APP_VERSION) so tests can compute
        # a stable expected token; in production this should be a secret.
        try:
            self.app_token = str(APP_VERSION)
        except Exception:
            # fallback to a random token if APP_VERSION isn't available
            self.app_token = secrets.token_hex(16)

    def start(self):
        logger.info("Starting server on %s", self.server_info.get("hostname"))
        try:
            # Create and bind control/data sockets
            self._setup_sockets()
            # Optionally start discovery responder
            self._maybe_start_discovery_responder()
        except Exception:
            logger.exception("Error occurred while starting server")

        # Once clients have started, start listening
        self.running = True
        try:
            # Start server threads
            control_thread = Thread(target=self.handle_control_connection, daemon=True)
            data_thread = Thread(target=self.handle_data_connection, daemon=True)

            control_thread.start()
            data_thread.start()

            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except Exception:
            logger.exception("Server error")
        finally:
            self.stop()

    def _setup_sockets(self):
        # Setup control socket
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.bind(("0.0.0.0", self.control_port))
        self.control_socket.listen(5)
        logger.info("Control server listening on %s:%s", self.address, self.control_port)

        # Setup data socket
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_socket.bind(("0.0.0.0", self.data_port))
        self.data_socket.listen(10)
        logger.info("Data server listening on %s:%s", self.address, self.data_port)

    def _maybe_start_discovery_responder(self):
        if not self.discoverable:
            return
        # Register service if available and start responder thread.
        self._register_service_if_available()
        Thread(target=self._discovery_responder, daemon=True).start()

    def _register_service_if_available(self):
        with contextlib.suppress(Exception):
            if not register_service:
                return
            props = self._discovery_props()
            register_service(
                self.server_info.get("hostname", "bioview"),
                port=self.control_port,
                properties=props,
            )

    # --- Simple challenge/response helpers ---
    def _generate_challenge(self) -> str:
        """Return a short random challenge string."""
        return secrets.token_hex(16)

    def _discovery_responder(self):
        """Actual discovery responder loop extracted to reduce complexity."""
        try:
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            udp.bind(("0.0.0.0", self.control_port))
            while self.running:
                try:
                    data, addr = udp.recvfrom(4096)
                    try:
                        msg = json.loads(data.decode("utf-8"))
                    except Exception:
                        continue
                    if msg.get("type") == "DISCOVER_REQUEST":
                        server_info = {}
                        raw = self.server_info or {}
                        for k in ("hostname", "app_name", "app_version"):
                            v = raw.get(k)
                            try:
                                json.dumps({k: v})
                                server_info[k] = v
                            except Exception:
                                server_info[k] = str(v)

                        resp = {
                            "type": Response.INFO.value,
                            "payload": {"server_info": server_info},
                        }
                        try:
                            udp.sendto(
                                json.dumps(self._sanitize_for_json(resp)).encode(
                                    "utf-8"
                                ),
                                addr,
                            )
                        except Exception:
                            continue
                except Exception:
                    continue
        finally:
            with contextlib.suppress(Exception):
                udp.close()

    def _discovery_props(self):
        """Build sanitized props for zeroconf registration."""
        props = {}
        try:
            props_raw = {
                "app_name": self.server_info.get("app_name"),
                "app_version": self.server_info.get("app_version"),
            }
            for k, v in props_raw.items():
                try:
                    json.dumps({k: v})
                    props[k] = v
                except Exception:
                    props[k] = str(v)
        except Exception:
            props = {}
        return props

    def _compute_response(self, challenge: str, token: str) -> str:
        """Compute expected response from challenge and shared token.

        Uses SHA-256 over "{challenge}:{token}" to produce a deterministic hex digest.
        """
        try:
            m = hashlib.sha256()
            m.update(f"{challenge}:{token}".encode())
            return m.hexdigest()
        except Exception:
            # fallback to simple concatenation if hashlib unexpectedly fails
            return f"{challenge}:{token}"

    def stop(self):
        logger.info("Stopping server")
        self.running = False

        self.handle_disconnect_from_client()
        logger.info("BioView server stopped")

    # Ensure connection validation for server security
    def _handle_initial_client_message(self, client_socket, address):
        try:
            data = client_socket.recv(MAX_BUFFER_SIZE)
            if not data:
                client_socket.close()
                return

            command = json.loads(data.decode("utf-8"))
            with contextlib.suppress(Exception):
                # Print received raw command to console for debugging
                print(f"[SERVER RECV] {address}: {command}")

            cmd_type = command.get("type")
            payload = command.get("payload", {})

            # Discovery mode handling
            if cmd_type == Command.PING_SERVER.value and self.discoverable:
                response = self.handle_ping()
                with contextlib.suppress(Exception):
                    print(f"[SERVER SEND] {address}: {response}")

                self.send_json_safe(client_socket, response)
                client_socket.close()
                return

            # Full authentication path for control commands
            if cmd_type == Command.CONNECT_SERVER.value:
                self.client_info = self.validate_and_authenticate_client(
                    client_socket, address, payload
                )
                Thread(
                    target=self.handle_commands, args=(client_socket,), daemon=True
                ).start()
                return

            # If we reach here, it's an invalid attempt
            error_response = {
                "type": Response.ERROR.value,
                "payload": {"message": f"Command '{cmd_type}' requires authentication"},
            }
            with contextlib.suppress(Exception):
                print(f"[SERVER SEND] {address}: {error_response}")

            self.send_json_safe(client_socket, error_response)
            client_socket.close()

        except json.JSONDecodeError:
            error = {
                "type": Response.ERROR.value,
                "payload": {"message": "Invalid JSON"},
            }
            with contextlib.suppress(Exception):
                print(f"[SERVER RECV][INVALID_JSON] {address}")
            with contextlib.suppress(Exception):
                self.send_json_safe(client_socket, error)
            client_socket.close()
        except Exception:
            logger.exception("Initial message handling error")
            with contextlib.suppress(Exception):
                client_socket.close()

    def validate_and_authenticate_client(
        self, client_socket, client_address, initial_payload
    ):
        """
        Authenticate after receiving CONNECT_SERVER.
        initial_payload is already parsed from first message.
        """
        client_ip = client_address[0]
        try:
            client_socket.settimeout(self.auth_timeout)

            client_info = {
                "hostname": initial_payload.get("hostname"),
                "app_name": initial_payload.get("app_name"),
                # accept either 'app_version' or legacy 'version'
                "app_version": initial_payload.get(
                    "app_version", initial_payload.get("version")
                ),
            }
            logger.info(
                "Connection request received from %s", client_info.get("hostname")
            )

            # Validate timestamp and start challenge/response flow using small helpers
            self._auth_validate_timestamp(initial_payload)

            challenge = self._generate_challenge()
            self._auth_send_challenge(client_socket, client_ip, challenge)

            parsed = self._auth_recv_response(client_socket)
            self._auth_validate_command_type(parsed)

            client_response_payload = parsed.get("payload", {})

            expected_token = self._compute_response(challenge, self.app_token)
            # accept either 'token' (client) or 'auth_token' (legacy)
            received_token = (
                client_response_payload.get("token")
                or client_response_payload.get("auth_token")
                or ""
            )

            if not secrets.compare_digest(expected_token, received_token):
                fail_msg = {
                    "type": Response.AUTHENTICATION_FAILURE.value,
                    "payload": {"message": "Invalid authentication token"},
                }
                self.send_json_safe(client_socket, fail_msg)
                raise AuthenticationError("Invalid authentication token") from None

            # Step 4: Send success confirmation
            self._auth_send_success(client_socket)
            logger.info("Successfully authenticated %s", client_info.get("hostname"))
            return client_info

        except socket.timeout:
            raise AuthenticationError("Authentication timeout") from None
        except Exception as e:
            logger.exception("Authentication error")
            raise AuthenticationError(f"Authentication failed: {str(e)}") from e

    # --- Authentication helpers (small, focused helpers reduce complexity) ---
    def _auth_validate_timestamp(self, initial_payload):
        client_timestamp = initial_payload.get("timestamp", 0)
        if abs(time.time() - client_timestamp) > AUTH_TIMEOUT * 60:
            raise AuthenticationError(
                "Client timestamp outside acceptable window"
            ) from None

    def _auth_send_challenge(self, client_socket, client_ip, challenge):
        challenge_msg = {
            "type": Response.SERVER_CHALLENGE.value,
            "payload": {"challenge": challenge, "timestamp": time.time()},
        }
        with contextlib.suppress(Exception):
            print(f"[SERVER SEND] {client_ip}: {challenge_msg}")
        self.send_json_safe(client_socket, challenge_msg)

    def _auth_recv_response(self, client_socket):
        # Receive and parse client auth response; raise on invalid JSON
        client_response_data = client_socket.recv(4096).decode("utf-8")
        with contextlib.suppress(Exception):
            print(f"[SERVER RECV] {client_response_data}")
        try:
            return json.loads(client_response_data)
        except Exception:
            # invalid JSON from client during auth
            try:
                err = {
                    "type": Response.ERROR.value,
                    "payload": {"message": "Invalid JSON in authentication payload"},
                }
                self.send_json_safe(client_socket, err)
            except Exception:
                pass
            raise

    def _auth_validate_command_type(self, parsed):
        if parsed.get("type") != Command.AUTHENTICATE_CLIENT.value:
            msg = f"Unexpected auth command type: {parsed.get('type')}"
            raise AuthenticationError(msg) from None

    def _auth_send_success(self, client_socket):
        # prepare sanitized server_info for response
        sanitized = {}
        raw = self.server_info or {}
        for k, v in raw.items():
            try:
                json.dumps({k: v})
                sanitized[k] = v
            except Exception:
                sanitized[k] = str(v)

        success_msg = {
            "type": Response.AUTHENTICATION_SUCCESS.value,
            "payload": {"server_info": sanitized, "timestamp": time.time()},
        }
        try:
            self.send_json_safe(client_socket, success_msg)
        except Exception:
            logger.exception("Failed to send authentication success message")

    def _generate_challenge(self) -> str:
        """Return a random challenge string for clients to sign."""
        return secrets.token_hex(16)

    def _sanitize_for_json(self, obj):
        """Recursively convert non-JSON-serializable objects to strings.

        Keeps dicts, lists and primitives; everything else becomes str(obj).
        """
        # primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # dict
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        # list/tuple
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(v) for v in obj]
        # fallback: string representation
        try:
            return str(obj)
        except Exception:
            return repr(obj)

    def send_json_safe(self, sock, obj):
        """Sanitize and send a JSON object over the given socket.

        If serialization fails, attempt to send a safe error response. If that
        also fails, close the socket silently.
        """
        try:
            safe = self._sanitize_for_json(obj)
            data = json.dumps(safe).encode("utf-8")
            sock.send(data)
        except Exception:
            logger.exception(
                "Failed to serialize/send JSON response; sending fallback error"
            )
            try:
                fallback = {
                    "type": Response.ERROR.value,
                    "payload": {"message": "Internal server error"},
                }
                sock.send(json.dumps(self._sanitize_for_json(fallback)).encode("utf-8"))
            except Exception:
                with contextlib.suppress(Exception):
                    sock.close()

    def _compute_response(self, challenge: str, token: str) -> str:
        """Compute the expected response token from a challenge and app token.

        Uses a simple SHA-256(challenge + token) hex digest. The client must
        use the same computation to succeed.
        """
        if challenge is None:
            return ""
        data = f"{challenge}:{token}".encode()
        return hashlib.sha256(data).hexdigest()

    def handle_control_connection(self):
        while self.running:
            try:
                client_socket, address = self.control_socket.accept()
                Thread(
                    target=self._handle_initial_client_message,
                    args=(client_socket, address),
                    daemon=True,
                ).start()
            except Exception:
                if self.running:
                    logger.exception("Error accepting control connection")

    def handle_data_connection(self):
        while self.running:
            try:
                client_socket, address = self.data_socket.accept()
                logger.info("Data client connected from %s", address)

                with self.data_lock:
                    self.data_clients.append(client_socket)

                # Handle client disconnect. Bind address into the monitor's default
                # argument to avoid loop-variable closure issues.
                def monitor_client(sock, addr=address):
                    try:
                        while self.running:
                            # Send keepalive
                            sock.send(b"")
                            time.sleep(1)
                    except Exception as e:
                        with self.data_lock:
                            if sock in self.data_clients:
                                self.data_clients.remove(sock)
                        with contextlib.suppress(Exception):
                            sock.close()

                        logger.info(f"Data client {addr} disconnected: {e}")

                Thread(target=monitor_client, args=(client_socket,), daemon=True).start()

            except Exception as e:
                if self.running:
                    logger.exception(f"Error accepting data connection: {e}")

    def handle_commands(self, client_socket):
        # Receives commands from clients and controls device handlers accordingly
        try:
            # Use a short recv timeout so we can check self.running periodically
            with contextlib.suppress(Exception):
                client_socket.settimeout(1.0)
                # Some socket-like objects may not support settimeout

            while self.running:
                data = self._recv_command_data(client_socket)
                if data is None:
                    continue
                if not data:
                    break

                try:
                    command = json.loads(data.decode("utf-8"))
                    # Log received command
                    print(f"[SERVER RECV CMD] {client_socket.getpeername()}: {command}")

                    response = self.process_command(command)
                    # Log response we're about to send
                    print(
                        f"[SERVER SEND RESP] {client_socket.getpeername()}: {response}"
                    )

                    self.send_json_safe(client_socket, response)
                except json.JSONDecodeError as e:
                    error_response = {
                        "type": Response.ERROR.value,
                        "payload": {"message": f"Invalid JSON: {e}"},
                    }
                    self.send_json_safe(client_socket, error_response)
                except Exception:
                    # Any unexpected serialization or processing error should
                    # still result in a valid JSON error response sent back.
                    err = {
                        "type": Response.ERROR.value,
                        "payload": {"message": "Internal processing error"},
                    }

                    with contextlib.suppress(Exception):
                        self.send_json_safe(client_socket, err)

        except Exception:
            logger.exception("Control client error")
        finally:
            with contextlib.suppress(Exception):
                client_socket.close()

    def _recv_command_data(self, client_socket):
        """Receive raw data from command socket with timeout handling.

        Returns:
            bytes | None (if timeout occurred and loop should continue)
        """
        try:
            data = client_socket.recv(MAX_BUFFER_SIZE)
            return data
        except (socket.timeout, TimeoutError):
            # no data received within timeout; loop and check running flag
            return None
        except Exception:
            # propagate other socket errors to outer handler
            raise

    def process_command(self, command):
        # Redirect received command from client to the appropriate callback
        cmd_type = command.get("type")
        payload = command.get("payload", {})
        try:
            return self._dispatch_command(cmd_type, payload)
        except Exception as e:
            return {
                "type": Response.ERROR.value,
                "payload": {"message": f"Command processing error: {e}"},
            }

    def _dispatch_command(self, cmd_type, payload):
        # Use a lookup table to simplify branching
        dispatch_map = {
            Command.PING_SERVER.value: lambda: self.handle_ping(),
            Command.CONNECT_SERVER.value: lambda: self.handle_connect_to_client(payload),
            Command.DISCOVER_DEVICES.value: lambda: self.handle_discover_devices(
                payload
            ),
            Command.DISCONNECT_SERVER.value: (
                lambda: self.handle_disconnect_from_client()
            ),
            Command.CONNECT_DEVICES.value: lambda: self.handle_connect_device(payload),
            Command.GET_DEVICE_STATUS.value: (
                lambda: self.handle_get_device_status(payload)
            ),
            Command.START_STREAMING.value: lambda: self.handle_start_streaming(),
            Command.STOP_STREAMING.value: lambda: self.handle_stop_streaming(),
            Command.UPDATE_RUNNING_PARAMETER.value: (
                lambda: self.handle_update_runing_parameter(payload)
            ),
            Command.UPDATE_DEVICE_FIRMWARE.value: (
                lambda: self.handle_update_device_firmware(payload)
            ),
            Command.DISCONNECT_DEVICES.value: lambda: self.handle_disconnect_device(),
        }

        handler = dispatch_map.get(cmd_type)
        if handler is None:
            return {
                "type": Response.ERROR.value,
                "payload": {"message": f"Unknown command: {cmd_type}"},
            }

        return handler()

    # Server commands
    def handle_ping(self):
        # Return server status
        return {
            "type": Response.INFO.value,
            "payload": {
                "hostname": self.server_info.get("hostname"),
                "version": self.server_info.get(
                    "app_version", self.server_info.get("version")
                ),
                "status": self.status.value,
            },
        }

    def handle_initialize_common_configuration(self, client_dict):
        """
        Server will only respond to commands if connected to a client.
        This handler validates a client and switches server state to
        be actually useful
        """
        # TODO: Initialize server common configuration
        # TODO: These should not be part of the device handling
        # exp_config = Configuration.from_json(params['exp_config'])
        # save = params.get('save', False)
        pass

    def handle_discover_devices(self, requested: Dict = None):
        """Discover devices and report which requested devices are present.

        If `requested` is None, performs a full discovery and returns the
        discovered devices per backend. If `requested` is provided as a mapping
        of group_id -> Configuration (or dict describing devices), this will
        return a per-group mapping of requested device_id -> found(boolean).
        """
        logger.info("Starting device discovery")

        try:
            # Populate discovered_devices from backends. Import device module at
            # runtime so tests can monkeypatch AVAILABLE_BACKENDS on the module.
            from bioview_server import device as _device

            for backend_type, backend_handler in _device.AVAILABLE_BACKENDS.items():
                # backend_handler.discover_devices() should return a list of dicts
                self.discovered_devices[
                    backend_type
                ] = backend_handler.discover_devices()
        except Exception as e:
            logger.exception("Device discovery failed")
            return {
                "type": Response.ERROR.value,
                "payload": {"message": f"Device discovery failed: {e}"},
            }

        # If no specific request, return full discovery results
        if not requested:
            logger.info("Device discovery completed successfully")
            num_devices = sum(len(v) for v in self.discovered_devices.values())
            return {
                "type": Response.SUCCESS.value,
                "payload": {
                    "message": f"Found {num_devices} devices",
                    "devices": self.discovered_devices,
                },
            }

        # Match requested device groups to discovered devices
        try:
            matched = self._match_requested_to_discovered(requested)
            return {"type": Response.SUCCESS.value, "payload": {"devices": matched}}
        except Exception as e:
            logger.exception("Device matching failed")
            return {
                "type": Response.ERROR.value,
                "payload": {"message": f"Device matching failed: {e}"},
            }

    def _match_requested_to_discovered(self, requested: Dict) -> Dict:
        """Given a requested mapping of group_id -> device configs, return a
        mapping group_id -> {device_id: True/False} indicating presence.

        We assume requested is a dict where top-level keys are group_ids and
        values are dicts mapping device_id -> Configuration-like dict.
        Matching is performed by inspecting discovered devices for the
        backend_type present in the configuration (Configuration.get_param)
        and comparing device_id to discovered 'device_id' fields.
        """
        results = {}

        for group_id, group_conf in requested.items():
            results[group_id] = {}

            # group_conf is expected to be a dict of device_id -> config
            for device_id, conf in (
                group_conf.items() if isinstance(group_conf, dict) else []
            ):
                # extract backend_type from provided configuration
                backend_type = None
                try:
                    # allow Configuration-like objects with get_param
                    backend_type = conf.get_param("backend_type")
                except Exception:
                    # fallback to dict-style
                    backend_type = (
                        conf.get("backend_type") if isinstance(conf, dict) else None
                    )

                found = False
                # If backend_type provided, only check that backend; otherwise
                # check across all backends (fallback)
                backends_to_check = (
                    [backend_type]
                    if backend_type and backend_type in self.discovered_devices
                    else list(self.discovered_devices.keys())
                )
                for b in backends_to_check:
                    for dev in self.discovered_devices.get(b, []):
                        # compare by device_id key on discovered device
                        if str(dev.get("device_id")) == str(device_id):
                            found = True
                            break
                    if found:
                        break

                results[group_id][device_id] = found

        return results

    def handle_disconnect_from_client(self):
        # Disconnect clients from servers
        logger.info("Disconnecting server from clients")

        try:
            # Close sockets
            if self.control_socket:
                self.control_socket.close()
            if self.data_socket:
                self.data_socket.close()

            return {
                "type": Response.SUCCESS.value,
                "payload": {"message": "Server disconnected successfully"},
            }
        except Exception as e:
            return {
                "type": Response.ERROR.value,
                "payload": {"message": f"Server disconnection error: {e}"},
            }
        finally:
            self.control_socket = None
            self.data_socket = None
            self.status = ServerStatus.CLIENT_DISCONNECTED

    # Device commands
    def handle_connect_device(self, configurations: Dict = None):
        """
        Provided params will typically include device specific configurations,
        using which all devices are initialized. Configurations provided by
        the client are considered canonical (vis-a-vis values), regardless of
        any pre-existing configs.
        """
        try:
            # Firstly, initialize a suitable backend handler
            for device_id, device_config_dict in configurations.items():
                # Make device configuration object
                device_configuration = Configuration.from_dict(device_config_dict)

                # Create the backend handler
                backend_type = device_configuration.get_param("backend_type")
                backend_module = AVAILABLE_BACKENDS[backend_type]

                # Each handler will have its own way of parsing provided
                # configuration but must be able to handle the following parameters
                handler = backend_module.get_backend_handler(
                    # Configuration
                    configuration=device_configuration,
                    # Shared queues for data handling
                    display_data_queue=self.display_data_queue,
                    command_queue=self.command_queue,
                    response_queue=self.response_queue,
                )

                # Now, initialize the handler (which will also initialize the device)
                handler.initialize()

                # Lastly, store reference
                self.backends[device_id] = handler

                # Communicate success
                return {
                    "type": Response.SUCCESS.value,
                    "payload": {"message": "Device inited successfully"},
                }

        except Exception as e:
            # Communicate failure
            return {
                "type": Response.ERROR.value,
                "payload": {
                    "message": f"Device initialization failed: {e}",
                    "traceback": traceback.format_exc(),
                },
            }

    def handle_get_device_status(self, param_dict):
        device_id = param_dict.get("device_id", None)
        if not device_id:
            return self.backends[device_id].get_device_status()

    def handle_start_streaming(self):
        # Order devices to start streaming
        if len(self.backends) == 0:
            return {
                "type": Response.WARNING.value,
                "payload": {"message": "No devices connected."},
            }

        try:
            logger.info("Starting data streaming")

            # Start your existing receive/transmit workers
            for backend in self.backends.values():
                backend.start_streaming()

            self.status = ServerStatus.STREAMING

            logger.info("Data streaming started")
            return {
                "type": Response.SUCCESS.value,
                "payload": {"message": "Data streaming started"},
            }
        except Exception as e:
            return {
                "type": Response.ERROR.value,
                "payload": {"message": f"Failed to start streaming: {e}"},
            }

    def handle_stop_streaming(self):
        if self.status != ServerStatus.STREAMING:
            return {
                "type": Response.WARNING.value,
                "payload": {"message": "Server is not currently streaming"},
            }
        try:
            logger.info("Stopping data streaming")
            for backend in self.backends.values():
                backend.stop_streaming()

            self.status = ServerStatus.DEVICES_CONNECTED

            return {
                "type": Response.SUCCESS.value,
                "payload": {"message": "Data streaming stopped"},
            }
        except Exception as e:
            return {
                "type": Response.ERROR.value,
                "payload": {"message": f"Failed to stop streaming: {e}"},
            }

    def handle_update_runing_parameter(self, param_dict):
        # TODO: This simply looks like adding the payload to command queue
        pass

    # We are not sure if this is a thing to do currently
    def handle_update_device_firmware(self):
        raise NotImplementedError

    def handle_disconnect_device(self):
        # Disconnect all devices
        try:
            for backend in self.backends.values():
                backend.disconnect()

            logger.info("Devices disconnected")

            self.status = ServerStatus.DEVICES_DISCONNECTED

            return {
                "type": Response.SUCCESS.value,
                "payload": {"message": "Devices successfully disconnected"},
            }
        except Exception as e:
            return {
                "type": Response.ERROR.value,
                "payload": {"message": f"Disconnect error: {e}"},
            }

    def handle_update_device_config(self, group_id, param, value, device_id=""):
        if group_id not in self.backends:
            return {
                "type": Response.ERROR.value,
                "message": f"Invalid device {group_id} specified for modification.",
            }

        backend = self.backends[group_id]  # type(BACKEND)

        backend.command_queue.put(
            {
                "param": param,
                "value": value,
                # For device groups with multiple devices
                "device_id": device_id,
            }
        )


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info("BioView Device Server, Version: %s", APP_VERSION)

    server = Server(discoverable=True)  # TODO: Control via argparse
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception:
        logger.exception("Server error")
    finally:
        try:
            server.stop()
        except Exception:
            logger.exception("Error while stopping server")
