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

import time 
import queue 
import argparse
import signal
import socket 
import logging 
import ipaddress
import contextlib 

from threading import Lock, Thread

import multiprocessing as mp 

from bioview_common import (
    APP_VERSION,
    AuthenticationError,
    CONTROL_PORT,
    Command,
    DATA_PORT,
    DeviceError,
    DeviceStatus,
    MAX_BUFFER_SIZE,
    Response,
    ServerStatus,
    ValidationError,
    generate_challenge,
    get_app_info,
    is_dict_of_dicts,
    log_print,
    parse_and_validate_command,
    parse_and_validate_response,
    recv_message,
    send_command,
    send_datachunk,
    send_response,
    validate_token
)



from bioview_server.device import AVAILABLE_BACKENDS, get_device_handler

SLEEP_DURATION = 0.001 # Confirm CPU load with varying this value


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


class Server:
    def __init__(
        self,
        local_only: bool,
        control_port: int,
        data_port: int,
        logger=None
    ):
        # Keep track of local PC information for client communication
        self.info = get_app_info()
        self.token = 42  # TODO: Load using secrets

        # Network info
        self.control_port = control_port
        self.data_port = data_port

        # Server status
        self.running = False

        # Client session status
        self.client_session_active = False

        # Client handling
        self.local_only = local_only
        self.discovered_clients = {}
        self.connected_client_info = {}

        # Device handling
        self.device_group_states = {}
        self.device_group_handlers = {}
        self.config = None
        self.data_sources = set()  # set(source: DataSource)
        self.discovered_devices_cache = {}
        self._device_op_lock = Lock()
        self._device_op_in_progress = False
        self._device_op_thread = None

        # Sockets
        self.data_socket = None
        self.client_data_conn = None
        self.control_socket = None
        self.client_control_conn = None

        # Threaded workers
        self.cmd_thread = None
        self.data_thread = None

        # Queue for overall logging
        self.response_queue = mp.Queue()

        # Queue for data output
        self.data_queue = mp.Queue()

        # Message logging
        if not logger:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(
                level=logging.DEBUG, format="%(asctime)s %(name)s: (%(levelname)s) %(message)s",
                datefmt='%m/%d %H:%M:%S'
            )
        else:
            self.logger = logger

    def start(self): 
        log_print(self.logger, 'info', 'Starting server')
        
        # Setup sockets 
        self._create_sockets()

        # Mark the server status as running 
        self.running = True

        while self.running:
            try:
                # Wait for client connection.
                try:
                    self.control_socket.settimeout(1.0)
                    control_conn, addr = self.control_socket.accept()
                    log_print(self.logger, "debug", f"Control connection initiated from {addr}")
                except socket.timeout:
                    # Timeout just means no one connected yet. Loop back and check self.running
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
                        sock = control_conn, 
                        response = Response.SUCCESS, 
                        params = self.info,  
                        logger = self.logger
                    )
                    control_conn.close() 
                    continue 
                elif cmd_type == Command.CONNECT_SERVER.name: 
                    hostname = payload.get('client_info', {}).get('hostname', None)
                    if hostname: 
                        log_print(self.logger, 'info', f'Incoming connection from: {hostname}')
                    
                    # Send challenge
                    challenge = generate_challenge() 
                    send_response(
                        sock = control_conn,
                        response = Response.SERVER_CHALLENGE, 
                        params = {
                            "challenge": challenge,  
                            "timestamp": time.time() 
                        },
                        logger = self.logger 
                    ),
                    
                    challenge_response = recv_message(control_conn, self.logger)
                    client_cmd, client_payload = parse_and_validate_command(challenge_response) 

                    if client_cmd != Command.AUTHENTICATE_CLIENT.name: 
                        # This is an invalid connection attemp. We should ideally log it: TODO
                        control_conn.close() 
                        continue 
                    
                    auth_token = client_payload.get('token', None)
                    if auth_token and validate_token(challenge, auth_token): 
                        send_response(
                            sock = control_conn, 
                            response = Response.AUTHENTICATION_SUCCESS, 
                            params = {
                                "server_info": self.info,
                                "timestamp": time.time() 
                            }, 
                            logger = self.logger
                        )              
                    else:
                        control_conn.close()
                        continue 


                    # Store connected client info
                    self.connected_client = {
                       "ip": payload.get("ip", ""),
                        "hostname": payload.get("hostname", ""),
                        "name": payload.get("name", ""),
                        "version": payload.get("version", ""),
                    }
                else: 
                    # Invalid command, just keep searching
                    control_conn.close() 
                    continue 

                '''
                Since we are here only when the client has been successfully authenticated,
                it makes sense to initiate the data connection. It is also useful to do it 
                at this stage since if the data connection cannot be made, the program is 
                pretty much useless and we should just try to restart the client 
                '''
                try:
                    data_conn, _ = self.data_socket.accept()
                    log_print(self.logger, "debug", "Data connection accepted.")
                except socket.timeout: 
                    log_print(self.logger, "error", "Client failed to connect data socket in time.")
                    control_conn.close() 
                    continue

                # Client is fully connected. Start workers.
                self.handle_client_session(control_conn, data_conn)
                
                # Cleanup: Session ended, ready for next client
                log_print(self.logger, "debug", "Client session ended. Cleaning up...")
                self.close_client_connections()
                
            except Exception as e:
                log_print(self.logger, "error", f"Error in main loop: {e}")
                self.close_client_connections()

    def _create_sockets(self):
        '''
        Since the client can shutdown at any time or have an error, we want to ensure 
        that the server only binds to sockets at launch and closes them when the server 
        shuts down. 
        '''
        # Create control socket 
        try: 
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.control_socket.bind(("0.0.0.0", self.control_port))
            # A LAN discovery scan opens many short-lived probe connections at
            # once; a generous backlog keeps them from being refused/reset.
            self.control_socket.listen(socket.SOMAXCONN)
            self.control_socket.settimeout(1) # Make sure that accept is non-blocking
            log_print(self.logger, 'debug', 'Control socket created')
        except OSError as e:
            # A port already in use almost always means another BioView server is
            # already running; surface it so the caller can exit cleanly rather
            # than spin on an unbound socket.
            log_print(self.logger, 'error', f'Unable to create control socket: {e}')
            raise

        # Create data socket
        try: 
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_socket.bind(("0.0.0.0", self.data_port))
            self.data_socket.listen(8)
            self.data_socket.settimeout(5) 
            log_print(self.logger, 'debug', 'Data socket connected')
        except OSError as e:
            log_print(self.logger, 'error', f'Unable to create data socket: {e}')
            raise

    def handle_client_session(self, control_conn, data_conn): 
        self.client_control_conn = control_conn 
        self.client_data_conn = data_conn
        
        self.cmd_thread = Thread(
            target = self._command_handler, 
            daemon = True 
        )
        self.data_thread = Thread(
            target = self._data_handler, 
            daemon = True 
        )

        # Mark session started
        self.client_session_active = True 
        
        self.cmd_thread.start() 
        self.data_thread.start()

        # Wait for the command thread to end
        self.cmd_thread.join() 
        self.client_session_active = False 

        # Force close data connection if it doesn't close normally
        with contextlib.suppress(Exception):
            self.client_data_conn.shutdown(socket.SHUT_RDWR)
        
        self.data_thread.join() 
    
    def close_client_connections(self): 
        log_print(self.logger, 'debug', 'Closing client conection')

        if self.client_control_conn: 
            with contextlib.suppress(Exception):
                self.client_control_conn.close() 
        
        if self.client_data_conn: 
            with contextlib.suppress(Exception):
                self.client_data_conn.close()

        self.client_control_conn = None
        self.client_data_conn = None

        # Reset status
        self.client_session_active = False

    def _data_handler(self):
        while self.client_session_active:
            try:
                # Use a real (short) timeout so the loop periodically re-checks
                # client_session_active and the thread can exit cleanly when the
                # session ends or while streaming is paused (no data queued).
                buff = self.data_queue.get(timeout=1.0)

                try: 
                    # Backends push {'data': ndarray, 'sources': [source dicts]}.
                    # The source list is forwarded as chunk metadata so the client
                    # can route each row to the correct plot/save column.
                    if isinstance(buff, dict) and "data" in buff:
                        send_datachunk(
                            self.client_data_conn,
                            buff["data"],
                            meta={"sources": buff.get("sources")},
                        )
                    else:
                        send_datachunk(self.client_data_conn, buff)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    log_print(self.logger, 'error', 'Client disconnected during data transmission.')
                    self.client_session_active = False  # Signal other threads to stop
                    break
            except queue.Empty:
                continue  # continue execution if no data arrived
            except Exception as e:
                log_print(self.logger, 'error', f'Unexpected data handler error: {e}')
                break

    def _command_handler(self):
        while self.client_session_active:
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
                
                if not data: break  # Control connection is closed

                # Parse received command and appropriately call background function 
                cmd_type, payload = parse_and_validate_command(data)
                log_print(self.logger, 'debug', f'Received {cmd_type} with {payload}')

                match cmd_type:
                    case Command.DISCONNECT_SERVER.name: 
                        break # A break is enough to close this thread and the client connection 

                    # Device commands 
                    case Command.DISCOVER_DEVICES.name: 
                        self._start_discover_devices_async(payload)
                    case Command.INITIALIZE_DEVICES.name: 
                        self._start_initialize_devices_async(payload)
                    case Command.GET_DEVICE_STATUS.name:
                        self._handle_get_device_status()
                    case Command.DISCONNECT_DEVICES.name: 
                        self._disconnect_devices()
                        
                    # Streaming 
                    case Command.START_STREAMING.name:
                        # Specify streaming parameters, typically pertaining to saving/display 
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
        try:
            if not isinstance(address, (list, tuple)):
                # Python socket will provide a tuple (ip, port)
                return False 
            

            return ipaddress.ip_address(address[0]).is_private
        except ipaddress.AddressValueError:
            log_print(self.logger, 'error', f"{address} is not a valid IP address")
            return False

    # Device command handling callbacks

    def _config_from_payload(self, payload):
        from bioview_common import Configuration

        return Configuration.from_dict(payload.get("device_groups", payload))

    def _connecting_states_for_config(self, config):
        return {
            device_id: DeviceStatus.CONNECTING.value
            for device_id in config.devices
        }

    def _handle_get_device_status(self):
        send_response(
            sock=self.client_control_conn,
            response=Response.SUCCESS,
            params={
                "pending": self._device_op_in_progress,
                "device_status": self.device_group_states,
                "data_sources": [
                    src.to_dict() for src in self.data_sources
                ],
            },
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
        for backend_type, backend in AVAILABLE_BACKENDS.items(): 
            try:
                found = backend.discover_devices()
                if isinstance(found, dict):
                    discovered_names.update(found.keys())
                    self.discovered_devices_cache.update(found)
                elif isinstance(found, list):
                    for entry in found:
                        if isinstance(entry, dict):
                            name = entry.get("name", "")
                            discovered_names.add(name)
                            if name:
                                self.discovered_devices_cache[name] = entry
                        else:
                            discovered_names.add(str(entry))
            except Exception as e: 
                msg = f"Device discovery failed for devices of type {backend_type} with error: {e}"
                log_print(self.logger, "warning", msg)
        
        discovered_names.discard("")
        log_print(self.logger, "debug", f"Found {sorted(discovered_names)}")
        
        if not self.config:
            self.config = self._config_from_payload(payload)

        from bioview_common.datatypes.devices import DeviceType

        self.device_group_states = {}

        for device_id, device_cfg in self.config.devices.items():
            device_type = device_cfg.get_param("device_type")
            if device_type == DeviceType.DUMMY.value:
                self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                continue

            if device_type == DeviceType.BIOPAC.value:
                hardware = device_cfg.get_param("hardware") or {}
                hw_names = set(hardware.keys()) if isinstance(hardware, dict) else set()
                biopac_discovered = {
                    name
                    for name, info in self.discovered_devices_cache.items()
                    if isinstance(info, dict)
                }
                if hw_names and hw_names.issubset(biopac_discovered):
                    self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                elif hw_names & biopac_discovered:
                    self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                elif biopac_discovered and not hw_names:
                    self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                else:
                    self.device_group_states[device_id] = DeviceStatus.UNAVAILABLE.value
                continue

            if device_id in discovered_names:
                self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
                continue

            hardware = device_cfg.get_param("hardware") or {}
            hw_names = set(hardware.keys()) if isinstance(hardware, dict) else set()
            if hw_names and hw_names.issubset(discovered_names):
                self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
            elif hw_names & discovered_names:
                self.device_group_states[device_id] = DeviceStatus.AVAILABLE.value
            else:
                self.device_group_states[device_id] = DeviceStatus.UNAVAILABLE.value

        log_print(self.logger, "info", "Device discovery completed successfully")

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
                msg = f"Unable to initialize device: {device_id}. Error: {e}"
                log_print(self.logger, "error", msg)
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
            log_print(self.logger, 'warning', msg)
            send_response(self.client_control_conn, Response.SUCCESS, params={"message": msg}, logger = self.logger)
            return
        
        try: 
            for handler in active_handlers.values():
                handler.disconnect()

            msg = "Devices disconnected successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.client_control_conn, Response.SUCCESS, params={"message": msg}, logger = self.logger)
        except Exception as e:
            msg = f"Failed to disconnect devices: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.client_control_conn, Response.ERROR, params={"message": msg}, logger = self.logger)
    
    # Handle streaming 
    def _start_streaming(self, payload): 
        active_handlers = self._active_device_handlers()
        if not active_handlers:
            msg = "Server has no initialized devices"
            log_print(self.logger, 'error', msg)
            send_response(self.client_control_conn, Response.ERROR, params={"message": msg}, logger = self.logger)
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

        self._sync_device_params_from_payload(payload)

        # Ask all backends to start
        try:
            log_print(self.logger, 'info', "Attempting to start data streaming")

            # Start your existing receive/transmit workers
            for handler in active_handlers.values():
                handler.start_streaming(stream_cfg)

            msg = "Data streaming started successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.client_control_conn, Response.SUCCESS, params={"message": msg}, logger = self.logger)
        except Exception as e:
            msg = f"Failed to start streaming: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.client_control_conn, Response.ERROR, params={"message": msg}, logger = self.logger)

    def _stop_streaming(self):
        active_handlers = self._active_device_handlers()
        if not active_handlers:
            msg = "Server has no initialized devices"
            log_print(self.logger, 'warning', msg)
            send_response(self.client_control_conn, Response.SUCCESS, params={"message": msg}, logger = self.logger)
            return

        try:
            log_print(self.logger, 'info', "Attempting to stop data streaming")

            for handler in active_handlers.values():
                handler.stop_streaming()

            msg = "Data streaming stopped successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.client_control_conn, Response.SUCCESS, params={"message": msg})
        except Exception as e:
            msg = f"Failed to stop streaming: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.client_control_conn, Response.ERROR, params={"message": msg})

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
            send_response(self.client_control_conn, Response.ERROR, params={"message": "Invalid payload"}, logger=self.logger)
            return

        log_print(self.logger, "info", f"Updating parameter for device {device_id}")

        # Update internal config
        if self.config:
            for param, value in config.items():
                self.config.update_device_param(device_id, param, value)

        # Find the handler managing this device. device_id is typically the group_id.
        handler = self.device_group_handlers.get(device_id)

        if handler is None:
            send_response(self.client_control_conn, Response.ERROR, params={"message": "Device handler not found"}, logger=self.logger)
            return

        try:
            handler.queue_param_update(**config)
            send_response(self.client_control_conn, Response.SUCCESS, params={"message": "Parameter updated"}, logger=self.logger)
        except Exception as e:
            send_response(self.client_control_conn, Response.ERROR, params={"message": str(e)}, logger=self.logger)

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
            if response.get("type") in (Response.SUCCESS, Response.SUCCESS.name, Response.SUCCESS.value):
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
        log_print(self.logger, 'debug', "Attempting to shutdown server")

        # Close any active client connections 
        self.close_client_connections()
        
        # Close sockets 
        if self.control_socket: 
            self.control_socket.close() 
        self.control_socket = None 

        if self.data_socket:
            self.data_socket.close() 
        self.data_socket = None 

        # Send signal to stop server 
        self.running = False 

        log_print(self.logger, 'debug', "Server shut down successfully")

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Launch BioView Backend Server")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Flag to make server restricteed only to local clients"
    )
    parser.add_argument(
        "--control-port",
        type=int,
        help=f"Port number to use for control connections. Default: {CONTROL_PORT}",
        required=False,
        default=CONTROL_PORT
    )
    parser.add_argument(
        "--data-port",
        type=int,
        help=f"Port number to use for data connections. Default: {DATA_PORT}",
        required=False,
        default=DATA_PORT
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(name)s: (%(levelname)s) %(message)s",
        datefmt='%m/%d %H:%M:%S'
    )
    log_print(logger, 'info', f"BioView Device Server, Version: {APP_VERSION}")

    args = parser.parse_args(argv)

    server = Server(
        local_only = args.local,
        control_port = args.control_port,
        data_port = args.data_port,
        logger = logger
    )

    # Stop cleanly when the launcher (or the OS) asks us to terminate, so sockets
    # are released promptly when the GUI that spawned us closes.
    def _handle_termination(signum, frame):
        log_print(logger, 'info', f"Received signal {signum}. Shutting down server...")
        server.running = False

    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, _handle_termination)

    exit_code = 0
    try:
        server.start()
    except KeyboardInterrupt:
        log_print(logger, 'warning', "Keyboard interrupt received. Shutting down server...")
    except OSError as e:
        # Most likely the control/data port is already bound by another server.
        log_print(logger, 'error', f"Unable to bind server sockets ({e}). Exiting...")
        exit_code = 1
    except Exception:
        log_print(logger, 'error', "Server error. Shutting down server...")
        exit_code = 1
    finally:
        try:
            server.stop()
        except Exception:
            log_print(logger, 'error', "Unable to shut down server. Exiting...")

    return exit_code


if __name__ == '__main__':
    import sys

    sys.exit(main())
