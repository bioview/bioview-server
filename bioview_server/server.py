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
import socket 
import logging 
import ipaddress
import contextlib 

from threading import Thread

import multiprocessing as mp 

from bioview_common import (    
    AuthenticationError,
    ValidationError, 
    DeviceError, 
    DeviceStatus, 
    get_app_info, 
    MAX_BUFFER_SIZE,
    CONTROL_PORT, DATA_PORT,
    ServerStatus, 
    Response, Command,
    APP_VERSION,
    log_print,
    is_dict_of_dicts
)

from bioview_server.utils import (
    send_response, 
    send_datachunk,
    generate_challenge, 
    validate_token, 
    parse_and_validate_command
)

from bioview_server.device import AVAILABLE_BACKENDS, get_device_group_handler

SLEEP_DURATION = 0.001 # Confirm CPU load with varying this value

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
        self.data_sources = set()  # set(source: DataSource)

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
        log_print(self.logger, 'info' 'Starting server')
        
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
                auth_data = control_conn.recv(MAX_BUFFER_SIZE)
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
                    
                    challenge_response = control_conn.recv(MAX_BUFFER_SIZE)
                    client_cmd, client_payload = parse_and_validate_command(challenge_response) 

                    if client_cmd is not Command.AUTHENTICATE_CLIENT.name: 
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
                    log_print(self.logger, "error" "Client failed to connect data socket in time.")
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
            self.control_socket.listen(1)
            self.control_socket.settimeout(1) # Make sure that accept is non-blocking
            log_print(self.logger, 'debug', 'Control socket created')
        except Exception as e: 
            log_print(self.logger, 'error', f'Unable to create control socket: {e}')

        # Create data socket
        try: 
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_socket.bind(("0.0.0.0", self.data_port))
            self.data_socket.listen(1)
            self.data_socket.settimeout(5) 
            log_print(self.logger, 'debug', 'Data socket connected')
        except Exception as e: 
            log_print(self.logger, 'error', f'Unable to create data socket: {e}') 

    def handle_client_session(self, control_conn, data_conn): 
        self.client_control_conn = control_conn 
        self.client_data_conn = data_conn
        
        self.cmd_thread = Thread(
            target = self._command_handler, 
            daemon = True 
        )
        self.data_thread = Thread(
            thread = self._data_handler, 
            daemon = True 
        )

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
                buff = self.data_queue.get(0.5)  # add a little delay to lower CPU usage

                try: 
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
                    data = self.client_control_conn.recv(MAX_BUFFER_SIZE)
                except (OSError, ConnectionResetError) as e: 
                    log_print(self.logger, "error", f"Connection reset by host: {e}")
                    break
                except socket.timeout: 
                    continue  # ensure timeouts do not kill this thread 

                if not data: break  # Control connection is closed

                # Parse received command and appropriately call background function 
                cmd_type, payload = parse_and_validate_command(data)
                log_print(self.logger, 'debug', f'Received {cmd_type} with {payload}')

                match cmd_type:
                    case Command.DISCONNECT_SERVER.name: 
                        break # A break is enough to close this thread and the client connection 

                    # Device commands 
                    case Command.DISCOVER_DEVICES.name: 
                        # Pass device configuration here 
                        self._discover_devices(payload)

                        # Send response here since we don't want to send this response during init
                        if self.device_group_states != {}: 
                            log_print(self.logger, 'debug', 'Successfully found devices')
                            send_response(
                                self.client_control_conn, 
                                Response.SUCCESS, 
                                params={
                                    "device_status": self.device_group_states,
                                    "data_sources": [src.to_dict() for src in self.data_sources]
                                },
                                logger = self.logger
                            )
                        else:
                            log_print(self.logger, 'debug', 'Failed to find devices')
                            send_response(
                                self.client_control_conn, 
                                Response.ERROR, 
                                params={"message": "Specified devices not found: \
                                    Are you sure they are plugged in and drivers are installed?"},
                                logger = self.logger
                            )
                    case Command.INITIALIZE_DEVICES.name: 
                        self._initialize_devices(payload)
                    case Command.DISCONNECT_DEVICES.name: 
                        self._disconnect_devices()
                        
                    # Streaming 
                    case Command.START_STREAMING.name:
                        # Specify streaming parameters, typically pertaining to saving/display 
                        self._start_streaming(payload)
                    case Command.STOP_STREAMING.name:
                        self._stop_streaming()

            except ValidationError as e:
                log_print(self.logger, "debug", f"Invalid command sent: {e}")
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
    def _discover_devices(self, payload): 
        log_print(self.logger, "info", "Discovering connected devices")
        
        # Refresh list of discovered devices
        self.discovered_devices = []
        for backend_type, backend in AVAILABLE_BACKENDS.items(): 
            try:
                self.discovered_devices.extend(backend.discover_devices())
            except Exception as e: 
                msg = f'Device discovery failed for devices of type {backend_type} with error: {e}'
                log_print(self.logger, 'warning', msg)
        
        log_print(self.logger, 'debug', f'Found {self.discovered_devices}')
        
        # Extract provided device configuration
        device_groups = payload.get('device_groups', {})
        if device_groups is {} or not is_dict_of_dicts(device_groups):
            return None

        # Check if all devices in the device groups are present 
        self.device_group_states = {} # Refresh

        for group_id, group_dict in device_groups.items(): 
            self.device_group_states[group_id] = {}
            for device_id in group_dict.keys():
                # Avoid populating with metadata
                if device_id == 'metadata':
                    continue

                if device_id in self.discovered_devices:
                    self.device_group_states[group_id][device_id] = DeviceStatus.AVAILABLE.value
                else: 
                    self.device_group_states[group_id][device_id] = DeviceStatus.UNAVAILABLE.value

        log_print(self.logger, "info", "Device discovery completed successfully")
    
    def _initialize_devices(self, payload):
        # For flexibility, we provide device configurations to both initialize and discover.
        self._discover_devices(payload)
        
        if self.device_group_states == {}: 
            send_response(self.control_conn, Response.ERROR, 
                params={"message": "Invalid configuration provided"},
                logger = self.logger) 
            return 

        log_print(self.logger, "info", "Initializing devices")

        response = Response.SUCCESS

        # Now that we have a valid device configuration, try initializing 
        self.device_group_handlers = {} # Refresh
        uninit_groups = []

        group_cfg_dict = payload.get("device_groups")

        for group_id in self.device_group_states: 
            self.device_group_handlers[group_id] = None 
            
            group_dict = group_cfg_dict[group_id]
            try:
                # Get handler
                handler = get_device_group_handler(group_dict, self.response_queue, self.data_queue)
                handler.start() # Start subprocess
                
                # Initialize
                resp = handler.initialize()

                # Check for response 
                if resp.get('type', None) != Response.SUCCESS:
                    raise DeviceError(resp.get('message', 'Unknown'))

                # Store 
                for device_id in group_dict:
                    if device_id == 'metadata':
                        continue

                    self.device_group_states[group_id][device_id] = DeviceStatus.CONNECTED.value
                
                # Ensure we can access group status overall 
                self.device_group_states[group_id]["metadata"] = DeviceStatus.CONNECTED.value

                # Provide data sources to the frontend for display
                self.data_sources.update(handler.get_data_sources())

                # Store handler 
                self.device_group_handlers[group_id] = handler
            except Exception as e:
                msg = f'Unable to initialize group: {group_id}. Error: {e}'
                response = Response.WARNING
                log_print(self.logger, 'error', msg)
                uninit_groups.append(group_id)

        # Send response (both discovered states as well as initialized devices)
        send_response(
            sock = self.control_conn, 
            response = response, 
            params = {
                "device_status": self.device_group_states,
                "data_sources": [src.to_dict() for src in self.data_sources]
            },
            logger = self.logger
        )

        if len(uninit_groups) > 0:
            log_print(self.logger, "warning", f"Device initialization failed for groups: {uninit_groups}")
        else:
            log_print(self.logger, "info", "All devices successfully initialized")
        
    def _disconnect_devices(self):
        if len(self.device_group_handlers) == 0:
            msg = "Server has no initialized devices"
            log_print(self.logger, 'warning', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg}, logger = self.logger)
        
        try: 
            for handler in self.device_group_handlers.values():
                handler.disconnect()

            msg = "Devices disconnected successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg}, logger = self.logger)
        except Exception as e:
            msg = f"Failed to disconnect devices: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.control_conn, Response.ERROR, params={"message": msg}, logger = self.logger)
    
    # Handle streaming 
    def _start_streaming(self, payload): 
        if len(self.device_group_handlers) == 0: 
            msg = "Server has no initialized devices"
            log_print(self.logger, 'error', msg)
            send_response(self.control_conn, Response.ERROR, params={"message": msg}, logger = self.logger)

        # Ask all backends to start
        try:
            log_print(self.logger, 'info', "Attempting to start data streaming")

            # Start your existing receive/transmit workers
            for handler in self.device_group_handlers.values():
                handler.start_streaming(payload)

            msg = "Data streaming started successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg}, logger = self.logger)
        except Exception as e:
            msg = f"Failed to start streaming: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.control_conn, Response.ERROR, params={"message": msg}, logger = self.logger)

    def _stop_streaming(self):
        if len(self.device_group_handlers) == 0:
            msg = "Server has no initialized devices"
            log_print(self.logger, 'warning', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg}, logger = self.logger)

        try:
            log_print(self.logger, 'info', "Attempting to stop data streaming")

            for handler in self.device_group_handlers.values():
                handler.stop_streaming()

            msg = "Data streaming stopped successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg})
        except Exception as e:
            msg = f"Failed to stop streaming: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.control_conn, Response.ERROR, params={"message": msg})

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

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Launch BioView Backend Server")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Flag to make server restricteed only to local clients"
    )
    parser.add_argument(
        "--control-port",
        help=f"Port number to use for control connections. Default: {CONTROL_PORT}",
        required=False,
        default=CONTROL_PORT
    )
    parser.add_argument(
        "--data-port",
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

    args = parser.parse_args()

    server = Server(
        local_only = args.local,
        control_port = args.control_port,
        data_port = args.data_port,
        logger = logger
    )

    try:
        server.start()
    except KeyboardInterrupt:
        log_print(logger, 'warning', "Keyboard interrupt received. Shutting down server...")
    except Exception:
        log_print(logger, 'error', "Server error. Shutting down server...")
    finally:
        try:
            server.stop()
        except Exception:
            log_print(logger, 'error', "Unable to shut down server. Exiting...")
