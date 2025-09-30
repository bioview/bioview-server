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
import argparse
import socket 
import logging 
import ipaddress
import contextlib

from threading import Thread

import multiprocessing as mp 

from bioview_common import (    
    AuthenticationError,
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
    generate_challenge, 
    validate_token, 
    parse_and_validate_command
)

from bioview_server.device import AVAILABLE_BACKENDS, get_device_group_handler

class Server: 
    def __init__(
        self,
        local_only: bool, 
        control_port: int,
        data_port: int,
        logger = None
    ): 
        # Keep track of local PC information for client communication 
        self.info = get_app_info()
        self.token = 42 # TODO: Load using secrets

        # Network info 
        self.control_port = control_port
        self.data_port = data_port

        # Server status 
        self.status = ServerStatus.CLIENT_DISCONNECTED
        self.running = False 

        # Client handling 
        self.local_only = local_only
        self.discovered_clients = {} 
        self.connected_client_info = {}
        
        # Device handling
        self.device_group_states = {} 
        self.device_group_handlers = {}

        # Sockets
        self.data_socket = None
        self.data_conn = None 
        self.control_socket = None
        self.control_conn = None 

        # Threaded workers 
        self.cmd_thread = None 

        # Queues for overall logging
        self.response_queue = mp.Queue() 

        # Message logging
        if not logger: 
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger 

    def start(self): 
        log_print(self.logger, 'info' 'Starting server')
        
        # Open sockets 
        self._create_sockets()

        # Create listener threads 
        self.running = True 

        try:
            # Start server threads
            control_thread = Thread(
                target=self._control_handler, 
                daemon=True
            )
            data_thread = Thread(
                target=self._data_handler, 
                daemon=True
            )

            control_thread.start()
            data_thread.start()

            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except Exception:
            log_print(self.logger, 'error', 'Unable to start server')
        finally:
            self.stop()

    def _create_sockets(self):
        # Setup control socket
        try: 
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.control_socket.bind(("0.0.0.0", self.control_port))
            self.control_socket.listen(5)
            log_print(self.logger, 'info', 'Control socket connected')
        except Exception as e: 
            log_print(self.logger, 'error', f'Unable to create control socket: {e}')

        # Setup data socket
        try: 
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_socket.bind(("0.0.0.0", self.data_port))
            self.data_socket.listen(10)
            log_print(self.logger, 'info', 'Data socket connected')
        except Exception as e: 
            log_print(self.logger, 'error', f'Unable to create data socket: {e}')

    def _control_handler(self): 
        while self.running:  
            if self.status is ServerStatus.CLIENT_DISCONNECTED:
                # Ensure that no other commands can be accepted 
                if self.cmd_thread: 
                    self.cmd_thread.join() 
                    self.cmd_thread = None 

                # Accept incoming connections (blocks)
                # Update socket reference to be able to accept
                conn, addr = self.control_socket.accept()
            
                # Authenticate
                if self._authenticate_client(conn, addr):
                    # Update status 
                    self.status = ServerStatus.CLIENT_CONNECTED
                    self.control_conn = conn 

            # If authenticated, parse commands
            if self.status is ServerStatus.CLIENT_CONNECTED:
                # Ensure we do not spawn endless threads
                if not self.cmd_thread: 
                    self.cmd_thread = Thread(
                        target=self._command_handler, 
                        daemon=True 
                    )
                    self.cmd_thread.start()

                # Check if sockets are still alive and update status accordingly
                time.sleep(1)

    def _data_handler(self):
        while self.running and self.status is ServerStatus.CLIENT_CONNECTED: 
            with contextlib.suppress(Exception):
                # Convert from listen() to accept() socket 
                self.data_conn, _ = self.data_socket.accept()
                # TODO: Complete

    def _command_handler(self):
        while self.running and self.status is ServerStatus.CLIENT_CONNECTED: 
            try:
                # Receive data continuously 
                data = self.control_conn.recv(MAX_BUFFER_SIZE)

                if data == b'\x00':
                    # if keep-alive ping 
                    continue

                # Dispatch appropriate function 
                cmd_type, payload = parse_and_validate_command(data)
                log_print(self.logger, 'debug', f'Received {cmd_type} with {payload}')

                match cmd_type:
                    # Client commands
                    case Command.DISCONNECT_SERVER.name: 
                        self._close_client_connection()

                    # Device commands 
                    case Command.DISCOVER_DEVICES.name: 
                        # Pass device configuration here 
                        discovery_states = self._discover_devices(payload)
                        if discovery_states:
                            send_response(
                                self.control_conn, 
                                Response.SUCCESS, 
                                params={"discovery_status": discovery_states}
                            )
                        else:
                            send_response(
                                self.control_conn, 
                                Response.ERROR, 
                                params={"message": "Invalid configuration provided"})
                    case Command.INITIALIZE_DEVICES.name: 
                        self._initialize_devices(payload)
                    case Command.DISCONNECT_DEVICES.name: 
                        self._disconnect_devices()
                        
                    # Streaming 
                    case Command.START_STREAMING.name: 
                        self._start_streaming()
                    case Command.STOP_STREAMING.name: 
                        self._stop_streaming()
            
            except (ConnectionResetError, ConnectionAbortedError): 
                # This will occur when client disconnects. Reset state 
                log_print(self.logger, "warning", "Client connection unexpectedly terminated.")
                self._close_client_connection()

    def _is_local_client(self, address):
        try:
            if not isinstance(address, (list, tuple)):
                # Python socket will provide a tuple (ip, port)
                return False 
            

            return ipaddress.ip_address(address[0]).is_private
        except ipaddress.AddressValueError:
            log_print(self.logger, 'error', f"{address} is not a valid IP address")
            return False

    def _authenticate_client(self, client_socket, address):
        # If local only mode, reject non-local connections
        if self.local_only and not self._is_local_client(address):
            client_socket.close()
            return False
        
        data = client_socket.recv(MAX_BUFFER_SIZE)
        if not data:
            client_socket.close()
            return False

        try: 
            cmd_type, payload = parse_and_validate_command(data)
        except Exception as e:
            log_print(self.logger, 'error', str(e))
            return False 
        
        log_print(self.logger, 'debug', f'Received {cmd_type} with {payload}')

        if cmd_type == Command.DISCOVER_SERVERS.name:
            # Respond with server details 
            send_response(
                sock = client_socket, 
                response = Response.SUCCESS, 
                params = self.info,  
                logger = self.logger
            )
            client_socket.close() # Close connection for now
            
            # We are still not authenticated
            return False 

        elif cmd_type == Command.CONNECT_SERVER.name:
            # Send challenge 
            challenge = generate_challenge()

            send_response(
                sock = client_socket, 
                response = Response.SERVER_CHALLENGE, 
                params = {
                    "challenge": challenge, 
                    "timestamp": time.time()
                },
                logger = self.logger
            )

            # Parse response to challenge
            try: 
                client_response = client_socket.recv(MAX_BUFFER_SIZE)
                client_cmd, client_payload = parse_and_validate_command(client_response)

                if client_cmd != Command.AUTHENTICATE_CLIENT.name:
                    raise ValueError(f'Unexpected client command: {client_cmd}')
            
                auth_token = client_payload.get("token", "")
                
                if not validate_token(challenge, auth_token):
                    send_response(
                        sock = client_socket, 
                        response = Response.AUTHENTICATION_FAILURE, 
                        params = {
                            "message": "Invalid authentication token"
                        },
                        logger = self.logger
                    )
                    raise AuthenticationError("Invalid authentication token") from None
                
                # If we are here, we have succeeded in authentication 
                send_response(
                    sock = client_socket, 
                    response = Response.AUTHENTICATION_SUCCESS, 
                    params = {
                        "server_info": self.info, 
                        "timestamp": time.time()
                    },
                    logger = self.logger
                )

                # Store client info 
                self.connected_client = {
                    "ip": payload.get("ip", ""),
                    "hostname": payload.get("hostname", ""),
                    "name": payload.get("name", ""),
                    "version": payload.get("version", ""),
                }
            except Exception as e: 
                log_print(self.logger, 'error', f'Client authentication failed: {e}')
                client_socket.close()
                return False 
            
            return True
        
        else: 
            return False 

    # Client command handling callbacks
    def _close_client_connection(self): 
        try:
            # Close accepted sockets
            if self.control_conn:
                self.control_conn.close()
            self.control_conn = None 

            if self.data_conn:
                self.data_conn.close()
            self.data_conn = None 

            log_print(self.logger, 'debug', "Server disconnected successfully")
        except Exception as e:
            log_print(self.logger, 'error', f"Server disconnection error: {e}")

        finally:
            self.status = ServerStatus.CLIENT_DISCONNECTED

    # Device command handling callbacks 
    def _discover_devices(self, payload): 
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
        for group_id, group_dict in device_groups.items(): 
            for device_id in group_dict.keys():
                if device_id in self.discovered_devices:
                    device_groups[group_id][device_id]['status'] = DeviceStatus.AVAILABLE.name
                else: 
                    device_groups[group_id][device_id]['status'] = DeviceStatus.UNAVAILABLE.name

        
        return device_groups

    def _initialize_devices(self, payload):
        # For flexibility, we provide device configurations to both initialize and discover.
        device_groups = self._discover_devices(payload)
        
        if not device_groups: 
            send_response(self.control_conn, Response.ERROR, 
                params={"message": "Invalid configuration provided"}) 
            return 

        response = Response.SUCCESS

        # Now that we have a valid device configuration, try initializing 
        self.device_group_states = {} # Refresh
        self.device_group_handlers = {} # Refresh
        
        for group_id, group_dict in device_groups.items(): 
            try:
                # Get handler
                handler = get_device_group_handler(group_dict, self.response_queue)
                # Initialize
                handler.initialize()
                # Store 
                self.device_group_states[group_id] = DeviceStatus.CONNECTED.name
                self.device_group_handlers[group_id] = handler
            except Exception as e:
                msg = f'Unable to initialize group: {group_id}. Error: {e}'
                response = Response.WARNING
                log_print(self.logger, 'error', msg)

        # Send response (both discovered states as well as initialized devices)
        send_response(
            sock = self.control_conn, 
            response = response, 
            params = {
                "discovered_devices": device_groups, 
                "connection_states": self.device_group_states
            }
        )

        
    def _disconnect_devices(self):
        if len(self.device_group_handlers) == 0:
            msg = "Server has no initialized devices"
            log_print(self.logger, 'warning', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg})
        
        try: 
            for handler in self.device_group_handlers.values():
                handler.disconnect()

            msg = "Devices disconnected successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg})
        except Exception as e:
            msg = f"Failed to disconnect devices: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.control_conn, Response.ERROR, params={"message": msg})
    
    # Handle streaming 
    def _start_streaming(self): 
        if len(self.device_group_handlers) == 0: 
            msg = "Server has no initialized devices"
            log_print(self.logger, 'error', msg)
            send_response(self.control_socket, Response.ERROR, params={"message": msg})

        # Ask all backends to start
        try:
            log_print(self.logger, 'info', "Attempting to start data streaming")

            # Start your existing receive/transmit workers
            for handler in self.device_group_handlers.values():
                handler.start_streaming()

            self.status = ServerStatus.STREAMING

            msg = "Data streaming started successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.control_socket, Response.SUCCESS, params={"message": msg})
        except Exception as e:
            msg = f"Failed to start streaming: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.control_socket, Response.ERROR, params={"message": msg})

    def _stop_streaming(self):
        if len(self.device_group_handlers) == 0:
            msg = "Server has no initialized devices"
            log_print(self.logger, 'warning', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg})

        try:
            log_print(self.logger, 'info', "Attempting to stop data streaming")

            for handler in self.device_group_handlers.values():
                handler.stop_streaming()

            self.status = ServerStatus.DEVICES_CONNECTED

            msg = "Data streaming stopped successfully"
            log_print(self.logger, 'info', msg)
            send_response(self.control_conn, Response.SUCCESS, params={"message": msg})
        except Exception as e:
            msg = f"Failed to stop streaming: {e}"
            log_print(self.logger, 'error', msg)
            send_response(self.control_conn, Response.ERROR, params={"message": msg})

    def stop(self): 
        log_print(self.logger, 'debug', f"Attempting to shutdown server")
        self.running = False
        
        self._close_client_connection()
        log_print(self.logger, 'debug', f"Server shut down successfully")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Launch BioView Backend Server")
    parser.add_argument(
        "--discoverable",
        action="store_true", 
        help="Flag to make non-local clients be able to discover the server"
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
        local_only = args.discoverable,   # TODO: Correct it to not args.discoverable
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