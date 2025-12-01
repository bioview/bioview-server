""" BioView Server - Refactored for Flat Architecture """

import time 
import queue 
import argparse
import socket 
import logging 
import ipaddress

from threading import Thread, Event
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

SLEEP_DURATION = 0.001 

class Server: 
    def __init__(self, local_only: bool, control_port: int, data_port: int, logger = None): 
        self.info = get_app_info()
        self.token = 42 

        # Network info 
        self.control_port = control_port
        self.data_port = data_port

        # State
        self.status = ServerStatus.CLIENT_DISCONNECTED
        self.running = False 
        self._stop_event = Event()

        # Client / Device info
        self.local_only = local_only
        self.connected_client_info = {}
        self.device_group_states = {} 
        self.device_group_handlers = {}
        self.data_sources = set() 
        self.discovered_devices = []

        # Sockets
        self.data_socket = None
        self.data_conn = None 
        self.control_socket = None
        self.control_conn = None 

        # Workers 
        self.cmd_thread = None 
        self.data_thread = None
        self.response_queue = mp.Queue() 
        self.data_queue = mp.Queue()

        # Logging
        if not logger: 
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s: (%(levelname)s) %(message)s", datefmt='%m/%d %H:%M:%S')
        else:
            self.logger = logger 

    def _create_sockets(self):
        """Creates the listening sockets once."""
        try: 
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.control_socket.bind(("0.0.0.0", self.control_port))
            self.control_socket.listen(1)
            
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_socket.bind(("0.0.0.0", self.data_port))
            self.data_socket.listen(1)
            
            log_print(self.logger, 'debug', 'Sockets created and listening')
        except Exception as e: 
            log_print(self.logger, 'error', f'Socket creation failed: {e}')
            raise e

    def start(self): 
        """Main Server Loop."""
        log_print(self.logger, 'info', 'Starting server')
        self.running = True 
        self._stop_event.clear()

        try:
            self._create_sockets()
            while self.running and not self._stop_event.is_set():
                self.status = ServerStatus.CLIENT_DISCONNECTED
                
                # Block here until a valid client is fully connected
                if not self._accept_new_client():
                    continue 

                # Run the session until disconnect
                self._run_client_session()
                
                # Cleanup after session ends
                self._close_client_connection()

        except KeyboardInterrupt:
            log_print(self.logger, 'warning', "Keyboard interrupt.")
        except Exception as e:
            log_print(self.logger, 'error', f'Critical server error: {e}')
        finally:
            self.stop()

    def _accept_new_client(self) -> bool:
        """
        Handles the handshake. Returns True if fully connected, False otherwise.
        This isolates the nested try/except blocks from the main loop.
        """
        log_print(self.logger, 'info', 'Waiting for client connection...')
        
        # 1. Accept Control Connection
        self.control_socket.settimeout(1.0)
        try:
            conn, addr = self.control_socket.accept()
        except socket.timeout:
            return False 
        except OSError as e:
            log_print(self.logger, 'error', f'Socket accept error: {e}')
            return False

        # 2. Authenticate
        conn.settimeout(None) 
        if not self._authenticate_client(conn, addr):
            return False
            
        self.control_conn = conn
        self.status = ServerStatus.CLIENT_CONNECTED
        
        # 3. Accept Data Connection
        try:
            log_print(self.logger, 'debug', 'Waiting for data connection...')
            self.data_socket.settimeout(10.0)
            data_conn, _ = self.data_socket.accept()
            self.data_conn = data_conn
            self.data_socket.settimeout(None)
            return True
        except socket.timeout:
            log_print(self.logger, 'error', 'Timed out waiting for data connection')
            return False

    def _run_client_session(self):
        """Blocks until the command thread finishes (client disconnects)."""
        log_print(self.logger, 'info', 'Starting client session threads')
        
        self.cmd_thread = Thread(target=self._command_loop, daemon=True)
        self.data_thread = Thread(target=self._data_loop, daemon=True)
        
        self.cmd_thread.start()
        self.data_thread.start()
        
        self.cmd_thread.join()
        if self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)

    def _data_loop(self):
        """Worker: Pushes data to client."""
        while self.running and self.status >= ServerStatus.CLIENT_CONNECTED:
            if self.status < ServerStatus.STREAMING: 
                time.sleep(SLEEP_DURATION)
                continue
            
            try: 
                buff = self.data_queue.get(timeout=0.5) 
                send_datachunk(self.data_conn, buff)
            except queue.Empty:
                continue
            except Exception as e:
                log_print(self.logger, 'debug', f'Data stream ended: {e}')
                break 

    def _command_loop(self):
        """Worker: Receives commands. Exits on disconnect."""
        log_print(self.logger, 'debug', 'Command handler started')
        self.control_conn.settimeout(1.0)

        while self.running and self.status >= ServerStatus.CLIENT_CONNECTED:
            try:
                data = self.control_conn.recv(MAX_BUFFER_SIZE)
                
                # Check for disconnect
                if not data:
                    log_print(self.logger, "info", "Client disconnected gracefully")
                    break 
                
                # Check for keep-alive
                if data.count(b'\x00') == len(data) and len(data) > 0:
                    continue 

                # Process
                should_continue = self._process_command(data)
                if not should_continue:
                    break

            except socket.timeout:
                continue
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:  
                log_print(self.logger, "warning", f"Client connection lost: {e}")
                break 
            except Exception as e:
                log_print(self.logger, "error", f"Command error: {e}")
                break

        self.status = ServerStatus.CLIENT_DISCONNECTED

    def _process_command(self, data) -> bool:
        """
        Parses and executes a single command. 
        Returns False if the server should disconnect/reset.
        """
        try:
            cmd_type, payload = parse_and_validate_command(data)
        except ValidationError as e:
            log_print(self.logger, "debug", f"Validation error: {e}")
            return True

        log_print(self.logger, 'debug', f'Received {cmd_type}')

        match cmd_type:
            case Command.DISCONNECT_SERVER.name: 
                log_print(self.logger, "info", "Client requested disconnect")
                return False

            case Command.DISCOVER_DEVICES.name: 
                self._handle_discover_devices(payload)

            case Command.INITIALIZE_DEVICES.name: 
                self._handle_initialize_devices(payload)
            
            case Command.DISCONNECT_DEVICES.name: 
                self._disconnect_devices()
                
            case Command.START_STREAMING.name:
                self._start_streaming(payload)
            
            case Command.STOP_STREAMING.name: 
                self._stop_streaming()
        
        return True

    def _authenticate_client(self, client_socket, address):
        """Handles auth handshake. Returns True on success."""
        # 1. Local Check
        if self.local_only and not self._is_local_client(address):
            log_print(self.logger, 'warning', f"Rejected non-local: {address}")
            client_socket.close()
            return False
        
        # 2. Receive Initial Command
        try:
            data = client_socket.recv(MAX_BUFFER_SIZE)
            if not data:
                client_socket.close()
                return False
            cmd_type, payload = parse_and_validate_command(data)
        except Exception as e:
            log_print(self.logger, 'error', f"Auth Protocol Error: {e}")
            client_socket.close()
            return False

        # 3. Handle Discovery (Stateless)
        if cmd_type == Command.DISCOVER_SERVERS.name:
            send_response(sock=client_socket, response=Response.SUCCESS, params=self.info, logger=self.logger)
            client_socket.close()
            return False 

        # 4. Handle Connect (Stateful)
        if cmd_type == Command.CONNECT_SERVER.name:
            return self._perform_challenge_response(client_socket, payload)
        
        return False 

    def _perform_challenge_response(self, client_socket, payload):
        """Executes the specific Challenge-Response logic."""
        try:
            # Challenge
            challenge = generate_challenge()
            send_response(sock=client_socket, response=Response.SERVER_CHALLENGE, 
                          params={"challenge": challenge, "timestamp": time.time()}, logger=self.logger)

            # Response
            client_resp = client_socket.recv(MAX_BUFFER_SIZE)
            client_cmd, client_payload = parse_and_validate_command(client_resp)

            if client_cmd != Command.AUTHENTICATE_CLIENT.name:
                raise ValueError(f'Unexpected command: {client_cmd}')
            
            if not validate_token(challenge, client_payload.get("token", "")):
                send_response(sock=client_socket, response=Response.AUTHENTICATION_FAILURE, 
                              params={"message": "Invalid token"}, logger=self.logger)
                raise AuthenticationError("Invalid token")
            
            # Success
            send_response(sock=client_socket, response=Response.AUTHENTICATION_SUCCESS, 
                          params={"server_info": self.info, "timestamp": time.time()}, logger=self.logger)
            
            self.connected_client_info = {
                "ip": payload.get("ip", ""),
                "hostname": payload.get("hostname", payload.get("ip", "Unknown"))
            }
            log_print(self.logger, 'info', 'Client authenticated successfully')
            return True

        except Exception as e:
            log_print(self.logger, 'error', f"Authentication failed: {e}")
            client_socket.close()
            return False

    def _is_local_client(self, address):
        try:
            if not isinstance(address, (list, tuple)): return False 
            return ipaddress.ip_address(address[0]).is_private
        except ipaddress.AddressValueError:
            return False

    def _close_client_connection(self): 
        try:
            if self.status == ServerStatus.STREAMING:
                self._stop_streaming()
            
            self.status = ServerStatus.CLIENT_DISCONNECTED

            for sock in [self.control_conn, self.data_conn]:
                if sock:
                    try: 
                        sock.shutdown(socket.SHUT_RDWR) 
                        sock.close()
                    except OSError: pass
            
            self.control_conn = None
            self.data_conn = None
            log_print(self.logger, 'debug', "Client connection cleaned up")
        except Exception as e:
            log_print(self.logger, 'error', f"Cleanup error: {e}")

    # --- Device Handlers (Refactored for Cleanliness) ---

    def _handle_discover_devices(self, payload):
        log_print(self.logger, "info", "Discovering devices")
        self.discovered_devices = []
        
        for backend_type, backend in AVAILABLE_BACKENDS.items(): 
            try:
                self.discovered_devices.extend(backend.discover_devices())
            except Exception as e: 
                log_print(self.logger, 'warning', f'{backend_type} discovery error: {e}')
        
        device_groups = payload.get('device_groups')
        if not device_groups or not is_dict_of_dicts(device_groups):
            # If no groups provided, we just return the raw discovery list usually, 
            # or in this protocol we might just return error if payload was required.
            # Assuming we just need to update internal state.
            return 

        self.device_group_states = {} 
        for group_id, group_dict in device_groups.items(): 
            self.device_group_states[group_id] = {}
            for device_id in group_dict.keys():
                if device_id == 'metadata': continue
                status = DeviceStatus.AVAILABLE.value if device_id in self.discovered_devices else DeviceStatus.UNAVAILABLE.value
                self.device_group_states[group_id][device_id] = status

        send_response(self.control_conn, Response.SUCCESS, 
            params={"device_status": self.device_group_states, "data_sources": self.data_sources})

    def _handle_initialize_devices(self, payload):
        self._handle_discover_devices(payload) # Updates state first
        
        if not self.device_group_states: 
            return send_response(self.control_conn, Response.ERROR, params={"message": "Invalid configuration"}) 

        log_print(self.logger, "info", "Initializing devices")
        self.device_group_handlers = {}
        uninit_groups = []

        for group_id, group_dict in payload.get("device_groups", {}).items(): 
            try:
                handler = get_device_group_handler(group_dict, self.response_queue, self.data_queue)
                handler.start() 
                
                resp = handler.initialize()
                if resp.get('type') != Response.SUCCESS:
                    raise DeviceError(resp.get('message', 'Unknown'))

                # Update Status
                for device_id in group_dict:
                    if device_id == 'metadata': continue
                    self.device_group_states[group_id][device_id] = DeviceStatus.CONNECTED.value
                self.device_group_states[group_id]["metadata"] = DeviceStatus.CONNECTED.value
                
                self.data_sources.update(handler.get_data_sources())
                self.device_group_handlers[group_id] = handler
            except Exception as e:
                log_print(self.logger, 'error', f'Init failed for {group_id}: {e}')
                uninit_groups.append(group_id)

        response = Response.WARNING if uninit_groups else Response.SUCCESS
        send_response(self.control_conn, response, 
            params={"device_status": self.device_group_states, "data_sources": [src.to_dict() for src in self.data_sources]})

    def _disconnect_devices(self):
        if not self.device_group_handlers:
            return send_response(self.control_conn, Response.SUCCESS, params={"message": "No devices active"})
        
        for handler in self.device_group_handlers.values():
            try: handler.disconnect()
            except Exception: pass
            
        self.device_group_handlers.clear()
        send_response(self.control_conn, Response.SUCCESS, params={"message": "Devices disconnected"})
    
    def _start_streaming(self, payload): 
        if not self.device_group_handlers: 
            return send_response(self.control_conn, Response.ERROR, params={"message": "No devices active"})

        try:
            for handler in self.device_group_handlers.values():
                handler.start_streaming(payload)
            self.status = ServerStatus.STREAMING
            send_response(self.control_conn, Response.SUCCESS, params={"message": "Streaming started"})
        except Exception as e:
            send_response(self.control_conn, Response.ERROR, params={"message": str(e)})

    def _stop_streaming(self):
        self.status = ServerStatus.DEVICES_CONNECTED
        if not self.device_group_handlers:
            return send_response(self.control_conn, Response.SUCCESS, params={"message": "Stopped"})

        try:
            for handler in self.device_group_handlers.values():
                handler.stop_streaming()
            send_response(self.control_conn, Response.SUCCESS, params={"message": "Streaming stopped"})
        except Exception as e:
            send_response(self.control_conn, Response.ERROR, params={"message": str(e)})

    def stop(self): 
        log_print(self.logger, 'debug', "Shutting down server...")
        self.running = False
        self._stop_event.set()
        self._close_client_connection()
        for sock in [self.control_socket, self.data_socket]:
            if sock:
                try: sock.close()
                except OSError: pass

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Launch BioView Backend Server")
    parser.add_argument("--discoverable", action="store_true", help="Allow non-local clients")
    parser.add_argument("--control-port", default=CONTROL_PORT)
    parser.add_argument("--data-port", default=DATA_PORT)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s: (%(levelname)s) %(message)s", datefmt='%m/%d %H:%M:%S')
    log_print(logger, 'info', f"BioView Device Server, Version: {APP_VERSION}")

    args = parser.parse_args() 

    server = Server(
        local_only = args.discoverable, # FIX LATER
        control_port = args.control_port,
        data_port = args.data_port,
        logger = logger
    )

    server.start()