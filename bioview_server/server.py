""" BioView Server

The server exposes a flexible way for connecting to devices by forwarding client commands
to the appropriate handlers. 

Note that, as of now, the server assumes a single connected client since there is ambiguity
regarding control from multiple clients. Specifically, it is not yet clear whether we 
require a main client and other observer clients or whether every client needs to be provided
the same level of access or whether each device should be considered as a different client 
connection. Subsequent experimentation and discussion will be pertinent for expanding 
functionality to handle the case for multiple clients. 
"""
import socket
import secrets 
import json
import time
import traceback
import multiprocessing as mp
from threading import Thread, Lock
from typing import List, Dict, Tuple, Any 

from bioview_server.datatypes import Backend
from bioview_server.device import AVAILABLE_BACKENDS
from bioview_server.utils import parse_and_validate_command

from bioview_common import (
    Command, Response, ServerStatus, Configuration, 
    MAX_BUFFER_SIZE, APP_VERSION, AUTH_TIMEOUT,
    CONTROL_PORT, DATA_PORT, RESPONSE_TIMEOUT,
    get_ip, get_app_info, AuthenticationError
)

print(f'Available Backends: {list(AVAILABLE_BACKENDS.keys())}')

class Server:
    def __init__(
        self, 
        control_port: int = CONTROL_PORT, 
        data_port: int = DATA_PORT, 
        discoverable: bool = False, # By default, run server using local-only mode for safety. 
        auth_timeout: int = AUTH_TIMEOUT,
        resp_timeout: int = RESPONSE_TIMEOUT

    ):
        # Server network information 
        self.address = get_ip()
        self.server_info = get_app_info()
        
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
        
    def start(self):
        print(f'Starting server on {self.server_info['hostname']}')
        
        try:     
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.control_socket.bind((self.address, self.control_port))
            self.control_socket.listen(5)
            print(f"✓ Control server listening on {self.address}:{self.control_port}")
            
            # Start data server
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_socket.bind((self.address, self.data_port))
            self.data_socket.listen(10)  # More clients for data
            print(f"✓ Data server listening on {self.address}:{self.data_port}")
            
        except Exception as e: 
            print(f'Error occurred while starting server: {e}')
        
        # Once client have started, start listening
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
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        print(f'Stopping server')
        self.running = False 
        
        self.handle_disconnect_from_client() 

        print("=" * 50)
        print("BioView server stopped")
        print("=" * 50)
    
    # Ensure connection validation for server security
    def validate_and_authenticate_client(
        self, 
        client_socket: socket.socket, 
        client_address: str
    ) -> Dict[str, Any]:
        """
        Returns client info if successful, raises AuthenticationError if failed
        """
        client_ip = client_address[0]
        
        try:
            client_socket.settimeout(self.auth_timeout)
            
            # Step 1: Receive client hello
            client_syn = client_socket.recv(4096).decode('utf-8')
            _, client_syn_payload = parse_and_validate_command(client_syn, Command.CONNECT_SERVER.value)

            # Get client information
            client_info = {
                "hostname": client_syn_payload['hostname'], 
                "app_name": client_syn_payload['app_name'], # TODO: Replace with app_token
                "app_version": client_syn_payload['version'],
            }
            print(f'Connection request received from {client_info['hostname']}')

            # Check timestamp to prevent replay attacks (optional)
            client_timestamp = client_syn_payload.get('timestamp', 0)
            current_time = time.time()
            if abs(current_time - client_timestamp) > AUTH_TIMEOUT * 60:  # Suitable window
                raise AuthenticationError("Client timestamp outside acceptable window")            
            
            # Step 2: Send challenge
            challenge = self._generate_challenge()
            server_challenge_dict = {
                'type': Response.SERVER_CHALLENGE.value,
                'payload': {
                    'challenge': challenge,    
                    'timestamp': time.time()
                }
            }            
            server_challenge = json.dumps(server_challenge_dict).encode('utf-8')
            client_socket.send(server_challenge)
            
            # Step 3: Receive client response for authentication
            client_response_data = client_socket.recv(4096).decode('utf-8')
            client_response_type, client_response_payload = parse_and_validate_command(client_response_data, Command.AUTHENTICATE_CLIENT.value)
            # Verify 
            expected_token = self._compute_response(challenge, self.app_token)
            received_token = client_response_payload.get('auth_token', '')
            
            if not secrets.compare_digest(expected_token, received_token):
                auth_failure = {
                    'type': Response.AUTHENTICATION_FAILURE.value,
                    'payload': {
                        'message': "Invalid authentication token"
                    }
                }
                server_response = json.dumps(auth_failure).encode('utf-8')
                client_socket.send(server_response)

                raise AuthenticationError("Invalid authentication token")
            
            # Step 4: Send success confirmation with hostname info
            auth_success = {
                'type': Response.AUTHENTICATION_SUCCESS.value,
                'payload': {
                    'hostname': self.server_info['hostname'],
                    'app_name': self.server_info['app_name'], 
                    'app_version': self.server_info['app_version'], 
                    'timestamp': time.time()
                }
            }
            server_response = json.dumps(auth_success).encode('utf-8')
            client_socket.send(server_response)
            
            print(f'Successfully connected to {client_info['hostname']}')
            return client_info
            
        except socket.timeout:
            raise AuthenticationError("Authentication timeout")
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"Authentication failed: {str(e)}")   

    def handle_control_connection(self):  
        while self.running:
            try:
                client_socket, address = self.control_socket.accept()
                self.client_info = self.validate_and_authenticate_client(
                    client_socket = client_socket, 
                    client_address = address
                )
                
                client_thread = Thread(
                    target=self.handle_commands,
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()
            except Exception as e:
                if self.running:
                    print(f"Error accepting control connection: {e}")

    def handle_data_connection(self): 
        while self.running:
            try:
                client_socket, address = self.data_socket.accept()
                print(f"Data client connected from {address}")
                
                with self.data_lock:
                    self.data_clients.append(client_socket)
                
                # Handle client disconnect
                def monitor_client(sock):
                    try:
                        while self.running:
                            # Send keepalive
                            sock.send(b'')
                            time.sleep(1)
                    except:
                        with self.data_lock:
                            if sock in self.data_clients:
                                self.data_clients.remove(sock)
                        sock.close()
                        print(f"Data client {address} disconnected")
                
                Thread(target=monitor_client, args=(client_socket,), daemon=True).start()
                
            except Exception as e:
                if self.running:
                    print(f"Error accepting data connection: {e}")
    
    def handle_commands(self, client_socket): 
        # Receives commands from clients and controls device handlers accordingly
        try:
            while self.running:
                data = client_socket.recv(MAX_BUFFER_SIZE)
                if not data:
                    break
                
                try:
                    command = json.loads(data.decode('utf-8'))
                    response = self.process_command(command)
                    
                    response_data = json.dumps(response).encode('utf-8')
                    client_socket.send(response_data)
                except json.JSONDecodeError as e:
                    error_response = {
                        'type': Response.ERROR.value,
                        'payload': {
                            'message': f"Invalid JSON: {e}"
                        }
                    }
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    
        except Exception as e:
            print(f"Control client error: {e}")
        finally:
            client_socket.close()
    
    def process_command(self, command):
        # Redirect received command from client to the appropriate callback 
        cmd_type = command.get('type')
        payload = command.get('payload', {})

        try:
            # Server commands 
            if cmd_type == Command.PING_SERVER.value:
                return self.handle_ping()
            elif cmd_type == Command.CONNECT_SERVER.value: 
                return self.handle_connect_to_client(payload)
            elif cmd_type == Command.DISCOVER_DEVICES.value:
                return self.handle_discover_devices()
            elif cmd_type == Command.DISCONNECT_SERVER.value:
                return self.handle_disconnect_from_client()
            
            # Device commands
            elif cmd_type == Command.CONNECT_DEVICES.value:
                return self.handle_connect_device(payload) 
            elif cmd_type == Command.GET_DEVICE_STATUS.value:
                return self.handle_get_device_status(payload)
            elif cmd_type == Command.START_STREAMING.value:
                return self.handle_start_streaming()
            elif cmd_type == Command.STOP_STREAMING.value:
                return self.handle_stop_streaming()
            elif cmd_type == Command.UPDATE_RUNNING_PARAMETER.value:
                return self.handle_update_runing_parameter(payload)
            elif cmd_type == Command.UPDATE_DEVICE_FIRMWARE.value: 
                return self.handle_update_device_firmware(payload)
            elif cmd_type == Command.DISCONNECT_DEVICES.value:
                return self.handle_disconnect_device()
            else:
                return {
                    'type': Response.ERROR.value,
                    'payload': {
                        'message': f"Unknown command: {cmd_type}",
                    }
                }
                
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'payload': {
                    'message': f"Command processing error: {e}"
                }
            }
    
    # Server commands 
    def handle_ping(self):
        # Return server status 
        return {
            'type': Response.INFO.value,
            'payload': {
                'hostname': self.server_info['hostname'],
                'version': self.server_info['version'],
                'status': self.status.value,
            }
        }
    
    def handle_initialize_common_configuration(self, client_dict):
        '''
        Server will only respond to commands if connected to a client. This handler validates a client and switches server state to be actually useful
        '''
# TODO: Initialize server common configuration
            # TODO: These should not be part of the device handling
        # exp_config = Configuration.from_json(params['exp_config'])
        # save = params.get('save', False)
        pass 

    def handle_discover_devices(self):
        # Call device.discover for all device backends 
        # Returns a list of devices along with handler objects (minimal init?)
        print("🔍 Starting device discovery...")

        try: 
            for backend_type, backend_handler in AVAILABLE_BACKENDS.items(): 
               self.discovered_devices[backend_type] = backend_handler.discover_devices()
        except Exception as e: 
            print("Device discovery failed.")
            return {
                    'type': Response.ERROR.value,
                    'payload': {
                        'message': f'Device discovery failed: {e}'
                    }
                }    
        
        print("Device discovery completed successfully.")
        return {
                'type': Response.SUCCESS.value,
                'payload': {
                    'message': f'Found {len(self.discovered_devices)} devices',
                    'devices': self.discovered_devices
                }
            }
    
    def handle_disconnect_from_client(self):
        # Disconnect clients from servers
        print("Disconnecting server from clients.")
        
        try:
            # Close sockets
            if self.control_socket:
                self.control_socket.close()
            if self.data_socket:
                self.data_socket.close()

            return {
                'type': Response.SUCCESS.value,
                'payload': {
                    'message': 'Server disconnected successfully'
                }
            }
        except Exception as e: 
            return {
                'type': Response.ERROR.value,
                'payload': {
                    'message': f'Server disconnection error: {e}'
                }
            }
        finally:
            self.control_socket = None 
            self.data_socket = None  
            self.status = ServerStatus.CLIENT_DISCONNECTED

    # Device commands
    def handle_connect_device(self, configurations: Dict[Dict]): 
        '''
        Provided params will typically include device specific configurations, using which all devices are initialized.
        Configurations provided by the client are considered canonical (vis-a-vis values), regardless of any pre-existing configs.
        '''
        try: 
            # Firstly, initialize a suitable backend handler
            for device_id, device_config_dict in configurations.items(): 
                # Make device configuration object
                device_configuration = Configuration.from_dict(device_config_dict)

                # Create the backend handler
                backend_type = device_configuration.get_param('backend_type')
                backend_module = AVAILABLE_BACKENDS[backend_type] 
                
                # Each handler will have its own way of parsing provided configuration but must be able to handle the following parameters 
                handler = backend_module.get_backend_handler(
                    # Configuration
                    configuration = device_configuration, 
                    # Shared queues for data handling
                    display_data_queue = self.display_data_queue,
                    command_queue = self.command_queue, 
                    response_queue = self.response_queue
                ) 
                
                # Now, initialize the handler (which will also initialize the device) 
                handler.initialize()

                # Lastly, store reference 
                self.backends[device_id] = handler

                # Communicate success
                return {
                    'type': Response.SUCCESS.value,
                    'payload': {
                        'message': 'Device inited successfully'
                    }                    
                }
                    
        except Exception as e:
            # Communicate failure 
            return {
                'type': Response.ERROR.value,
                'payload': {
                    'message': f'Device initialization failed: {e}',
                    'traceback': traceback.format_exc()
                }                    
            }
    
    def handle_get_device_status(self, param_dict): 
        device_id = param_dict.get('device_id', None)
        if not device_id: 
            return self.backends[device_id].get_device_status() 
         
    def handle_start_streaming(self): 
        # Order devices to start streaming 
        if len(self.backends) == 0: 
            return {
                'type': Response.WARNING.value,
                'payload': {
                    'message': 'No devices connected.'
                }
            }
        
        try:
            print("🚀 Starting data streaming...")
            
            # Start your existing receive/transmit workers
            for backend in self.backends.values():
                backend.start_streaming()
            
            self.status = ServerStatus.STREAMING 
            
            print("✓ Data streaming started")
            return {
                'type': Response.SUCCESS.value,
                'payload': {
                    'message': 'Data streaming started'
                }
            }   
        except Exception as e: 
            return {
                'type': Response.ERROR.value,
                'payload': {
                    'message': f'Failed to start streaming: {e}'
                }
            }
    
    def handle_stop_streaming(self): 
        if self.status != ServerStatus.STREAMING: 
            return {
                'type': Response.WARNING.value,
                'payload': {
                    'message': 'Server is not currently streaming'
                }
            }
        try: 
            print("🛑 Stopping data streaming...")
            for backend in self.backends.values(): 
                backend.stop_streaming() 
                    
            self.status = ServerStatus.DEVICES_CONNECTED

            return {
                'type': Response.SUCCESS.value,
                'payload': {
                    'message': 'Data streaming stopped'
                }
            }
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'payload': {
                    'message': f'Failed to stop streaming: {e}'
                }
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
            
            print("✓ Devices disconnected")
            
            self.status = ServerStatus.DEVICES_DISCONNECTED

            return {
                'type': Response.SUCCESS.value,
                'payload': {
                    'message': 'Devices successfully disconnected'
                }
            }   
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'payload': {
                    'message': f'Disconnect error: {e}'
                }
            }
        
    def handle_update_device_config(self, group_id, param, value, device_id = ''):        
        if group_id not in self.backends.keys(): 
            return {
                'type': Response.ERROR.value,
                'message': f'Invalid device {group_id} specified for modification.'
            }
        
        backend = self.backends[group_id] # type(BACKEND)

        backend.command_queue.put({
            'param': param,
            'value': value,
            'device_id': device_id # For device groups where individual access is required 
        })         
    
if __name__ == "__main__":
    print("=" * 50)
    print(f"BioView Device Server, Version: {APP_VERSION}")
    print("=" * 50)
    
    server = Server()
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
    finally:
        server.stop()