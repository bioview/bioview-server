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
import sys 

# Populate available backends 
AVAILABLE_BACKENDS = {}
try:
    # Ensure uhd is available
    import uhd # Crashes occur without this
    # Ensure device is importable 
    from bioview_server.device import USRPBackend
    AVAILABLE_BACKENDS['usrp'] = USRPBackend
except Exception as e: 
    print(f'USRP backend not available: {e}')

try: 
    # Ensure platform is windows 
    if sys.platform != 'win32':
        raise OSError(f'Invalid platfrom {sys.platform}. Ensure you are using Windows')
    # Ensure mpdev.dll exists 
    from bioview_server.device.biopac import load_mpdev_dll
    if load_mpdev_dll() == None:
        raise ValueError('mpdev.dll not found')
    from bioview_server.device import BIOPACBackend
    AVAILABLE_BACKENDS['biopac'] = BIOPACBackend
except Exception as e:  
    print(f'BIOPAC backend not available: {e}')

print(f'Available Backends: {list(AVAILABLE_BACKENDS.keys())}')

import socket
import json
import time
from threading import Thread, Lock
import multiprocessing as mp
import traceback
from enum import Enum, auto

from bioview_server.datatypes import Configuration

from bioview_common import Command, Response, MAX_BUFFER_SIZE, APP_VERSION, CONTROL_PORT, DATA_PORT, get_ip, get_app_info

class ServerStatus(Enum): 
    DEFAULT = auto      # Nothing is going on
    CLIENT_CONNECTED = auto
    CLIENT_DISCONNECTED = auto
    DEVICES_CONNECTED = auto
    DEVICES_DISCONNECTED = auto 
    STREAMING = auto

class Server:
    def __init__(self, control_port=CONTROL_PORT, data_port=DATA_PORT):
        # Server network information 
        self.address = get_ip()
        self.server_info = get_app_info()
        
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
        self.device_handlers = [] 
        self.data_queue = mp.Queue()
        
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
        
        self.handle_disconnect_server() 

        print("=" * 50)
        print("BioView server stopped")
        print("=" * 50)
         
    def handle_control_connection(self):  
        while self.running:
            try:
                client_socket, address = self.control_socket.accept()
                # TODO: Validate by syn-ack
                print(f"Control client connected from {address}")
                
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
        
        try:
            if cmd_type == Command.PING_SERVER.value:
                return self.handle_ping()
            elif cmd_type == Command.DISCOVER_DEVICES.value:
                return self.handle_discover_devices()
            elif cmd_type == Command.INIT_DEVICES.value: 
                return self.handle_init_device(command.get('payload', {})) 
            elif cmd_type == Command.CONNECT_DEVICES.value:
                return self.handle_connect_device()
            elif cmd_type == Command.DISCONNECT_DEVICES.value:
                return self.handle_disconnect_device()
            elif cmd_type == Command.GET_DEVICE_STATUS.value:
                return self.handle_get_device_status()
            elif cmd_type == Command.START_STREAMING.value:
                return self.handle_start_streaming()
            elif cmd_type == Command.STOP_STREAMING.value:
                return self.handle_stop_streaming()
            elif cmd_type == Command.UPDATE_DEVICE_CONFIGURATION.value:
                return self.handle_update_device_configuration(command.get('payload', {}))
            elif cmd_type == Command.UPDATE_DEVICE_FIRMWARE.value: 
                return self.handle_update_device_firmware(command.get('payload', {}))
            elif cmd_type == Command.DISCONNECT_SERVER.value:
                return self.handle_disconnect_server()
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
    
    def handle_init_device(self, params): 
        '''
        Provided params will typically include device specific configurations, using which all devices are initialized.
        Configurations provided by the client are considered canonical, regardless of any pre-existing configs.
        '''
        try: 
            for device_id, device_config in params.items(): 
                config = Configuration.from_dict(device_config)
        
        # TODO: These should not be part of the device handling
        # exp_config = Configuration.from_json(params['exp_config'])
        # save = params.get('save', False)

                self.device_handlers[device_id] = DeviceHandler(config=config, data_queue=self.data_queue, exp_config=exp_config, save=save)
            print("✓ Device inited")
            
            return {
                'type': Response.SUCCESS.value,
                'message': 'Device inited successfully'
            }
            
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'message': f'Initialization failed: {e}',
                'traceback': traceback.format_exc()
            }
        
    def handle_connect_device(self):
        # Send command to connect all devices
        try:
            for device in self.device_handlers.values(): 
                device.connect()
            
            print("✓ Devices connected")
            
            self.status = ServerStatus.DEVICES_CONNECTED

            return {
                'type': Response.SUCCESS.value,
                'payload': {
                    'message': 'Connect successful'   
                }
            }
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'payload': {
                    'message': f'Connect error: {e}'
                }
            }
    
    def handle_disconnect_device(self):
        # Disconnect all devices
        try:
            for device in self.device_handlers.values(): 
                device.disconnect()
            
            print("✓ Devices disconnected")
            
            self.status = ServerStatus.DEVICES_DISCONNECTED

            return {
                'type': Response.SUCCESS.value,
                'payload': {
                    'message': 'Disconnect successful'
                }
            }   
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'payload': {
                    'message': f'Disconnect error: {e}'
                }
            }
        
    # TODO
    def handle_update_device_config(self, params):
        device_id = params['id']
        
        if device_id not in self.device_handlers.keys(): 
            return {
                'type': Response.ERROR.value,
                'message': f'Device not initialized'
            }
        
        device_handler = self.device_handlers[device_id]
        
        # Updated config occurs here 
        for key, value in params['config']: 
            device_handler.update_config(key, value) 

    # TODO   
    def handle_update_device_param(self, params): 
        device_id = params['id']
        
        if device_id not in self.device_handlers.keys(): 
            return {
                'type': Response.ERROR.value,
                'message': f'Device not initialized'
            }
        
        device_handler = self.device_handlers[device_id]
        
        # Updated config occurs here 
        for key, value in params['config']: 
            device_handler.update_param(key, value) 
    
    def handle_disconnect_server(self):
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
        
    def handle_start_streaming(self): 
        # Order devices to start streaming 
        if len(self.device_handlers) == 0: 
            return {
                'type': Response.WARNING.value,
                'payload': {
                    'message': 'No devices connected.'
                }
            }
        
        try:
            print("🚀 Starting data streaming...")
            
            # Start your existing receive/transmit workers
            for handler in self.device_handlers.values():
                handler.start()
            
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
            for handler in self.device_handlers.values(): 
                handler.stop() 
                    
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
    
    # TODO: Handle data streaming 
    def handle_data(self): 
        # Sends data received from device handlers to clients
        while self.status == ServerStatus.STREAMING: 
            pass
    
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