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
try:
    import uhd # Crashes occur without this
except ImportError: 
    print(f'uhd not available for import. Install if functionality is required.')

import socket
import json
import time
import sys
from threading import Thread, Lock
import multiprocessing as mp
import traceback

from bioview_server.device import discover_devices, get_device_object
from bioview_server.constants import APP_VERSION, Configuration, Command, Response, MAX_BUFFER_SIZE


SUPPORTED_COMMANDS = [
    Command.PING,
    Command.DISCOVER,
    Command.INIT, 
    Command.CONNECT,
    Command.DISCONNECT,
    Command.CONFIGURE, 
    Command.START,
    Command.STOP,
    Command.UPDATE, 
    Command.STATUS, 
    Command.SHUTDOWN
]

class Server:
    def __init__(self, address='0.0.0.0', control_port=8888, data_port=8889):
        self.address = address # This can be a list 
        self.control_port = control_port
        self.data_port = data_port
        
        # Sockets
        self.control_socket = None
        self.data_socket = None
        self.data_clients = []  # List of connected data clients
        self.data_lock = Lock()
        
        # Server state
        self.running = False
        self.is_streaming = False

        # Device state
        self.device_handlers = [] 
        self.data_queue = mp.Queue()
        
        
    def start(self):
        print(f'Starting server at {self.address}:{self.control_port} (Control), {self.address}:{self.data_port} (Data)')
        
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
            
            # Get and display network information
            local_ip = self.get_local_ip()

            if local_ip and self.address == '0.0.0.0':
                print(f"Local network address: {local_ip}")
                print(f"Clients should connect to: {local_ip}:{self.data_port},{self.control_port}")
            
        except Exception as e: 
            print(f'Error occurred while starting server: {e}')
        
        # Once client have started, start listening
        self.running = True
        try:
            # Start server threads
            control_thread = Thread(target=self.run_control_server, daemon=True)
            data_thread = Thread(target=self.run_data_server, daemon=True)
            
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
        print(f'Stopping server at {self.address}:{self.control_port} (Control), {self.address}:{self.data_port} (Data)')
        self.running = False 
        
        # Close sockets
        if self.control_socket:
            self.control_socket.close()
        if self.data_socket:
            self.data_socket.close()
            
        print("Client server stopped")

    def get_local_ip(self):
        """Get the local IP address"""
        try:
            # Connect to a remote address to determine local IP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
            return local_ip
        except Exception:
            return None
         
    def run_control_server(self):  
        while self.running:
            try:
                client_socket, address = self.control_socket.accept()
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
            
    def run_data_server(self): 
        """Run data streaming server"""
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
                        'message': f"Invalid JSON: {e}"
                    }
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    
        except Exception as e:
            print(f"Control client error: {e}")
        finally:
            client_socket.close()
    
    def process_command(self, command):
        """Process incoming commands"""
        cmd_type = command.get('type')
        
        # TODO: Add -> UPDATE = 'update_param'       
        try:
            if cmd_type == Command.PING.value:
                return self.handle_ping()
            elif cmd_type == Command.DISCOVER.value:
                return self.handle_discover_devices()
            elif cmd_type == Command.INIT.value: 
                return self.handle_init_device(command.get('params', {})) 
            elif cmd_type == Command.CONNECT.value:
                return self.handle_connect_device()
            elif cmd_type == Command.DISCONNECT.value:
                return self.handle_disconnect_device()
            elif cmd_type == Command.START.value:
                return self.handle_start_streaming()
            elif cmd_type == Command.STOP.value:
                return self.handle_stop_streaming()
            elif cmd_type == Command.STATUS.value:
                return self.handle_get_status()
            elif cmd_type == Command.CONFIGURE.value:
                return self.handle_update_device_config(command.get('params', {}))
            elif cmd_type == Command.UPDATE.value: 
                return self.handle_update_device_param(command.get('params', {}))
            elif cmd_type == Command.SHUTDOWN.value:
                return self.handle_shutdown()
            else:
                return {
                    'type': Response.ERROR.value,
                    'message': f"Unknown command: {cmd_type}"
                }
                
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'message': f"Command processing error: {e}",
                'traceback': traceback.format_exc()
            }
    
    def handle_ping(self):
        """Handle ping command"""
        return {
            'type': Response.SUCCESS.value,
            'message': 'pong',
            'server_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'devices': len(self.device_handlers),
                'is_streaming': self.is_streaming,
            }
        }
    
    def handle_discover_devices(self):
        '''
        For all available backends, this will try to discover devices
        ''' 
        
        print("🔍 Starting device discovery...")
        self.discovered_devices = [] 
        try: 
            self.discovered_devices = discover_devices()
        except Exception as e: 
            return {
                    'type': Response.ERROR.value,
                    'message': f'Device discovery failed: {e}',
                    'step': 'discovery'
                }    
        
        return {
                'type': Response.SUCCESS.value,
                'message': f'Found {len(self.discovered_devices)} devices',
                'devices': self.discovered_devices,
                'step': 'discovery'
            }
    
    def handle_connect_device(self):
        """Handle device connection"""
        try:
            for device in self.device_handlers.values(): 
                device.connect()
            
            print("✓ Devices connected")
            
            return {
                'type': Response.SUCCESS.value,
                'message': 'Connect successful'
            }
            
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'message': f'Connect error: {e}'
            }
    
    def handle_disconnect_device(self):
        """ Tell all devices to disconnect """
        try:
            for device in self.device_handlers.values(): 
                device.disconnect()
            
            print("✓ Devices disconnected")
            
            return {
                'type': Response.SUCCESS.value,
                'message': 'Disconnect successful'
            }
            
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'message': f'Disconnect error: {e}'
            }
    
    def handle_get_status(self):
        """Get current server status"""
        return {
            'type': Response.SUCCESS.value,
            'message': 'Status retrieved',
            'status': {
                'devices': self.discovered_devices,
                'is_streaming': self.is_streaming,
            }
        }
    
    def handle_init_device(self, params): 
        device_id = params['id']
        config = Configuration.from_json(params['config'])
        exp_config = Configuration.from_json(params['exp_config'])
        save = params.get('save', False)

        # Create device handler objects with provided config, regardless of whether a prior config existed
        try:
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
    
    def handle_shutdown(self):
        """Handle server shutdown"""
        print("🛑 Shutdown requested")
        
        # Disconnect device first
        self.handle_disconnect_device()
        
        # Schedule server stop
        def stop_server():
            time.sleep(0.5)
            self.stop()
        
        Thread(target=stop_server, daemon=True).start()
        
        return {
            'type': Response.SUCCESS.value,
            'message': 'Server shutting down'
        }
        
    def handle_start_streaming(self): 
        """Start real-time data streaming"""
        if len(self.device_handlers) == 0: 
            return {
                'type': Response.ERROR.value,
                'message': 'No device connected'
            }
        
        try:
            print("🚀 Starting data streaming...")
            
            # Start your existing receive/transmit workers
            for handler in self.device_handlers.values():
                handler.start()
            
            self.is_streaming = True 
            
            print("✓ Data streaming started")
            return {
                'type': Response.SUCCESS.value,
                'message': 'Data streaming started'
            }
            
        except Exception as e:
            self.is_streaming = False 
            return {
                'type': Response.ERROR.value,
                'message': f'Failed to start streaming: {e}',
                'traceback': traceback.format_exc()
            }
    
    def handle_stop_streaming(self): 
        """Stop data streaming"""
        try:
            if self.is_streaming: 
                print("🛑 Stopping data streaming...")
                for handler in self.device_handlers.values(): 
                    handler.stop() 
                    
            self.is_streaming = False 
            return {
                'type': Response.SUCCESS.value,
                'message': 'Data streaming stopped'
            }
        except Exception as e:
            return {
                'type': Response.ERROR.value,
                'message': f'Failed to stop streaming: {e}'
            }
    
    def handle_data(self): 
        # Sends data received from device handlers to clients
        while self.running and self.is_streaming: 
            pass
    

""" Device Handler
This is a standalone process which interfaces with devices of a particular kind. In order to isolate
hardware interfaces from each other, every unique kind of device is ru

The DeviceHandler() class maintains all necessary device info 
& can also read & write data to & from the device 
along with storing the read data in a data dump in storage

"""

class DeviceHandler(mp.Process):
    def __init__(self, config: Configuration, exp_config: Configuration, data_queue: mp.Queue, save):
        # Device configuration
        self.config = config 
        self.exp_config = exp_config
        self.save = save 
         
        self.device_name = config.get_param('device_name', 'dummy_device')
        self.device = None 
        self.data_queue = data_queue
        
        # Device status 
        self.is_connected = False 
        self.is_streaming = False 
        
        self.running = False 
        
    def connect(self):
        # Create device object 
        self.device = get_device_object(
            device_name = self.device_name, 
            config = self.config,
            data_queue = self.data_queue, 
            resp_queue=None, 
            save = self.save,
            exp_config = self.exp_config
        )
        
        self.device.connect()
    
    def start(self):
        self.device.run()
    
    def stop(self):
        self.device.stop()
        
    def disconnect(self):
        self.device.disconnect()
        
    def update_config(self, param, value): 
        self.device.update_config(param, value)    
    
    def update_param(self, param, value): 
        self.device.update_param(param, value)
    
if __name__ == "__main__":
    print("=" * 50)
    print(f"BioView Device Server, Version: {APP_VERSION}")
    print("=" * 50)
    
    server = Server()
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping...")
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
    finally:
        server.stop()