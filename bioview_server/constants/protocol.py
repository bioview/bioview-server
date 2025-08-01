'''
Declares commonly supported commands that may be supported wholly or in part by different servers and clients 
'''
from enum import Enum 

MAX_BUFFER_SIZE = 4096

# Command from client to server 
class Command(Enum): 
    PING = 'ping'
    DISCOVER = 'discover_device'
    INIT = 'init_device'
    CONNECT = 'connect_device'
    DISCONNECT = 'disconnect_device'
    CONFIGURE = 'configure_device'
    UPDATE = 'update_param'    
    START = 'start_streaming'
    STOP = 'stop_streaming'
    STATUS = 'get_status'
    SHUTDOWN = 'shutdown'
 
# Response from server to client   
class Response(Enum): 
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"

'''
A collection of commands that can be sent to and responses that can be received from backend processes. 
'''
import time 
from enum import Enum 

class Message:
    def __init__(self, msg_type, value=None, id=None):
        self.msg_type = msg_type
        self.value = value
        self.id = id or int(time.time() * 1000)