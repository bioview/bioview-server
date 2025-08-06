'''
A collection of commands that can be sent to and responses that can be received from backend processes. 
'''
import time 

class Message:
    def __init__(self, msg_type, value=None, id=None):
        self.msg_type = msg_type
        self.value = value
        self.id = id or int(time.time() * 1000)