from enum import Enum


# Handle connection as an enum for better clarity
class ConnectionStatus(Enum):
    DISCONNECTED = "Not Connected"
    CONNECTING = "Connecting"
    CONNECTED = "Connected"


class RunningStatus(Enum):
    NOINIT = False
    RUNNING = True
    STOPPED = False