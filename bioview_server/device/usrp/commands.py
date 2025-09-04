"""
Parses commands received by USRPBackend and forwards them
to the correct sub-component. The following components
are currently supported -
1. ReceiveWorker
2. TransmitWorker
3. DisplayWorker
"""


class CommandHandler:
    def __init__(
        self,
        rx_command_queue,
        tx_command_queue,
    ):
        pass
