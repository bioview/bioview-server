class Backend():
    def __init__(self):
        pass

    def discover_devices(self):
        # This returns device details provided by firmware, including identifiers that will allow us to make updates to firmware
        raise NotImplementedError
    
    def initialize_device(self):
        # Given a configuration, initialize and return a device object
        raise NotImplementedError

    def update_firmware(self):
        raise NotImplementedError