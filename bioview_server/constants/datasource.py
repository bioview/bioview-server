# This represents a custom data source for the plot
class DataSource:
    def __init__(self, device, channel, label):
        self.device = device  # Keep track of device handler
        self.channel = channel
        self.label = label  # Human-readable label for the source

    def __eq__(self, other):
        return self.device == other.device and self.channel == other.channel

    def __hash__(self):
        # Hash based on the same attributes used in __eq__
        # Use id() for device since device objects might not be hashable
        # Use the channel string directly since it should be hashable
        return hash((id(self.device), self.channel))

    def __repr__(self):
        # String display
        return f"{self.device.device_name}: {self.label}"

    def get_disp_freq(self):
        return self.device.get_disp_freq()
