from ctypes import byref, c_double, c_int
from bioview_server.utils import emit_signal
from .utils import load_mpdev_dll, wrap_result_code


class ConnectWorker:
    def __init__(self, config):
        super().__init__()
        # Signals
        self.init_succeeded = None 
        self.log_event = None 
        self.init_failed = None 
        
        # Variables
        self.biopac = None

    def run(self):
        # Load the DLL
        self.biopac = load_mpdev_dll(self.config.mpdev_path)
        if self.biopac is None:
            emit_signal(self.log_event, "error", "mpdev.dll was not found. Ensure it is available in either the system path or the custom path provided")
            return

        # Connect to the device - Currently, the types values are hard-coded but they needed to be taken from the device name
        try:
            wrap_result_code(
                self.biopac.connectMPDev(c_int(103), c_int(10), b"auto"),
                "Initialization",
            )

            # Set channels
            if sum(self.config.channels) == 0:
                raise ValueError("At least one channel must be active")
            channels_array = (c_int * 16)(*self.config.get_channels())
            wrap_result_code(
                self.biopac.setAcqChannels(byref(channels_array)), "Set Channels"
            )

            # Set sample rate
            wrap_result_code(
                self.biopac.setSampleRate(c_double(self.config.get_samp_time())),
                "Set Sample Rate",
            )

            # Emit device to main app
            emit_signal(self.init_succeeded, self.biopac)
        except Exception as e:
            emit_signal(self.init_failed, f"Unable to initialize device: {e}")

    def stop(self):
        # Cleanup device handler
        try:
            wrap_result_code(self.biopac.disconnectMPDev())
        except Exception:
            emit_signal(self.log_event, "info", "BIOPAC connection already closed")
