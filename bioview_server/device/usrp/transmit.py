import math
from threading import Thread
import numpy as np
import uhd

from .defaults import INIT_DELAY
from .config import UsrpConfiguration

from bioview_server.utils import emit_signal

class TransmitWorker(Thread):
    def __init__(
        self,
        config: UsrpConfiguration,
        usrp,
        tx_streamer,
        running: bool = True
    ):
        super().__init__()
        # Signals 
        self.log_event = None 
        
        # Modifiable params
        self.tx_gain = config.get_param("tx_gain").copy()
        self.tx_amplitude = config.get_param("tx_amplitude")

        # Fixed params
        self.samp_rate = config.get_param("samp_rate")
        self.if_freq = config.get_param("if_freq")
        self.tx_channels = config.get_param("tx_channels")

        self._generate_tx_waveforms()

        self.config = config
        self.usrp = usrp
        self.tx_streamer = tx_streamer
        self.running = running
        self.tx_buffer_size = self.tx_streamer.get_max_num_samps()

    def _get_buf_size(self, freq):
        return self.samp_rate * freq / (math.gcd(int(self.samp_rate), int(freq)) ** 2)

    def _get_lcm(self, a, b):
        return int(a * b / math.gcd(int(a), int(b)))

    def _generate_tx_waveforms(self):
        """
        Generate sine waves for each Tx channel, using as minimum a buffer size as possible.
        The buffer is made larger in length to be able to read circularly without causing overflow issues
        """

        if len(self.if_freq) == 1:
            self.tx_waveform_size = self._get_buf_size(self.if_freq[0])
        else:
            # Return the least common multiple
            self.tx_waveform_size = self._get_lcm(
                self._get_buf_size(self.if_freq[0]), self._get_buf_size(self.if_freq[1])
            )

        len_buf = 20 * self.tx_waveform_size

        self.tx_waveform = np.zeros(
            (len(self.tx_channels), len_buf), dtype=np.complex64
        )

        # Generate IQ Modulated IF signals
        for idx, _ in enumerate(self.tx_channels):
            self.tx_waveform[idx] = uhd.dsp.signals.get_continuous_tone(
                self.samp_rate,
                self.if_freq[idx],
                self.tx_amplitude[idx],
                desired_size=len_buf,
                max_size=(2 * self.samp_rate),
                waveform="sine",
            )

    def run(self):
        emit_signal(self.log_event, "debug", "Transmission Started")
        tx_metadata = uhd.types.TXMetadata()
        tx_metadata.start_of_burst = True
        tx_metadata.end_of_burst = False
        tx_metadata.has_time_spec = True
        tx_metadata.time_spec = uhd.types.TimeSpec(
            self.usrp.get_time_now().get_real_secs() + INIT_DELAY
        )

        while self.running:
            # Check for updated parameters
            curr_tx_gain = self.config.get_param("tx_gain")
            if curr_tx_gain != self.tx_gain:
                for chan in self.config.tx_channels:
                    self.usrp.set_tx_gain(curr_tx_gain[chan], chan)
                emit_signal(self.log_event, "debug", f"Tx gain updated to {curr_tx_gain}. Current {self.tx_gain}")
                self.tx_gain = curr_tx_gain

            try:
                # Send samples
                buffer_iter = self.tx_waveform
                num_samps = self.tx_streamer.send(buffer_iter, tx_metadata)
            except RuntimeError as ex:
                emit_signal(self.log_event, "error", f"Runtime error in transmit: {ex}")
                continue

            # Continue transmission
            tx_metadata.start_of_burst = False
            tx_metadata.has_time_spec = False

            if num_samps < self.tx_buffer_size:
                emit_signal(self.log_event, "warning", f"Tx Sent only {num_samps} samples")

        # End transmission
        tx_metadata.end_of_burst = True
        self.tx_streamer.send(np.zeros_like(self.tx_waveform), tx_metadata)
        emit_signal(self.log_event, "debug", "Transmission Stopped")

    def stop(self):
        self.running = False
