import queue
from threading import Thread
from typing import Callable, List

import numpy as np
import uhd

from bioview_common import log_print

INIT_DELAY = 0.05  # 50mS initial delay before transmit
# This is a good balance between real time display and spikes
SAVE_BUFFER_SIZE = 20


class ReceiveWorker(Thread):
    def __init__(
        self,
        usrp,
        rx_gain: List[float],
        rx_channels: List[int],
        rx_streamer,
        rx_queue: queue.Queue,
        cmd_queue: queue.Queue,
        running: bool = False,
        logger = None
    ):
        super().__init__()

        self.daemon = True

        # Signals
        self.logger = logger

        # Modifiable params
        self.rx_gain = rx_gain
        self.rx_channels = rx_channels

        # Device params
        self.usrp = usrp
        self.rx_streamer = rx_streamer
        self.rx_queue = rx_queue  # Data
        self.cmd_queue = cmd_queue  # Commands (such as gain change)

        self.running = running

    def run(self):
        self.running = True

        log_print(self.logger, "debug", "Receiving Started")
        if self.usrp is None or self.rx_streamer is None:
            log_print(self.logger, "error", "USRP or Rx streamer not initialized.")
            return

        rx_metadata = uhd.types.RXMetadata()

        # Buffer for receiving samples
        num_channels = self.rx_streamer.get_num_channels()
        max_samps_per_packet = self.rx_streamer.get_max_num_samps()

        # Make receive buffer larger than max_samps_per_packet.
        # This adds a latency of recv_buffer_size / sample_rate (in seconds)
        recv_buffer_size = max_samps_per_packet * SAVE_BUFFER_SIZE

        recv_buffer = np.empty((num_channels, recv_buffer_size), dtype=np.complex64)

        # Setup streaming using continuous saving mode by default
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)

        # When using multiple devices, we need to set stream_now to False
        # to align time edges of packets
        stream_cmd.stream_now = False
        stream_cmd.time_spec = uhd.types.TimeSpec(
            self.usrp.get_time_now().get_real_secs() + INIT_DELAY
        )
        self.rx_streamer.issue_stream_cmd(stream_cmd)

        # Initialize
        total_samps_received = 0
        timeout = 0.5  # Larger timeout initially
        had_an_overflow = False
        last_overflow = uhd.types.TimeSpec(0)

        # Setup the statistic counters
        num_rx_samps = 0
        num_rx_dropped = 0

        rate = self.usrp.get_rx_rate()

        while self.running:
            # Check for updated parameters
            try:
                current_command = self.cmd_queue.get()

                # Make changes to adjustable params
                param = current_command["param"]
                val = current_command["value"]

                if param == "rx_gain":
                    if val != self.rx_gain:
                        for chan in self.rx_channels:
                            self.usrp.set_rx_gain(val[chan], chan)

                    log_print(
                        self.logger, "debug",
                        f"Rx gain updated to {val}. Current {self.rx_gain}",
                    )
                    self.rx_gain = val
                # NOTE: Any other modifiable parameters may be added here
                else:
                    pass

            except queue.Empty:
                pass

            try:
                # Receive samples
                num_rx_samps = self.rx_streamer.recv(recv_buffer, rx_metadata, timeout)
            except RuntimeError as ex:
                log_print(self.logger, "error", f"Receiver Runtime Eror: {ex}")
                continue

            timeout = INIT_DELAY  # Reduce timeout for subsequent transmissions

            # Reference: uhd/examples/python/benchmark_rate.py
            # Handle the error codes
            if rx_metadata.error_code == uhd.types.RXMetadataErrorCode.none:
                # Reset the overflow flag
                if had_an_overflow:
                    had_an_overflow = False
                    num_rx_dropped += (rx_metadata.time_spec - last_overflow).to_ticks(
                        rate
                    )
            elif rx_metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                had_an_overflow = True
                # Need to make sure that last_overflow is a new TimeSpec object, not
                # a reference to metadata.time_spec, or it would not be useful
                # further up.
                last_overflow = uhd.types.TimeSpec(
                    rx_metadata.time_spec.get_full_secs(),
                    rx_metadata.time_spec.get_frac_secs(),
                )
                log_print(
                    self.logger, "warning",
                    f"Receiver Overflow: {rx_metadata.strerror()}",
                )
            elif rx_metadata.error_code == uhd.types.RXMetadataErrorCode.late:
                log_print(
                    self.logger, "warning",
                    f"Receiver Late: {rx_metadata.strerror()}, restarting...",
                )
                # Radio core will be in the idle state.
                # Issue stream command to restart streaming.
                stream_cmd.time_spec = uhd.types.TimeSpec(
                    self.usrp.get_time_now().get_real_secs() + INIT_DELAY
                )
                stream_cmd.stream_now = num_channels == 1
                self.rx_streamer.issue_stream_cmd(stream_cmd)
            elif rx_metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
                log_print(
                    self.logger, "warning",
                    f"Receiver Timeout: {rx_metadata.strerror()}",
                )
            else:
                log_print(
                    self.logger, "warning",
                    f"Receiver Error: {rx_metadata.strerror()}",
                )

            total_samps_received += num_rx_samps

            # Copy samples to avoid buffer overwrite and put in queue
            # recv_buffer.dtype = np.complex64 (since default cpu_format = 'fc32')
            try:
                self.rx_queue.put(recv_buffer)
            except queue.Full:
                log_print(self.logger, "warning", "Rx Queue full, dropping buffer")
            except queue.Empty:
                log_print(self.logger, "debug", "Rx Queue Empty")
                continue

        # Gracefully close once receiving is finished
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        log_print(self.logger, "debug", "Receiving Stopped")

    def stop(self):
        self.running = False
