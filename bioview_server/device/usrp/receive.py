import queue

import numpy as np


try:
    import uhd
except ImportError:  # pragma: no cover
    uhd = None
from bioview_common import QUEUE_PUT_TIMEOUT_S, PausableWorker, log_print, put_or_drop


INIT_DELAY = 0.05
SAVE_BUFFER_SIZE = 20


class ReceiveWorker(PausableWorker):
    def __init__(
        self,
        usrp,
        rx_gain: list[float],
        rx_channels: list[int],
        rx_streamer,
        rx_queue: queue.Queue,
        cmd_queue: queue.Queue,
        global_rx_offset: int = 0,
        running: bool = False,
        logger=None,
    ):
        super().__init__()

        self.logger = logger

        self.rx_gain = rx_gain
        self.rx_channels = rx_channels
        self.global_rx_offset = global_rx_offset

        self.usrp = usrp
        self.rx_streamer = rx_streamer
        self.rx_queue = rx_queue
        self.cmd_queue = cmd_queue

        self.running = running

        self.buffers_dropped = 0
        self._last_drop_logged = 0

    def work(self):
        log_print(self.logger, "debug", "Receiving Started")
        if self.usrp is None or self.rx_streamer is None:
            log_print(self.logger, "error", "USRP or Rx streamer not initialized.")
            return

        rx_metadata = uhd.types.RXMetadata()

        num_channels = self.rx_streamer.get_num_channels()
        max_samps_per_packet = self.rx_streamer.get_max_num_samps()

        recv_buffer_size = max_samps_per_packet * SAVE_BUFFER_SIZE

        recv_buffer = np.empty((num_channels, recv_buffer_size), dtype=np.complex64)

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)

        stream_cmd.stream_now = False
        stream_cmd.time_spec = uhd.types.TimeSpec(
            self.usrp.get_time_now().get_real_secs() + INIT_DELAY
        )
        self.rx_streamer.issue_stream_cmd(stream_cmd)

        total_samps_received = 0
        timeout = 0.5
        had_an_overflow = False
        last_overflow = uhd.types.TimeSpec(0)

        num_rx_samps = 0
        num_rx_dropped = 0

        rate = self.usrp.get_rx_rate()

        while self.is_running:
            try:
                current_command = self.cmd_queue.get_nowait()

                param = current_command["param"]
                val = current_command["value"]

                if param == "rx_gain":
                    gains = val if isinstance(val, list) else [val]
                    local_gains = gains[
                        self.global_rx_offset : self.global_rx_offset
                        + len(self.rx_channels)
                    ]
                    if local_gains != self.rx_gain:
                        for idx, chan in enumerate(self.rx_channels):
                            self.usrp.set_rx_gain(local_gains[idx], chan)

                    log_print(
                        self.logger,
                        "debug",
                        f"Rx gain updated to {local_gains}. Current {self.rx_gain}",
                    )
                    self.rx_gain = local_gains
                else:
                    pass

            except queue.Empty:
                pass

            try:
                num_rx_samps = self.rx_streamer.recv(recv_buffer, rx_metadata, timeout)
            except RuntimeError as ex:
                log_print(self.logger, "error", f"Receiver Runtime Eror: {ex}")
                continue

            timeout = INIT_DELAY

            if rx_metadata.error_code == uhd.types.RXMetadataErrorCode.none:
                if had_an_overflow:
                    had_an_overflow = False
                    num_rx_dropped += (rx_metadata.time_spec - last_overflow).to_ticks(
                        rate
                    )
            elif rx_metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                had_an_overflow = True
                last_overflow = uhd.types.TimeSpec(
                    rx_metadata.time_spec.get_full_secs(),
                    rx_metadata.time_spec.get_frac_secs(),
                )
                log_print(
                    self.logger,
                    "warning",
                    f"Receiver Overflow: {rx_metadata.strerror()}",
                )
            elif rx_metadata.error_code == uhd.types.RXMetadataErrorCode.late:
                log_print(
                    self.logger,
                    "warning",
                    f"Receiver Late: {rx_metadata.strerror()}, restarting...",
                )
                stream_cmd.time_spec = uhd.types.TimeSpec(
                    self.usrp.get_time_now().get_real_secs() + INIT_DELAY
                )
                stream_cmd.stream_now = num_channels == 1
                self.rx_streamer.issue_stream_cmd(stream_cmd)
            elif rx_metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
                log_print(
                    self.logger,
                    "warning",
                    f"Receiver Timeout: {rx_metadata.strerror()}",
                )
            else:
                log_print(
                    self.logger,
                    "warning",
                    f"Receiver Error: {rx_metadata.strerror()}",
                )

            total_samps_received += num_rx_samps

            if not put_or_drop(
                self.rx_queue, recv_buffer.copy(), timeout=QUEUE_PUT_TIMEOUT_S
            ):
                self.buffers_dropped += 1
                if self.buffers_dropped - self._last_drop_logged >= 20:
                    self._last_drop_logged = self.buffers_dropped
                    log_print(
                        self.logger,
                        "warning",
                        f"Rx queue full; {self.buffers_dropped} buffers dropped "
                        "(demodulation is not keeping up)",
                    )

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)
        log_print(self.logger, "debug", "Receiving Stopped")

    def cleanup(self):
        if self.rx_streamer is not None:
            try:
                stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                self.rx_streamer.issue_stream_cmd(stream_cmd)
                log_print(self.logger, "debug", "Receiving stopped cleanly")
            except Exception as ex:
                log_print(self.logger, "error", f"Error stopping receiving: {ex}")
