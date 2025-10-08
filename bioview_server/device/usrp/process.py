import queue
from typing import List 

import numpy as np

from bioview_common import log_print, PausableWorker
from bioview_server.utils import apply_filter, get_filter


class ProcessWorker(PausableWorker):
    def __init__(
        self,
        data_sources,
        samp_rate,
        channel_ifs,
        if_filter_bw,
        rx_queues: List[queue.Queue],
        save_queue: queue.Queue = None,
        display_queue: queue.Queue = None,
        save_ds: int = 1, 
        save_imaginary: bool = False, 
        save_iq: bool = False,
        display_imaginary: bool = False,
        logger = None
    ):
        super().__init__()

        # Signal
        self.logger = logger

        self.data_sources = data_sources.copy()

        # Variables
        self.samp_rate = samp_rate
        self.channel_ifs = channel_ifs
        self.save_ds = save_ds
        self.save_imaginary = save_imaginary
        self.save_iq = save_iq
        self.display_imaginary = display_imaginary

        # Queues
        self.rx_queues = rx_queues
        self.save_queue = save_queue
        self.display_queue = display_queue

        # State
        self.running = False

        # Load IF filters
        self.if_filts = [
            self._load_filter(freq, if_filter_bw[idx])
            for idx, freq in enumerate(channel_ifs)
        ]

        # Keep track of phase and filter states for all data sources
        for source in self.data_sources.values():
            source.accumulated_phase = 0.0
            source.filter_state = None

    def _load_filter(self, freq: float, bandwidth: float, order: int = 2):
        low_cutoff = freq - bandwidth / 2
        high_cutoff = freq + bandwidth / 2

        filter = get_filter(
            bounds=[low_cutoff, high_cutoff],
            samp_rate=self.samp_rate,
            btype="band",
            order=order,
        )
        return filter

    def _process_chunk(self, data, source, filter, if_freq):
        # Early return for empty data
        if len(data) == 0:
            return np.array([]), np.array([])

        # Store last sample for continuity checking
        if hasattr(source, "last_samples"):
            # Check for significant discontinuity
            discontinuity = abs(data[0] - source.last_samples)
            if discontinuity > 3 * np.std(data[: min(100, len(data))]):
                log_print(
                    self.logger,
                    "debug", f"[USRP] Potential discontinuity detected in {source.channel}"
                )

        # Store last sample for next buffer
        if not hasattr(source, "last_samples"):
            source.last_samples = data[-1]

        # Stateful filtering
        current_filter_state = source.filter_state
        filt_data, new_filter_state = apply_filter(data, filter, zi=current_filter_state)
        source.filter_state = new_filter_state

        # Get the current accumulated phase for this channel
        current_phase = source.accumulated_phase

        # Get phase for all samples
        phase_increment = 2 * np.pi * if_freq / self.samp_rate
        phases = current_phase + np.arange(len(filt_data)) * phase_increment

        # Down-convert from IF to baseband with phase continuity
        downconversion = np.exp(-1j * phases)
        baseband_data = filt_data * downconversion

        # Update phase accumulator for next buffer (mod 2π to prevent numerical drift)
        source.accumulated_phase = phases[-1] + phase_increment

        # Downsampling logic
        step = self.save_ds
        end_idx = len(baseband_data) - step + 1
        num_windows = (end_idx + step - 1) // step  # Calculate the number of windows

        if num_windows <= 0:
            return np.array([]), np.array([])

        # Create indices for the start of each window
        start_indices = np.arange(0, end_idx, step)

        # Use advanced indexing to get all the windows
        window_indices = start_indices[:, np.newaxis] + np.arange(step)
        windows = baseband_data[window_indices]

        if self.save_iq:
            first_comp = np.mean(np.real(windows), axis=1)
            second_comp = np.mean(np.imag(windows), axis=1)
        else:
            first_comp = np.mean(np.abs(windows), axis=1)
            second_comp = np.mean(np.angle(windows, True), axis=1)

        return first_comp, second_comp

    def _process_save(self, buffer):
        # Use numpy preallocated array for speed
        num_sources = len(self.data_sources)
        len_samples = int(buffer.shape[1] // self.save_ds)

        # We might not always want to save imaginary components
        if self.save_imaginary:
            save_list = np.empty((num_sources, len_samples, 2))
        else:
            save_list = np.empty((num_sources, len_samples))

        for source in self.data_sources.values():
            data = buffer[source.rx_idx, :]
            first_comp, second_comp = self._process_chunk(
                data=data,
                source=source,
                filter=self.if_filts[source.tx_idx],
                if_freq=self.channel_ifs[source.tx_idx],
            )

            log_print(self.logger, "debug", f"[USRP] Processed channel {source.channel}")

            if self.save_imaginary:
                save_list[source.channel, :, 0] = first_comp
                save_list[source.channel, :, 1] = second_comp
            else:
                save_list[source.channel, :] = first_comp

        # for r_idx, row in enumerate(self.channel_mapping):
        #     x = buffer[r_idx, :]
        #     for t_idx, channel_key in enumerate(row):
        #         channel_idx = self.data_mapping[channel_key]
        #         source = self.data_sources[]
        #         # Pass the channel key for state tracking
        #         first_comp, second_comp = self._process_chunk(
        #             data=x,
        #             source=source,
        #             filter=self.if_filts[t_idx],
        #             if_freq=self.channel_ifs[t_idx],
        #             channel_key=channel_key,
        #         )

        #         if self.config.get_param('save_imaginary'):
        #             save_list[channel_idx, :, 0] = first_comp
        #             save_list[channel_idx, :, 1] = second_comp
        #         else:
        #             save_list[channel_idx, :] = first_comp

        # Return all processed samples
        return save_list
    
    def work(self):
        # Preallocate empty buffer to get faster performance
        data_buf = None
        samples = [None] * len(self.rx_queues)

        while self.is_running:
            try:
                # Get from all queues
                for idx, key in enumerate(self.rx_queues):
                    rx_q = self.rx_queues[rx_q]
                    samples[idx] = rx_q.get()
                data_buf = np.transpose(np.vstack(samples))

                buffer_data = np.transpose(np.vstack(data_buf))
                processed = self._process_save(buffer_data)

                # Add to save queue 
                if self.save_queue is not None:
                    try:
                        self.save_queue.put(processed)
                        log_print(self.logger, "debug", "[USRP] Added to save queue")
                    except queue.Full:
                        log_print(self.logger, "debug", "[USRP] Save Queue Full")

                # Add to display queue
                try:
                    # If we do not save imaginary components, pass processed data
                    if self.save_imaginary is False:
                        self.display_queue.put(processed)
                    else:
                        # Depending on whether we want to display imaginary or not
                        if self.display_imaginary:
                            self.display_queue.put(processed[:, :, 1])
                        else:
                            self.display_queue.put(processed[:, :, 0])
                        log_print(self.logger, "debug", "[USRP] Added to display queue")
                except queue.Full:
                    log_print(self.logger, "debug", "[USRP] Display Queue Full")

            except queue.Empty:
                log_print(self.logger, "debug", "[USRP] Rx Queue Empty")
            except queue.Full:
                log_print(self.logger, "debug", "[USRP] Rx Queue Full")
            except Exception as e:
                log_print(self.logger, "error", f"[USRP] Processing error: {e}")

        log_print(self.logger, "debug", "[USRP] Processing stopped")

    # TODO: Confirm if we need any cleanup