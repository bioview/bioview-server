import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from bioview_common import log_print, PausableWorker
from bioview_common import apply_filter, get_filter
from bioview_common.signal_schemes import (
    FmcwScheme,
    PulsedDopplerScheme,
    differential_phase,
    normalized_amplitude,
)


class ProcessWorker(PausableWorker):
    def __init__(
        self,
        data_sources,
        cal_ref_sources,
        samp_rate,
        channel_ifs,
        if_filter_bw,
        rx_queues: Dict,
        rx_device_order: List[str],
        schemes_by_device: Dict,
        global_tx_to_device: Dict,
        signal_scheme: str = "cw",
        fmcw_scheme: Optional[FmcwScheme] = None,
        save_queue: queue.Queue = None,
        display_queue: queue.Queue = None,
        save_ds: int = 1,
        save_imaginary: bool = False,
        save_iq: bool = False,
        display_imaginary: bool = False,
        record_cal_ref: bool = True,
        logger=None,
    ):
        super().__init__()

        self.logger = logger
        self.mimo_sources = sorted(data_sources, key=lambda s: s.channel)
        self.cal_ref_sources = sorted(cal_ref_sources or [], key=lambda s: s.channel)
        self.all_sources = self.mimo_sources + self.cal_ref_sources
        self.data_sources = self.all_sources

        self.samp_rate = samp_rate
        self.channel_ifs = channel_ifs
        self.save_ds = save_ds
        self.save_imaginary = save_imaginary
        self.save_iq = save_iq
        self.display_imaginary = display_imaginary
        self.record_cal_ref = record_cal_ref
        self.signal_scheme = signal_scheme
        self.fmcw_scheme = fmcw_scheme

        self.rx_queues = rx_queues
        self.rx_device_order = rx_device_order
        self.schemes_by_device = schemes_by_device
        self.global_tx_to_device = global_tx_to_device

        self.save_queue = save_queue
        self.display_queue = display_queue

        self.global_sample_idx = 0
        self.latest_metrics = {}
        self._metrics_lock = threading.Lock()
        self._partial_rows = {}

        num_tx = len(channel_ifs)
        self.if_filts = [
            self._load_filter(channel_ifs[idx], if_filter_bw[idx])
            for idx in range(num_tx)
        ]

        for source in self.mimo_sources:
            source.accumulated_phase = 0.0
            source.filter_state = None
            source.prev_phase = None
            source.accumulated_sample_idx = 0

    def _load_filter(self, freq: float, bandwidth: float, order: int = 2):
        low_cutoff = freq - bandwidth / 2
        high_cutoff = freq + bandwidth / 2
        return get_filter(
            bounds=[low_cutoff, high_cutoff],
            samp_rate=self.samp_rate,
            btype="band",
            order=order,
        )

    def get_metric(self, measure_tx: int, target_rx: int) -> Optional[float]:
        key = (measure_tx, target_rx)
        with self._metrics_lock:
            return self.latest_metrics.get(key)

    def _process_chunk(self, data, source, filt, if_freq, scheme):
        if len(data) == 0:
            return np.array([]), np.array([])

        current_filter_state = source.filter_state
        filt_data, new_filter_state = apply_filter(
            data, filt, zi=current_filter_state
        )
        source.filter_state = new_filter_state

        current_phase = source.accumulated_phase
        phase_increment = 2 * np.pi * if_freq / self.samp_rate
        phases = current_phase + np.arange(len(filt_data)) * phase_increment

        downconversion = np.exp(-1j * phases)
        baseband_data = filt_data * downconversion

        if self.signal_scheme == "fmcw" and self.fmcw_scheme is not None:
            ref = self.fmcw_scheme.get_dechirp_reference(
                len(filt_data), source.accumulated_sample_idx
            )
            baseband_data = baseband_data * ref

        source.accumulated_phase = phases[-1] + phase_increment
        source.accumulated_sample_idx += len(filt_data)

        step = self.save_ds
        end_idx = len(baseband_data) - step + 1
        num_windows = (end_idx + step - 1) // step
        if num_windows <= 0:
            return np.array([]), np.array([])

        start_indices = np.arange(0, end_idx, step)
        window_indices = start_indices[:, np.newaxis] + np.arange(step)
        windows = baseband_data[window_indices]

        dev_name, local_tx = self.global_tx_to_device.get(
            source.tx_idx, (None, source.tx_idx)
        )
        dev_scheme = self.schemes_by_device.get(dev_name) if dev_name else scheme
        local_tx_idx = local_tx if dev_name else source.tx_idx
        active_scheme = dev_scheme if dev_scheme is not None else scheme
        tx_amp = (
            active_scheme.get_tx_amplitude(local_tx_idx) if active_scheme else 1.0
        )

        if self.save_iq:
            first_comp = np.mean(np.real(windows), axis=1)
            second_comp = np.mean(np.imag(windows), axis=1)
        else:
            amps = []
            phases_out = []
            for w_i, win in enumerate(windows):
                center_idx = source.accumulated_sample_idx - len(baseband_data) + int(
                    start_indices[w_i] + step // 2
                )
                if active_scheme:
                    tx_phase = active_scheme.tx_phase_at(local_tx_idx, center_idx)
                else:
                    tx_phase = 0.0
                amp = normalized_amplitude(win, tx_amp)
                ph, source.prev_phase = differential_phase(
                    win, tx_phase, source.prev_phase
                )
                amps.append(amp)
                phases_out.append(ph)
            first_comp = np.array(amps)
            second_comp = np.array(phases_out)

        return first_comp, second_comp

    def _decimate_cal_ref(self, envelope: np.ndarray) -> np.ndarray:
        step = self.save_ds
        end_idx = len(envelope) - step + 1
        if end_idx <= 0:
            return np.array([])
        start_indices = np.arange(0, end_idx, step)
        window_indices = start_indices[:, np.newaxis] + np.arange(step)
        windows = envelope[window_indices]
        return np.mean(windows, axis=1)

    def _process_mimo_chunk(self, buffer):
        """Process all MIMO sources once; update metrics and return per-source comps."""
        results = {}
        for source in self.mimo_sources:
            data = buffer[source.rx_idx, :]
            dev_name, _ = self.global_tx_to_device.get(
                source.tx_idx, (self.rx_device_order[0], source.tx_idx)
            )
            scheme = self.schemes_by_device.get(dev_name)
            first_comp, second_comp = self._process_chunk(
                data=data,
                source=source,
                filt=self.if_filts[source.tx_idx],
                if_freq=self.channel_ifs[source.tx_idx],
                scheme=scheme,
            )
            results[source.channel] = (first_comp, second_comp)
            if len(first_comp) > 0:
                with self._metrics_lock:
                    self.latest_metrics[(source.tx_idx, source.rx_idx)] = float(
                        np.mean(first_comp)
                    )
        return results

    def _assemble_outputs(self, buffer, mimo_results):
        num_mimo = len(self.mimo_sources)
        num_cal = len(self.cal_ref_sources) if self.record_cal_ref else 0
        len_samples = int(buffer.shape[1] // self.save_ds)

        if self.save_imaginary:
            save_list = np.empty((num_mimo + num_cal, len_samples, 2))
            display_list = np.empty((num_mimo, len_samples, 2))
        else:
            save_list = np.empty((num_mimo + num_cal, len_samples))
            display_list = np.empty((num_mimo, len_samples))

        for source in self.mimo_sources:
            first_comp, second_comp = mimo_results[source.channel]
            if self.save_imaginary:
                save_list[source.channel, :, 0] = first_comp
                save_list[source.channel, :, 1] = second_comp
                display_list[source.channel, :, 0] = first_comp
                display_list[source.channel, :, 1] = second_comp
            else:
                save_list[source.channel, :] = first_comp
                display_list[source.channel, :] = first_comp

        if self.record_cal_ref:
            for source in self.cal_ref_sources:
                global_tx = source.tx_idx
                dev_name, local_tx = self.global_tx_to_device[global_tx]
                scheme = self.schemes_by_device[dev_name]
                raw_env = scheme.get_calibration_reference(
                    local_tx, self.global_sample_idx, buffer.shape[1]
                )
                cal_data = self._decimate_cal_ref(raw_env)
                if self.save_imaginary:
                    # Calibration reference is a real-valued envelope.
                    # Store it in channel 0 and keep the imaginary component at 0.
                    save_list[source.channel, : len(cal_data), 0] = cal_data
                    save_list[source.channel, : len(cal_data), 1] = 0.0
                else:
                    save_list[source.channel, : len(cal_data)] = cal_data

        self.global_sample_idx += buffer.shape[1]
        return save_list, display_list

    def work(self):
        while self.is_running:
            try:
                for key in self.rx_device_order:
                    if key not in self._partial_rows:
                        rx_q = self.rx_queues[key]
                        self._partial_rows[key] = rx_q.get_nowait()
                
                rows = [self._partial_rows[key].copy() for key in self.rx_device_order]
                buffer = np.vstack(rows)
                self._partial_rows.clear()

                mimo_results = self._process_mimo_chunk(buffer)
                save_data, display_data = self._assemble_outputs(buffer, mimo_results)

                if self.save_queue is not None:
                    try:
                        self.save_queue.put(save_data)
                    except queue.Full:
                        log_print(self.logger, "debug", "[USRP] Save Queue Full")

                if self.display_queue is not None:
                    try:
                        if self.save_imaginary is False:
                            self.display_queue.put(display_data)
                        else:
                            if self.display_imaginary:
                                self.display_queue.put(display_data[:, :, 1])
                            else:
                                self.display_queue.put(display_data[:, :, 0])
                    except queue.Full:
                        log_print(self.logger, "debug", "[USRP] Display Queue Full")

            except queue.Empty:
                time.sleep(0.001)
                # log_print(self.logger, "debug", "[USRP] Rx Queue Empty")
            except Exception as e:
                log_print(self.logger, "error", f"[USRP] Processing error: {e}")

        log_print(self.logger, "debug", "[USRP] Processing stopped")
