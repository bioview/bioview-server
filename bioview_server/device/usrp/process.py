import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from bioview_common import (
    QUEUE_PUT_TIMEOUT_S,
    PausableWorker,
    apply_filter,
    get_filter,
    log_print,
    put_drop_oldest,
    put_or_drop,
)
from bioview_common.signal_schemes import (
    FmcwScheme,
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
        # (measure_tx, measure_rx) -> (magnitude, phasor, seq). ``seq`` lets DPIC
        # balance wait for a measurement taken *after* it changed the injection,
        # rather than reading whatever value happens to be sitting there.
        self.latest_metrics = {}
        self._metrics_cv = threading.Condition()
        self._partial_rows = {}

        num_tx = len(channel_ifs)
        self.if_filts = [
            self._load_filter(channel_ifs[idx], if_filter_bw[idx])
            for idx in range(num_tx)
        ]

        self.save_chunks_dropped = 0
        self.display_chunks_dropped = 0
        self._last_drops_logged = 0

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

    def _log_drops(self):
        total = self.save_chunks_dropped + self.display_chunks_dropped
        if total - self._last_drops_logged >= 50:
            self._last_drops_logged = total
            log_print(
                self.logger,
                "warning",
                f"[USRP] Downstream queues full: {self.save_chunks_dropped} save / "
                f"{self.display_chunks_dropped} display chunks dropped",
            )

    def get_metric(self, measure_tx: int, measure_rx: int) -> Optional[float]:
        with self._metrics_cv:
            entry = self.latest_metrics.get((measure_tx, measure_rx))
        return entry[0] if entry else None

    def get_metric_complex(self, measure_tx: int, measure_rx: int) -> Optional[complex]:
        with self._metrics_cv:
            entry = self.latest_metrics.get((measure_tx, measure_rx))
        return entry[1] if entry else None

    def wait_for_metric(
        self,
        measure_tx: int,
        measure_rx: int,
        min_new: int = 2,
        timeout: float = 2.0,
    ) -> Optional[float]:
        """Block until ``min_new`` fresh chunks have been measured, then return.

        The Rx path buffers ~20 packets plus whatever is queued, so a value read
        immediately after a phase/amplitude change still describes the *old*
        setting. Waiting for new sequence numbers removes that stale-read bias
        from the DPIC search.
        """
        entry = self._wait_for_entry(measure_tx, measure_rx, min_new, timeout)
        return entry[0] if entry else None

    def wait_for_metric_complex(
        self,
        measure_tx: int,
        measure_rx: int,
        min_new: int = 2,
        timeout: float = 2.0,
    ) -> Optional[complex]:
        """Fresh complex residual phasor, for DPIC's closed-form solve."""
        entry = self._wait_for_entry(measure_tx, measure_rx, min_new, timeout)
        return entry[1] if entry else None

    def _wait_for_entry(self, measure_tx, measure_rx, min_new, timeout):
        key = (measure_tx, measure_rx)
        deadline = time.monotonic() + timeout
        with self._metrics_cv:
            entry = self.latest_metrics.get(key)
            target = (entry[2] if entry else -1) + min_new
            while True:
                entry = self.latest_metrics.get(key)
                if entry and entry[2] >= target:
                    return entry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return entry
                self._metrics_cv.wait(remaining)

    def _process_chunk(self, data, source, filt, if_freq, scheme):
        """Returns (first_comp, second_comp, metric).

        ``metric`` is always the mean baseband magnitude, normalized by the Tx
        amplitude. It must not be derived from ``first_comp``: under ``save_iq``
        that is mean(Re{.}), a signed quantity whose minimum is the most
        negative excursion rather than a null.
        """
        if len(data) == 0:
            return np.array([]), np.array([]), None

        current_filter_state = source.filter_state
        filt_data, new_filter_state = apply_filter(data, filt, zi=current_filter_state)
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
        num_windows = len(baseband_data) // step
        if num_windows <= 0:
            return np.array([]), np.array([]), None

        # Save windows are contiguous, non-overlapping blocks of ``step``
        # samples, so a reshape is an exact substitute for the old fancy-index
        # gather -- and it is a view rather than an (n_windows, step) copy.
        usable = num_windows * step
        windows = baseband_data[:usable].reshape(num_windows, step)

        dev_name, local_tx = self.global_tx_to_device.get(
            source.tx_idx, (None, source.tx_idx)
        )
        dev_scheme = self.schemes_by_device.get(dev_name) if dev_name else scheme
        local_tx_idx = local_tx if dev_name else source.tx_idx
        active_scheme = dev_scheme if dev_scheme is not None else scheme
        tx_amp = active_scheme.get_tx_amplitude(local_tx_idx) if active_scheme else 1.0

        if self.save_iq:
            first_comp = np.mean(np.real(windows), axis=1)
            second_comp = np.mean(np.imag(windows), axis=1)
        else:
            # Amplitude: one reduction over the whole (n_windows, step) block.
            first_comp = np.abs(windows).mean(axis=1)
            if tx_amp > 0:
                first_comp = first_comp / tx_amp

            # Phase: unwrap the chunk once, carrying continuity in from the
            # previous chunk, then reduce per window. Chaining np.unwrap per
            # window (the old loop) produces exactly this, one window at a time.
            angles = np.angle(baseband_data[:usable])
            if source.prev_phase is None:
                unwrapped = np.unwrap(angles)
            else:
                unwrapped = np.unwrap(np.concatenate(([source.prev_phase], angles)))[1:]
            source.prev_phase = float(unwrapped[-1])

            # Static Tx phase only. tx_phase_at() also carries the
            # 2*pi*f_if*n/fs ramp that the downconversion above already removed;
            # subtracting it twice turns this channel into a ramp.
            tx_phase = (
                active_scheme.tx_phase_offset(local_tx_idx)
                if active_scheme is not None
                else 0.0
            )
            second_comp = unwrapped.reshape(num_windows, step).mean(axis=1) - tx_phase

        # Magnitude metric for DPIC, independent of the save format.
        metric = normalized_amplitude(baseband_data, tx_amp)
        # Complex residual phasor (the DC term of the downconverted signal).
        # DPIC's closed-form solve needs the phasor, not just its magnitude.
        metric_complex = complex(np.mean(baseband_data))
        if tx_amp > 0:
            metric_complex /= tx_amp
        return first_comp, second_comp, (metric, metric_complex)

    def _decimate_cal_ref(self, envelope: np.ndarray) -> np.ndarray:
        step = self.save_ds
        num_windows = len(envelope) // step
        if num_windows <= 0:
            return np.array([])
        return envelope[: num_windows * step].reshape(num_windows, step).mean(axis=1)

    def _process_mimo_chunk(self, buffer):
        """Process all MIMO sources once; update metrics and return per-source comps."""
        results = {}
        metrics = {}
        for source in self.mimo_sources:
            data = buffer[source.rx_idx, :]
            dev_name, _ = self.global_tx_to_device.get(
                source.tx_idx, (self.rx_device_order[0], source.tx_idx)
            )
            scheme = self.schemes_by_device.get(dev_name)
            first_comp, second_comp, metric = self._process_chunk(
                data=data,
                source=source,
                filt=self.if_filts[source.tx_idx],
                if_freq=self.channel_ifs[source.tx_idx],
                scheme=scheme,
            )
            results[source.channel] = (first_comp, second_comp)
            if metric is not None:
                metrics[(source.tx_idx, source.rx_idx)] = metric

        if metrics:
            with self._metrics_cv:
                for key, value in metrics.items():
                    prev_seq = self.latest_metrics.get(key, (None, None, -1))[2]
                    self.latest_metrics[key] = (float(value[0]), value[1], prev_seq + 1)
                self._metrics_cv.notify_all()
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

                # Save path: recorded data matters, so absorb a short disk
                # stall before giving up on a chunk.
                if self.save_queue is not None and not put_or_drop(
                    self.save_queue, save_data, timeout=QUEUE_PUT_TIMEOUT_S
                ):
                    self.save_chunks_dropped += 1
                    self._log_drops()

                # Display path: only the newest chunk is useful, so evict the
                # oldest instead of adding latency.
                if self.display_queue is not None:
                    if self.save_imaginary is False:
                        display_payload = display_data
                    elif self.display_imaginary:
                        display_payload = display_data[:, :, 1]
                    else:
                        display_payload = display_data[:, :, 0]
                    # float32 on the wire halves the streamed volume; no plot
                    # resolves more. The save path stays float64.
                    display_payload = np.ascontiguousarray(
                        display_payload, dtype=np.float32
                    )
                    if not put_drop_oldest(self.display_queue, display_payload):
                        self.display_chunks_dropped += 1
                        self._log_drops()

            except queue.Empty:
                time.sleep(0.001)
                # log_print(self.logger, "debug", "[USRP] Rx Queue Empty")
            except Exception as e:
                log_print(self.logger, "error", f"[USRP] Processing error: {e}")

        log_print(self.logger, "debug", "[USRP] Processing stopped")
