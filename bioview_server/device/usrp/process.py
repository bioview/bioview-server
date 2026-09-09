import queue
import threading
import time

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
        rx_queues: dict,
        rx_device_order: list[str],
        schemes_by_device: dict,
        global_tx_to_device: dict,
        signal_scheme: str = "cw",
        fmcw_scheme: FmcwScheme | None = None,
        save_queue: queue.Queue = None,
        display_queue: queue.Queue = None,
        save_ds: int = 1,
        display_ds: int = 1,
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
        self.display_ds = max(1, int(display_ds))
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
        # (measure_tx, measure_rx) -> (magnitude, seq); ``seq`` lets DPIC
        # wait for a measurement taken after it changed the injection.
        self.latest_metrics = {}
        self._metrics_cv = threading.Condition()
        self._partial_rows = {}

        num_tx = len(channel_ifs)
        # Kept so a channel can be retuned after construction; DPIC balance
        # moves the inject Tx onto the measure Tx's IF.
        self.if_filter_bw = list(if_filter_bw)
        self.if_filts = [
            self._load_filter(channel_ifs[idx], if_filter_bw[idx])
            for idx in range(num_tx)
        ]

        self.save_chunks_dropped = 0
        self.display_chunks_dropped = 0
        self._last_drops_logged = 0

        self._rebuild_demod_sources()

    def _rebuild_demod_sources(self):
        """Pick one source per (Tx, Rx) pair to carry the demodulator state.

        A group streaming both amplitude and phase advertises two sources for
        the same physical pair. Demodulation is per pair, not per row: running
        it once per source would advance the phase accumulator and the filter
        state twice per chunk, and the second pass would see a buffer it has
        already consumed. Only the representative holds state; the other rows
        read the components it produced.
        """
        seen = {}
        for source in self.mimo_sources:
            seen.setdefault((source.tx_idx, source.rx_idx), source)
        self._demod_sources = list(seen.values())

        # More rows than pairs means components were named explicitly, so each
        # row already carries exactly one component and the save path stores
        # them as plain rows rather than as a stacked real/imaginary pair.
        self._per_source_components = len(self.mimo_sources) > len(self._demod_sources)

        for source in self.mimo_sources:
            source.accumulated_phase = 0.0
            source.filter_state = None
            source.prev_phase = None
            source.accumulated_sample_idx = 0

    def _component_of(self, source) -> str:
        """Which of the two derived quantities this row carries.

        Falls back to the legacy group-wide ``display_imaginary`` switch for
        sources built before components were named per row.
        """
        component = getattr(source, "component", None)
        if component is not None:
            return component
        return "phase" if self.display_imaginary else "amplitude"

    def _load_filter(self, freq: float, bandwidth: float, order: int = 2):
        low_cutoff = freq - bandwidth / 2
        high_cutoff = freq + bandwidth / 2
        return get_filter(
            bounds=[low_cutoff, high_cutoff],
            samp_rate=self.samp_rate,
            btype="band",
            order=order,
        )

    def set_channel_if(self, tx_idx: int, freq: float):
        """Retune one Tx's demodulation IF and its band-pass, live.

        ``channel_ifs`` is the backend's own list, so the frequency may already
        be updated by the time this is called; the filter is not, and a stale
        band-pass would reject the very tone it is meant to pass. Filter state
        is dropped for the affected sources because it describes the old
        passband.
        """
        if tx_idx < 0 or tx_idx >= len(self.if_filts):
            return
        freq = float(freq)
        self.channel_ifs[tx_idx] = freq
        self.if_filts[tx_idx] = self._load_filter(freq, self.if_filter_bw[tx_idx])
        for source in self.all_sources:
            if getattr(source, "tx_idx", None) == tx_idx:
                source.filter_state = None

    def set_sources(self, data_sources, cal_ref_sources, channel_ifs, if_filter_bw):
        """Adopt a new channel map's rows, filters and per-source state.

        Everything ``__init__`` derives from the source list, redone: the
        channel map decides how many rows are emitted and what each one is, so
        a stale list here means rows demodulated against the wrong Tx.
        Per-source demodulator state and the DPIC metrics are dropped -- both
        are indexed by the meanings that just changed.
        """
        self.mimo_sources = sorted(data_sources, key=lambda s: s.channel)
        self.cal_ref_sources = sorted(cal_ref_sources or [], key=lambda s: s.channel)
        self.all_sources = self.mimo_sources + self.cal_ref_sources
        self.data_sources = self.all_sources

        # Held by reference, as in __init__: the backend mutates this list when
        # DPIC coerces an inject Tx's IF.
        self.channel_ifs = channel_ifs
        self.if_filter_bw = list(if_filter_bw)
        self.if_filts = [
            self._load_filter(self.channel_ifs[idx], self.if_filter_bw[idx])
            for idx in range(len(self.channel_ifs))
        ]

        self._partial_rows.clear()
        with self._metrics_cv:
            self.latest_metrics.clear()

        self._rebuild_demod_sources()

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

    def wait_for_metric(
        self,
        measure_tx: int,
        measure_rx: int,
        min_new: int = 2,
        timeout: float = 2.0,
    ) -> float | None:
        """Block until ``min_new`` fresh chunks have been measured.

        The Rx path buffers deeply, so a value read straight after a setting
        change still describes the old one.
        """
        entry = self._wait_for_entry(measure_tx, measure_rx, min_new, timeout)
        return entry[0] if entry else None

    def _wait_for_entry(self, measure_tx, measure_rx, min_new, timeout):
        key = (measure_tx, measure_rx)
        deadline = time.monotonic() + timeout
        with self._metrics_cv:
            entry = self.latest_metrics.get(key)
            target = (entry[1] if entry else -1) + min_new
            while True:
                entry = self.latest_metrics.get(key)
                if entry and entry[1] >= target:
                    return entry
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return entry
                self._metrics_cv.wait(remaining)

    def _process_chunk(self, data, source, filt, if_freq, scheme):
        """Returns (first_comp, second_comp, metric).

        ``metric`` is always the normalized mean baseband magnitude; it must
        not be derived from ``first_comp``, which is signed under ``save_iq``.
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

        # Save windows are contiguous blocks of ``step`` samples, so this
        # reshape is a view rather than a copy.
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

            # Unwrap once with continuity from the previous chunk, then
            # reduce per window.
            angles = np.angle(baseband_data[:usable])
            if source.prev_phase is None:
                unwrapped = np.unwrap(angles)
            else:
                unwrapped = np.unwrap(np.concatenate(([source.prev_phase], angles)))[1:]
            source.prev_phase = float(unwrapped[-1])

            # Static Tx phase only: tx_phase_at() also carries the IF ramp
            # that the downconversion already removed.
            tx_phase = (
                active_scheme.tx_phase_offset(local_tx_idx)
                if active_scheme is not None
                else 0.0
            )
            second_comp = unwrapped.reshape(num_windows, step).mean(axis=1) - tx_phase

        # Magnitude metric for DPIC, independent of the save format.
        metric = normalized_amplitude(baseband_data, tx_amp)
        return first_comp, second_comp, metric

    def _decimate_display(self, payload: np.ndarray) -> np.ndarray:
        """Reduce the display stream by ``display_ds`` on top of ``save_ds``.

        The advertised ``disp_freq`` on every source is
        ``samp_rate / (save_ds * display_ds)``; the two must stay in step or
        the client's time axis scrolls at the wrong speed.
        """
        step = self.display_ds
        if step <= 1:
            return payload
        n_rows, n_samples = payload.shape
        num_windows = n_samples // step
        if num_windows <= 0:
            # Chunk shorter than one display window: keep a single averaged
            # point rather than dropping the chunk entirely.
            return payload.mean(axis=1, keepdims=True)
        usable = num_windows * step
        return payload[:, :usable].reshape(n_rows, num_windows, step).mean(axis=2)

    def _decimate_cal_ref(self, envelope: np.ndarray) -> np.ndarray:
        step = self.save_ds
        num_windows = len(envelope) // step
        if num_windows <= 0:
            return np.array([])
        return envelope[: num_windows * step].reshape(num_windows, step).mean(axis=1)

    def _process_mimo_chunk(self, buffer):
        """Demodulate every (Tx, Rx) pair once.

        Keyed by ``(tx_idx, rx_idx)`` rather than by channel because several
        display rows can share one pair -- one per streamed component.
        """
        results = {}
        metrics = {}
        for source in self._demod_sources:
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
            results[(source.tx_idx, source.rx_idx)] = (first_comp, second_comp)
            if metric is not None:
                metrics[(source.tx_idx, source.rx_idx)] = metric

        if metrics:
            with self._metrics_cv:
                for key, value in metrics.items():
                    prev_seq = self.latest_metrics.get(key, (None, -1))[1]
                    self.latest_metrics[key] = (float(value), prev_seq + 1)
                self._metrics_cv.notify_all()
        return results

    def _assemble_outputs(self, buffer, mimo_results):
        num_mimo = len(self.mimo_sources)
        num_cal = len(self.cal_ref_sources) if self.record_cal_ref else 0
        len_samples = int(buffer.shape[1] // self.save_ds)

        # Calibration reference rows go to the display as well as to disk;
        # the backend advertises a CalRef_* source for each.
        n_rows = num_mimo + num_cal
        # The display payload is always one value per advertised source: the
        # component that source names. The save path keeps its older stacked
        # (row, sample, 2) form unless the rows already carry one component
        # each, in which case stacking would just duplicate them.
        stack_save = self.save_imaginary and not self._per_source_components
        if stack_save:
            save_list = np.zeros((n_rows, len_samples, 2))
        else:
            save_list = np.zeros((n_rows, len_samples))
        display_list = np.zeros((n_rows, len_samples))

        for source in self.mimo_sources:
            first_comp, second_comp = mimo_results[(source.tx_idx, source.rx_idx)]
            component = (
                second_comp if self._component_of(source) == "phase" else first_comp
            )
            display_list[source.channel, :] = component
            if stack_save:
                save_list[source.channel, :, 0] = first_comp
                save_list[source.channel, :, 1] = second_comp
            else:
                save_list[source.channel, :] = component

        if self.record_cal_ref:
            for source in self.cal_ref_sources:
                global_tx = source.tx_idx
                dev_name, local_tx = self.global_tx_to_device[global_tx]
                scheme = self.schemes_by_device[dev_name]
                raw_env = scheme.get_calibration_reference(
                    local_tx, self.global_sample_idx, buffer.shape[1]
                )
                cal_data = self._decimate_cal_ref(raw_env)
                if stack_save:
                    # Real-valued envelope: channel 0, imaginary left at 0.
                    save_list[source.channel, : len(cal_data), 0] = cal_data
                    save_list[source.channel, : len(cal_data), 1] = 0.0
                else:
                    save_list[source.channel, : len(cal_data)] = cal_data
                display_list[source.channel, : len(cal_data)] = cal_data

        self.global_sample_idx += buffer.shape[1]
        return save_list, display_list

    def work(self):
        # work() is re-entered on every resume; rows held from before the
        # pause would misalign the MIMO buffer.
        self._partial_rows.clear()

        while self.is_running:
            try:
                for key in self.rx_device_order:
                    if key not in self._partial_rows:
                        rx_q = self.rx_queues[key]
                        self._partial_rows[key] = rx_q.get_nowait()

                rows = [self._partial_rows[key].copy() for key in self.rx_device_order]
                self._partial_rows.clear()
                widths = {row.shape[1] for row in rows}
                if len(widths) > 1:
                    # Different-length buffers: trim to the common length
                    # rather than letting vstack raise per chunk.
                    n = min(widths)
                    rows = [row[:, :n] for row in rows]
                buffer = np.vstack(rows)

                mimo_results = self._process_mimo_chunk(buffer)
                # Index of this chunk's first *save* sample, taken before
                # _assemble_outputs advances the raw counter. The recorder uses
                # it to tell a contiguous chunk from one that follows a drop.
                save_sample_idx = self.global_sample_idx // max(1, int(self.save_ds))
                save_data, display_data = self._assemble_outputs(buffer, mimo_results)

                # Save path: absorb a short disk stall before dropping.
                save_item = {
                    "data": save_data,
                    "sample_idx": save_sample_idx,
                    "t_wall": time.time(),
                }
                if self.save_queue is not None and not put_or_drop(
                    self.save_queue, save_item, timeout=QUEUE_PUT_TIMEOUT_S
                ):
                    self.save_chunks_dropped += 1
                    self._log_drops()

                # Display path: evict the oldest rather than add latency.
                if self.display_queue is not None:
                    display_payload = self._decimate_display(display_data)
                    # float32 on the wire; the save path stays float64.
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
