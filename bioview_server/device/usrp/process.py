import queue
import threading
import time

import numpy as np
from bioview_common import (
    QUEUE_PUT_TIMEOUT_S,
    PausableWorker,
    apply_filter,
    get_filter,
    initial_state,
    log_print,
    put_drop_oldest,
    put_or_drop,
)
from bioview_common.signal_schemes import (
    FmcwScheme,
    normalized_amplitude,
)
from scipy import signal


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
        if_filter_type: str = "ellip",
        if_filter_order: int = 2,
        disp_filter_btype: str = "off",
        disp_filter_low: float = 0.5,
        disp_filter_high: float = 40.0,
        disp_filter_order: int = 2,
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
        self.latest_metrics = {}
        self._metrics_cv = threading.Condition()
        self._partial_rows = {}

        self.if_filter_bw = list(if_filter_bw)
        self.if_filter_type = str(if_filter_type or "ellip")
        self.if_filter_order = max(1, int(if_filter_order or 2))
        self._ramp_cache = {}
        self._rebuild_filters()

        self.disp_filter_btype = str(disp_filter_btype or "off")
        self.disp_filter_low = float(disp_filter_low)
        self.disp_filter_high = float(disp_filter_high)
        self.disp_filter_order = max(1, int(disp_filter_order or 2))
        self._disp_filt = None
        self._disp_zi = None
        self._rebuild_display_filter()

        self.save_chunks_dropped = 0
        self.display_chunks_dropped = 0
        self._last_drops_logged = 0

        self._rebuild_demod_sources()

    def _rebuild_demod_sources(self):
        """Pick one source per (Tx, Rx) pair to carry the demodulator state."""
        seen = {}
        for source in self.mimo_sources:
            seen.setdefault((source.tx_idx, source.rx_idx), source)
        self._demod_sources = list(seen.values())

        self._per_source_components = len(self.mimo_sources) > len(self._demod_sources)

        for source in self.mimo_sources:
            source.accumulated_phase = 0.0
            source.filter_state = None
            source.prev_phase = None
            source.accumulated_sample_idx = 0

    def _component_of(self, source) -> str:
        """Which of the two derived quantities this row carries."""
        component = getattr(source, "component", None)
        if component is not None:
            return component
        return "phase" if self.display_imaginary else "amplitude"

    def _load_filter(self, freq: float, bandwidth: float, order: int = None):
        """Band-pass isolating one Tx's IF tone, in the configured response."""
        nyquist = self.samp_rate / 2.0
        half = max(float(bandwidth), 1.0) / 2.0
        low_cutoff = max(freq - half, nyquist * 1e-6)
        high_cutoff = min(freq + half, nyquist * (1.0 - 1e-6))
        if high_cutoff <= low_cutoff:
            raise ValueError(
                f"IF {freq:.0f} Hz with bandwidth {bandwidth:.0f} Hz leaves no "
                f"passband below the {nyquist:.0f} Hz Nyquist limit"
            )
        return get_filter(
            bounds=[low_cutoff, high_cutoff],
            samp_rate=self.samp_rate,
            ftype=self.if_filter_type,
            btype="band",
            order=self.if_filter_order if order is None else order,
            dtype=np.float32,
        )

    def _rebuild_filters(self):
        """Redesign every Tx's band-pass from the current filter settings."""
        filters = [
            self._load_filter(self.channel_ifs[idx], self.if_filter_bw[idx])
            for idx in range(len(self.channel_ifs))
        ]
        self.if_filts = filters
        self._ramp_cache.clear()
        for source in self.all_sources:
            source.filter_state = None

    # -- display-only filter ---------------------------------------------------

    def get_display_rate(self) -> float:
        """Rate of the stream the display filter sees, after both decimations."""
        return float(self.samp_rate) / (
            max(1, int(self.save_ds)) * max(1, int(self.display_ds))
        )

    def _rebuild_display_filter(self):
        """Design the display-only filter. Never touches the saved stream."""
        self._disp_zi = None
        btype = self.disp_filter_btype
        if btype not in ("low", "high", "band"):
            self._disp_filt = None
            return

        bounds = (
            [self.disp_filter_low, self.disp_filter_high]
            if btype == "band"
            else (self.disp_filter_low if btype == "high" else self.disp_filter_high)
        )
        self._disp_filt = get_filter(
            bounds=bounds,
            samp_rate=self.get_display_rate(),
            ftype="butter",
            btype=btype,
            order=self.disp_filter_order,
            dtype=np.float32,
        )

    def set_display_filter(self, btype=None, low=None, high=None, order=None) -> bool:
        """Adopt new display-filter settings live. Returns True if it changed."""
        previous = (
            self.disp_filter_btype,
            self.disp_filter_low,
            self.disp_filter_high,
            self.disp_filter_order,
        )
        if btype is not None:
            self.disp_filter_btype = str(btype)
        if low is not None:
            self.disp_filter_low = float(low)
        if high is not None:
            self.disp_filter_high = float(high)
        if order is not None:
            self.disp_filter_order = max(1, int(order))

        current = (
            self.disp_filter_btype,
            self.disp_filter_low,
            self.disp_filter_high,
            self.disp_filter_order,
        )
        if previous == current:
            return False

        try:
            self._rebuild_display_filter()
        except Exception:
            (
                self.disp_filter_btype,
                self.disp_filter_low,
                self.disp_filter_high,
                self.disp_filter_order,
            ) = previous
            self._rebuild_display_filter()
            raise
        return True

    def _filter_display(self, payload: np.ndarray) -> np.ndarray:
        """Run the display filter across rows, carrying state between chunks."""
        sos = self._disp_filt
        if sos is None or payload.size == 0:
            return payload
        if self._disp_zi is None or self._disp_zi.shape[1] != payload.shape[0]:
            # sosfilt wants (n_sections, n_rows, 2) when filtering along axis 1,
            # seeded from the first sample of each row so the plot does not open
            # with a step response.
            steady = signal.sosfilt_zi(sos)[:, np.newaxis, :]
            self._disp_zi = (steady * payload[np.newaxis, :, 0, np.newaxis]).astype(
                sos.dtype
            )
        filtered, self._disp_zi = signal.sosfilt(sos, payload, axis=1, zi=self._disp_zi)
        return filtered

    # -- demodulation ----------------------------------------------------------

    def _phasor_ramp(self, if_freq: float, n: int) -> np.ndarray:
        """``exp(-1j*inc*arange(n))`` for one IF, cached across chunks and Rx.

        Every Rx demodulating the same Tx walks the identical ramp, and the
        chunk length is fixed by the Rx buffer, so the transcendental is paid
        once per configuration rather than once per source per chunk. That one
        substitution is 0.62 ms of the 2.2 ms each source used to cost.
        """
        key = (float(if_freq), int(n))
        ramp = self._ramp_cache.get(key)
        if ramp is None:
            increment = 2 * np.pi * float(if_freq) / self.samp_rate
            ramp = np.exp(-1j * increment * np.arange(n)).astype(np.complex64)
            if len(self._ramp_cache) >= self._RAMP_CACHE_MAX:
                self._ramp_cache.clear()
            self._ramp_cache[key] = ramp
        return ramp

    _RAMP_CACHE_MAX = 64

    def set_filter_params(self, bandwidths=None, ftype: str = None, order: int = None):
        """Adopt new IF band-pass settings live. Returns True if anything changed."""
        previous = (list(self.if_filter_bw), self.if_filter_type, self.if_filter_order)

        if bandwidths is not None:
            values = (
                [float(bw) for bw in bandwidths]
                if isinstance(bandwidths, list | tuple)
                else [float(bandwidths)] * len(self.channel_ifs)
            )
            for idx, value in enumerate(values[: len(self.if_filter_bw)]):
                self.if_filter_bw[idx] = value
        if ftype is not None:
            self.if_filter_type = str(ftype)
        if order is not None:
            self.if_filter_order = max(1, int(order))

        if previous == (self.if_filter_bw, self.if_filter_type, self.if_filter_order):
            return False

        try:
            self._rebuild_filters()
        except Exception:
            self.if_filter_bw, self.if_filter_type, self.if_filter_order = (
                list(previous[0]),
                previous[1],
                previous[2],
            )
            raise
        return True

    def set_samp_rate(self, samp_rate: float) -> bool:
        """Demodulate at a new sample rate. Returns True if anything changed."""
        samp_rate = float(samp_rate)
        if samp_rate == self.samp_rate:
            return False

        previous = self.samp_rate
        self.samp_rate = samp_rate
        try:
            self._rebuild_filters()
        except Exception:
            self.samp_rate = previous
            self._rebuild_filters()
            raise
        self._rebuild_display_filter()
        self._rebuild_demod_sources()
        return True

    def set_channel_if(self, tx_idx: int, freq: float):
        """Retune one Tx's demodulation IF and its band-pass, live."""
        if tx_idx < 0 or tx_idx >= len(self.if_filts):
            return
        freq = float(freq)
        self.channel_ifs[tx_idx] = freq
        self.if_filts[tx_idx] = self._load_filter(freq, self.if_filter_bw[tx_idx])
        for source in self.all_sources:
            if getattr(source, "tx_idx", None) == tx_idx:
                source.filter_state = None

    def set_sources(self, data_sources, cal_ref_sources, channel_ifs, if_filter_bw):
        """Adopt a new channel map's rows, filters and per-source state."""
        self.mimo_sources = sorted(data_sources, key=lambda s: s.channel)
        self.cal_ref_sources = sorted(cal_ref_sources or [], key=lambda s: s.channel)
        self.all_sources = self.mimo_sources + self.cal_ref_sources
        self.data_sources = self.all_sources

        self.channel_ifs = channel_ifs
        self.if_filter_bw = list(if_filter_bw)
        self._rebuild_filters()
        self._disp_zi = None

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
        """Block until ``min_new`` fresh chunks have been measured."""
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
        """Returns (first_comp, second_comp, metric)."""
        if len(data) == 0:
            return np.array([]), np.array([]), None

        if source.filter_state is None:
            source.filter_state = initial_state(filt, data[0])
        filt_data, new_filter_state = apply_filter(data, filt, zi=source.filter_state)
        source.filter_state = new_filter_state

        n_in = len(filt_data)
        current_phase = source.accumulated_phase
        phase_increment = 2 * np.pi * if_freq / self.samp_rate

        downconversion = np.complex64(np.exp(-1j * current_phase)) * self._phasor_ramp(
            if_freq, n_in
        )
        baseband_data = filt_data * downconversion

        if self.signal_scheme == "fmcw" and self.fmcw_scheme is not None:
            ref = self.fmcw_scheme.get_dechirp_reference(
                n_in, source.accumulated_sample_idx
            )
            baseband_data = baseband_data * ref

        # Kept inside one turn: the accumulator used to grow without bound, so a
        # long session slowly lost mantissa on the very quantity the phase
        # channel is made of.
        source.accumulated_phase = (current_phase + n_in * phase_increment) % (2 * np.pi)
        source.accumulated_sample_idx += n_in

        step = self.save_ds
        num_windows = len(baseband_data) // step
        if num_windows <= 0:
            return np.array([]), np.array([]), None

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
            first_comp = np.abs(windows).mean(axis=1)
            if tx_amp > 0:
                first_comp = first_comp / tx_amp

            tx_phase = (
                active_scheme.tx_phase_offset(local_tx_idx)
                if active_scheme is not None
                else 0.0
            )
            # Vector-average each window, then take one angle per saved point.
            # The old path unwrapped all `save_ds * num_windows` raw angles and
            # averaged those: 0.87 ms per source per chunk against 0.04 ms here,
            # and it produced a phase that ran away linearly instead of sitting
            # in (-pi, pi]. Averaging the complex samples is also the better
            # estimator -- noise cancels in the sum rather than folding through
            # arctan2 first.
            second_comp = np.angle(windows.mean(axis=1) * np.exp(-1j * tx_phase))

        metric = normalized_amplitude(baseband_data, tx_amp)
        return first_comp, second_comp, metric

    def _decimate_display(self, payload: np.ndarray) -> np.ndarray:
        """Reduce the display stream by ``display_ds`` on top of ``save_ds``."""
        step = self.display_ds
        if step <= 1:
            return payload
        n_rows, n_samples = payload.shape
        num_windows = n_samples // step
        if num_windows <= 0:
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
        """Demodulate every (Tx, Rx) pair once."""
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

        n_rows = num_mimo + num_cal
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
                    save_list[source.channel, : len(cal_data), 0] = cal_data
                    save_list[source.channel, : len(cal_data), 1] = 0.0
                else:
                    save_list[source.channel, : len(cal_data)] = cal_data
                display_list[source.channel, : len(cal_data)] = cal_data

        self.global_sample_idx += buffer.shape[1]
        return save_list, display_list

    def work(self):
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
                    n = min(widths)
                    rows = [row[:, :n] for row in rows]
                buffer = np.vstack(rows)

                mimo_results = self._process_mimo_chunk(buffer)
                save_sample_idx = self.global_sample_idx // max(1, int(self.save_ds))
                save_data, display_data = self._assemble_outputs(buffer, mimo_results)

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

                if self.display_queue is not None:
                    display_payload = self._decimate_display(display_data)
                    display_payload = self._filter_display(
                        np.ascontiguousarray(display_payload, dtype=np.float32)
                    )
                    # The wire format assumes a contiguous float32 block, and
                    # the filter is the last thing that could have broken that.
                    display_payload = np.ascontiguousarray(
                        display_payload, dtype=np.float32
                    )
                    if not put_drop_oldest(self.display_queue, display_payload):
                        self.display_chunks_dropped += 1
                        self._log_drops()

            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                log_print(self.logger, "error", f"[USRP] Processing error: {e}")

        log_print(self.logger, "debug", "[USRP] Processing stopped")
