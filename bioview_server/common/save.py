"""Session recorder: one ``bioview-raw-v3`` file per session, written server-side."""

import contextlib
import multiprocessing as mp
import queue
import time
from datetime import UTC

import numpy as np
from bioview_common import (
    BVR_FORMAT,
    FLAG_COMPLEX,
    FLAG_FLOAT64,
    PausableWorker,
    encode_header,
    encode_trailer,
    log_print,
    pack_record,
    put_or_drop,
)


DEFAULT_BATCH_RECORDS = 16


def flatten_chunk(chunk) -> np.ndarray:
    """Fold a chunk into a 2-D ``(rows, samples)`` block ready to write."""
    arr = np.asarray(chunk)
    if arr.ndim == 3:
        return np.vstack([arr[:, :, 0], arr[:, :, 1]])
    return arr


class BvrWriter(PausableWorker):
    """Drains the shared save queue into a single ``.bvr`` file."""

    METADATA_TYPES = frozenset({"param_change"})

    QUEUE_TIMEOUT_S = 0.1

    def __init__(
        self,
        save_path,
        data_queue: mp.Queue,
        devices: list[dict],
        device_config: dict = None,
        batch_records: int = DEFAULT_BATCH_RECORDS,
        sample_limits: dict = None,
        logger=None,
    ):
        super().__init__()
        self.logger = logger

        self.save_path = str(save_path)
        self.data_queue = data_queue
        self.device_config = device_config or {}
        self.batch_records = max(int(batch_records), 1)

        self.devices = list(devices)
        self._idx_of = {d["device_id"]: i for i, d in enumerate(self.devices)}
        # Per-device hard sample budget. A routine asks for a fixed number of
        # seconds, but the stream keeps running until the stop command has made
        # a round trip, so without this the file length varied with whatever the
        # timer, the network and the device spin-up happened to cost that run.
        self.sample_limits = dict(sample_limits or {})

        self._file = None
        self._pending = []
        self._pending_count = 0
        self.t0_unix = None

        self._stats = {
            d["device_id"]: {"records": 0, "samples": 0, "gaps": 0, "dropped_samples": 0}
            for d in self.devices
        }
        self._last_end = {}
        self._annotations = []
        self._param_changes = []
        self.records_written = 0
        self.unknown_device_records = 0

    def open(self, t0_unix=None):
        """Create the file and write the header. Call before the thread starts."""
        self.t0_unix = float(t0_unix if t0_unix is not None else time.time())
        header = {
            "format": BVR_FORMAT,
            "t0_unix": self.t0_unix,
            "t0_utc": _iso(self.t0_unix),
            "devices": self.devices,
            "device_config": self.device_config,
        }
        self._file = open(self.save_path, "wb")  # noqa: SIM115
        self._file.write(encode_header(header))
        log_print(self.logger, "info", f"[Save] Recording to {self.save_path}")

    def _offset_us(self, t_wall):
        if t_wall is None or self.t0_unix is None:
            return 0
        return int(round((float(t_wall) - self.t0_unix) * 1e6))

    def record_annotation(self, text, t_wall=None):
        """Add a "Mark Event" note; stored as an offset from ``t0``."""
        entry = {
            "offset_us": self._offset_us(t_wall if t_wall is not None else time.time()),
            "text": str(text),
        }
        self._annotations.append(entry)
        return entry

    def record_change(self, device_id, param, value, t_wall=None):
        """Add a device-parameter change; stored as an offset from ``t0``."""
        self._param_changes.append(
            {
                "offset_us": self._offset_us(
                    t_wall if t_wall is not None else time.time()
                ),
                "device_id": device_id,
                "param": param,
                "value": value,
            }
        )

    def _handle_metadata(self, item) -> bool:
        """Consume a metadata queue item. Returns True if it was one."""
        if not isinstance(item, dict) or "type" not in item:
            return False
        kind = item.get("type")
        if kind not in self.METADATA_TYPES:
            log_print(
                self.logger,
                "debug",
                f"[Save] Ignoring unknown metadata record {kind!r}",
            )
            return True
        if kind == "param_change":
            self.record_change(
                item.get("device_id"),
                item.get("param"),
                item.get("value"),
                t_wall=item.get("t_wall"),
            )
        return True

    def _append(self, item):
        device_id = item.get("device_id")
        idx = self._idx_of.get(device_id)
        if idx is None:
            self.unknown_device_records += 1
            return

        block = flatten_chunk(item.get("data"))
        if block.ndim != 2 or block.size == 0:
            return

        expected_rows = self.devices[idx].get("n_rows")
        if expected_rows and block.shape[0] != expected_rows:
            log_print(
                self.logger,
                "error",
                f"[Save] {device_id} chunk has {block.shape[0]} rows, header says "
                f"{expected_rows}; dropping the record",
            )
            return

        flags = 0
        if block.dtype == np.float64:
            flags |= FLAG_FLOAT64
        elif block.dtype != np.float32:
            block = np.ascontiguousarray(block, dtype=np.float32)
        if np.asarray(item.get("data")).ndim == 3:
            flags |= FLAG_COMPLEX

        n_samples = block.shape[1]
        sample_idx = int(item.get("sample_idx") or 0)

        stats = self._stats[device_id]
        limit = self.sample_limits.get(device_id)
        if limit is not None:
            remaining = limit - stats["samples"]
            if remaining <= 0:
                return
            if n_samples > remaining:
                block = block[:, :remaining]
                n_samples = remaining

        prev_end = self._last_end.get(device_id)
        if prev_end is not None and sample_idx != prev_end:
            stats["gaps"] += 1
            stats["dropped_samples"] += max(0, sample_idx - prev_end)
        self._last_end[device_id] = sample_idx + n_samples

        self._pending.append(
            pack_record(
                device_idx=idx,
                n_samples=n_samples,
                t_offset_us=self._offset_us(item.get("t_wall")),
                sample_idx=sample_idx,
                flags=flags,
            )
        )
        self._pending.append(np.ascontiguousarray(block).tobytes())
        self._pending_count += 1

        stats["records"] += 1
        stats["samples"] += n_samples
        self.records_written += 1

    def _flush(self):
        if not self._pending or self._file is None:
            return
        try:
            self._file.write(b"".join(self._pending))
            self._file.flush()
        except Exception as e:
            log_print(self.logger, "error", f"[Save] Write failed: {e}")
        finally:
            self._pending = []
            self._pending_count = 0

    def work(self):
        if self.data_queue is None:
            return

        while self.is_running:
            try:
                item = self.data_queue.get(timeout=self.QUEUE_TIMEOUT_S)
            except queue.Empty:
                self._flush()
                continue
            except (OSError, ValueError):
                break

            try:
                if self._handle_metadata(item):
                    continue
                self._append(item)
            except Exception as e:
                log_print(self.logger, "error", f"[Save] Bad chunk: {e}")
                continue

            if self._pending_count >= self.batch_records:
                self._flush()

        self._flush()
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.flush()

    def _write_trailer(self):
        elapsed_us = self._offset_us(time.time())
        devices = []
        for i, dev in enumerate(self.devices):
            stats = self._stats[dev["device_id"]]
            achieved = None
            if elapsed_us > 0 and stats["samples"]:
                achieved = stats["samples"] / (elapsed_us / 1e6)
            devices.append(
                {
                    "device_idx": i,
                    "device_id": dev["device_id"],
                    "records": stats["records"],
                    "samples": stats["samples"],
                    "sample_limit": self.sample_limits.get(dev["device_id"]),
                    "gaps": stats["gaps"],
                    "dropped_samples": stats["dropped_samples"],
                    "achieved_fs": achieved,
                }
            )
        trailer = {
            "t_end_offset_us": elapsed_us,
            "devices": devices,
            "param_changes": self._param_changes,
            "Annotations": self._annotations,
        }
        self._file.write(encode_trailer(trailer))

    def cleanup(self):
        self._flush()
        if self._file is None:
            return
        try:
            self._write_trailer()
            self._file.flush()
            self._file.close()
            log_print(self.logger, "debug", f"[Save] Closed {self.save_path}")
        except Exception as e:
            log_print(self.logger, "error", f"[Save] Close failed: {e}")
        finally:
            self._file = None


def _iso(unix_time):
    from datetime import datetime

    return datetime.fromtimestamp(unix_time, tz=UTC).isoformat()


class SaveForwarder(PausableWorker):
    """Tags one device's save chunks and forwards them to the session recorder."""

    QUEUE_TIMEOUT_S = 0.1

    def __init__(self, device_id, data_input_queue, data_output_queue, logger=None):
        super().__init__()
        self.logger = logger
        self.device_id = device_id
        self.data_input_queue = data_input_queue
        self.data_output_queue = data_output_queue
        self.chunks_dropped = 0
        self._samples_seen = 0

    def work(self):
        if self.data_input_queue is None or self.data_output_queue is None:
            return

        while self.is_running:
            try:
                item = self.data_input_queue.get(timeout=self.QUEUE_TIMEOUT_S)
            except queue.Empty:
                continue
            except (OSError, ValueError):
                break

            if isinstance(item, dict):
                data = item.get("data")
                sample_idx = item.get("sample_idx")
                t_wall = item.get("t_wall")
            else:
                data, sample_idx, t_wall = item, None, None

            if data is None:
                continue

            arr = np.asarray(data)
            n_samples = arr.shape[1] if arr.ndim >= 2 else arr.size
            if sample_idx is None:
                sample_idx = self._samples_seen
            self._samples_seen = int(sample_idx) + int(n_samples)

            payload = {
                "device_id": self.device_id,
                "data": data,
                "sample_idx": int(sample_idx),
                "t_wall": float(t_wall) if t_wall is not None else time.time(),
            }
            if not put_or_drop(self.data_output_queue, payload, timeout=1.0):
                self.chunks_dropped += 1
                log_print(
                    self.logger,
                    "error",
                    f"[Save] {self.device_id}: recorder queue full, dropped a "
                    f"chunk ({self.chunks_dropped} total)",
                )
