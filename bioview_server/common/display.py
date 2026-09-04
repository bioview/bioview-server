import queue

from bioview_common import DataSource, PausableWorker, log_print, put_drop_oldest


class DisplayWorker(PausableWorker):
    """Forwards processed data to the client.

    Emits a contiguous (num_sources, num_samples) array plus the ordered source
    list describing each row.
    """

    def __init__(
        self,
        display_sources: list[DataSource] = None,
        data_input_queue: queue.Queue = None,  # Data comes in
        data_output_queue: queue.Queue = None,  # Data pushed to client
        logger=None,
    ):
        super().__init__()
        self.set_display_sources(display_sources)

        # Drop bookkeeping, read by work().
        self.dropped_chunks = 0
        self._last_drop_logged = 0

        # Queues
        self.data_input_queue = data_input_queue
        self.data_output_queue = data_output_queue

        self.logger = logger

    def _log_drop(self):
        if self.dropped_chunks - self._last_drop_logged >= 50:
            self._last_drop_logged = self.dropped_chunks
            log_print(
                self.logger,
                "warning",
                f"[Display] Client is not keeping up; {self.dropped_chunks} "
                "chunks dropped",
            )

    def set_display_sources(self, display_sources):
        """Replace the row -> source map used to label emitted chunks.

        Updatable while the worker is alive: a channel change alters both.
        """
        # Ordered list of sources; row i of each emitted array corresponds to
        # display_sources[i]. Ordered by channel so it matches ProcessWorker output.
        self.display_sources = sorted(
            display_sources if display_sources is not None else [],
            key=lambda s: s.channel,
        )

        # Precompute serializable source descriptors (sent as chunk metadata)
        self._source_dicts = [s.to_dict() for s in self.display_sources]

    def work(self):
        # Nothing to do if we have no queues; return
        if self.data_input_queue is None or self.data_output_queue is None:
            return

        try:
            # Get a processed (num_sources, num_samples) chunk
            samples = self.data_input_queue.get(timeout=0.1)
        except queue.Empty:
            return

        try:
            payload = {"data": samples, "sources": self._source_dicts}
            # Live display: only the newest chunk has value, so evict the oldest
            # rather than stalling here and letting latency grow unbounded.
            if not put_drop_oldest(self.data_output_queue, payload):
                self.dropped_chunks += 1
                self._log_drop()
        except Exception as e:
            log_print(self.logger, "error", f"Error occurred: {e}")
