import queue

from bioview_common import DataSource, PausableWorker, log_print, put_drop_oldest


class DisplayWorker(PausableWorker):
    """Forwards processed data to the client."""

    def __init__(
        self,
        display_sources: list[DataSource] = None,
        data_input_queue: queue.Queue = None,
        data_output_queue: queue.Queue = None,
        logger=None,
    ):
        super().__init__()
        self.set_display_sources(display_sources)

        self.dropped_chunks = 0
        self._last_drop_logged = 0

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
        """Replace the row -> source map used to label emitted chunks."""
        self.display_sources = sorted(
            display_sources if display_sources is not None else [],
            key=lambda s: s.channel,
        )

        self._source_dicts = [s.to_dict() for s in self.display_sources]

    def work(self):
        if self.data_input_queue is None or self.data_output_queue is None:
            return

        try:
            samples = self.data_input_queue.get(timeout=0.1)
        except queue.Empty:
            return

        try:
            payload = {"data": samples, "sources": self._source_dicts}
            if not put_drop_oldest(self.data_output_queue, payload):
                self.dropped_chunks += 1
                self._log_drop()
        except Exception as e:
            log_print(self.logger, "error", f"Error occurred: {e}")
