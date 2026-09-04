"""A second START_STREAMING must not restart a session that is already running.

Start/Stop is served on the session's own command thread, and the client's Start
button stays live for the minute a start can take (every worker is an OS process
spawn on Windows). A user who clicks it again -- or a second window on the same
shared server -- used to have every click reach the devices, tearing down and
rebuilding the workers under a session that was already streaming.
"""

from bioview_common import Response

from bioview_server.server import Server


class _Recorder:
    """Captures what the server sent back instead of writing to a socket."""

    def __init__(self):
        self.sent = []


def _server(monkeypatch, recorder):
    srv = Server(local_only=True, control_port=0, data_port=0)

    import bioview_server.server as server_mod

    def _capture(sock, response, params=None, logger=None):
        recorder.sent.append((response, params or {}))

    monkeypatch.setattr(server_mod, "send_response", _capture)
    return srv


class _Handler:
    def __init__(self):
        self.starts = 0
        self.stops = 0

    def start_streaming(self, _cfg=None):
        self.starts += 1

    def stop_streaming(self):
        self.stops += 1


def test_a_repeat_start_is_answered_without_touching_the_devices(monkeypatch):
    recorder = _Recorder()
    srv = _server(monkeypatch, recorder)

    handler = _Handler()
    monkeypatch.setattr(srv, "_active_device_handlers", lambda: {"USRP": handler})

    srv._start_streaming({"Experiment": {}})
    assert handler.starts == 1

    for _ in range(5):
        srv._start_streaming({"Experiment": {}})

    assert handler.starts == 1, "repeat starts must not reach the device"

    # SUCCESS, not ERROR: the session *is* streaming, which is what was asked
    # for, and an error would drop a working client into a failed state.
    assert all(resp is Response.SUCCESS for resp, _ in recorder.sent)


def test_stopping_lets_the_next_start_through(monkeypatch):
    recorder = _Recorder()
    srv = _server(monkeypatch, recorder)

    handler = _Handler()
    monkeypatch.setattr(srv, "_active_device_handlers", lambda: {"USRP": handler})

    srv._start_streaming({"Experiment": {}})
    srv._stop_streaming()
    srv._start_streaming({"Experiment": {}})

    assert handler.starts == 2
    assert handler.stops == 1


def test_a_failed_start_does_not_latch_the_session_as_streaming(monkeypatch):
    """Otherwise one bad start would block every retry for the session's life."""
    recorder = _Recorder()
    srv = _server(monkeypatch, recorder)

    class _Broken(_Handler):
        def start_streaming(self, _cfg=None):
            super().start_streaming(_cfg)
            raise RuntimeError("USRP did not answer START_STREAMING within 90s")

    handler = _Broken()
    monkeypatch.setattr(srv, "_active_device_handlers", lambda: {"USRP": handler})

    srv._start_streaming({"Experiment": {}})
    srv._start_streaming({"Experiment": {}})

    assert handler.starts == 2
    assert recorder.sent[0][0] is Response.ERROR
