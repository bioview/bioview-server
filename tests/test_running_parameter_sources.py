"""A live parameter change republishes the server's data sources.

The set of streams a device produces is not fixed: a BIOPAC channel mask decides
it. Rebuilding the list (rather than merging into it) is what lets a channel
disappear, and returning it with the reply is what lets the client's plot-source
selector follow along without waiting for a reconnect.
"""

from bioview_common import DataSource, Response

from bioview_server.server import Server


class FakeHandler:
    def __init__(self, sources):
        self.sources = set(sources)
        self.updates = []

    def get_data_sources(self):
        return self.sources

    def queue_param_update(self, **params):
        self.updates.append(params)
        channels = params.get("channels")
        if channels is not None:
            self.sources = {
                DataSource(group_id="BIOPAC", channel=i, label=f"Ch{i + 1}")
                for i, on in enumerate(channels)
                if on
            }


class FailingHandler(FakeHandler):
    def get_data_sources(self):
        raise RuntimeError("backend is wedged")


def _server(handlers):
    srv = Server(local_only=True, control_port=0, data_port=0)
    srv.device_group_handlers = handlers
    srv.data_sources = set()
    return srv


def _capture(srv, monkeypatch):
    sent = {}

    def fake_send(sock, response, params=None, logger=None):
        sent["response"] = response
        sent["params"] = params or {}

    monkeypatch.setattr("bioview_server.server.send_response", fake_send)
    monkeypatch.setattr(Server, "client_control_conn", property(lambda self: None))
    return sent


def _sources(n):
    return {
        DataSource(group_id="BIOPAC", channel=i, label=f"Ch{i + 1}") for i in range(n)
    }


def test_a_channel_change_is_reported_back_to_the_client(monkeypatch):
    srv = _server({"BIOPAC": FakeHandler(_sources(2))})
    sent = _capture(srv, monkeypatch)

    srv._update_running_parameter({"id": "BIOPAC", "config": {"channels": [1, 1, 1, 1]}})

    assert sent["response"] == Response.SUCCESS
    labels = sorted(s["label"] for s in sent["params"]["data_sources"])
    assert labels == ["Ch1", "Ch2", "Ch3", "Ch4"]


def test_a_disabled_channel_actually_disappears(monkeypatch):
    """A set union can only ever add; the list has to be rebuilt."""
    srv = _server({"BIOPAC": FakeHandler(_sources(4))})
    srv._refresh_data_sources()
    sent = _capture(srv, monkeypatch)

    srv._update_running_parameter({"id": "BIOPAC", "config": {"channels": [1, 0, 0, 0]}})

    labels = sorted(s["label"] for s in sent["params"]["data_sources"])
    assert labels == ["Ch1"]
    assert sorted(s.label for s in srv.data_sources) == ["Ch1"]


def test_other_devices_keep_their_sources(monkeypatch):
    srv = _server(
        {
            "BIOPAC": FakeHandler(_sources(2)),
            "USRP": FakeHandler({DataSource("USRP", 0, "Tx1Rx1")}),
        }
    )
    sent = _capture(srv, monkeypatch)

    srv._update_running_parameter({"id": "BIOPAC", "config": {"channels": [1, 0, 0, 0]}})

    labels = sorted(s["label"] for s in sent["params"]["data_sources"])
    assert labels == ["Ch1", "Tx1Rx1"]


def test_an_uninitialized_group_is_skipped():
    srv = _server({"BIOPAC": FakeHandler(_sources(1)), "USRP": None})
    assert sorted(s.label for s in srv._refresh_data_sources()) == ["Ch1"]


def test_one_wedged_backend_does_not_lose_every_other_source():
    srv = _server(
        {
            "BIOPAC": FakeHandler(_sources(1)),
            "USRP": FailingHandler({DataSource("USRP", 0, "Tx1Rx1")}),
        }
    )
    assert sorted(s.label for s in srv._refresh_data_sources()) == ["Ch1"]
