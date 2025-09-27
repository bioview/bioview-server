import pytest
from bioview_server.server import Server
from bioview_common import Response


class FakeBackend:
    def __init__(self, discovered=None):
        # discovered is a list of dicts with 'device_id' keys
        self._discovered = discovered or []
        self.inited = False

    def discover_devices(self):
        return self._discovered

    def get_backend_handler(self, configuration, display_data_queue, command_queue, response_queue):
        # Return a handler-like object with initialize/start/stop
        class Handler:
            def __init__(self):
                self.initialized = False

            def initialize(self):
                self.initialized = True

            def start_streaming(self):
                pass

            def stop_streaming(self):
                pass

        return Handler()


def test_connect_devices_warns_on_missing_group(monkeypatch):
    """
    If a whole group requested by client is missing from discovery, server
    should warn (return success) and initialize any present groups.
    """
    s = Server()

    # Prepare AVAILABLE_BACKENDS with a fake backend that discovers one device
    fake = FakeBackend(discovered=[{"device_id": "dev1"}])
    import bioview_server.device as device_mod
    monkeypatch.setattr(device_mod, 'AVAILABLE_BACKENDS', {'fake': fake})

    # Request groups: group_a contains dev1 (present), group_b contains dev2 (missing)
    payload = {
        'group_a': {'dev1': {'backend_type': 'fake', 'samp_rate': 1000}},
        'group_b': {'dev2': {'backend_type': 'fake', 'samp_rate': 1000}},
    }

    resp = s.handle_connect_device(payload)

    # Expect a warning (we decided to change missing-group to warn and proceed)
    assert resp['type'] in (Response.SUCCESS.value, Response.INFO.value)


def test_connect_devices_errors_on_partial_group(monkeypatch):
    """
    If a group has some devices present and some missing, the server should
    error because partial groups produce meaningless data.
    """
    s = Server()

    # fake backend discovers only dev1
    fake = FakeBackend(discovered=[{"device_id": "dev1"}])
    import bioview_server.device as device_mod
    monkeypatch.setattr(device_mod, 'AVAILABLE_BACKENDS', {'fake': fake})

    # group_c expects both dev1 and dev2 -> partial
    payload = {
        'group_c': {'dev1': {'backend_type': 'fake', 'samp_rate': 1000}, 'dev2': {'backend_type': 'fake', 'samp_rate': 1000}}
    }

    resp = s.handle_connect_device(payload)

    assert resp['type'] == Response.ERROR.value
