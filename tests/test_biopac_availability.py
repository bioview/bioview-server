"""Whether a configured BIOPAC group counts as attached hardware.

A BIOPAC group's hardware keys are labels the user picked ("BIOPAC_MP36");
discovery names the unit after its Windows device name ("BIOPAC MP36 USB Data
Acquisition Unit"). Requiring those to match marked a physically connected unit
as unavailable, so the Monitor would not discover or initialize it.
"""

from bioview_common import Configuration, DeviceStatus

import bioview_server.server as server_mod
from bioview_server.server import Server


BIOPAC_CFG = {
    "BIOPAC": {
        "type": "BIOPAC",
        "model": "MP36",
        "samp_rate": 1000,
        "channels": [1, 1, 0, 0],
        "hardware": {"BIOPAC_MP36": {"channels": [1, 1, 0, 0]}},
    }
}


class FakeBackend:
    def __init__(self, found):
        self._found = found

    def discover_devices(self):
        return self._found


def _states(monkeypatch, backends, config=BIOPAC_CFG):
    srv = Server(local_only=True, control_port=0, data_port=0)
    srv.config = Configuration.from_dict(dict(config))
    monkeypatch.setattr(server_mod, "AVAILABLE_BACKENDS", backends)
    srv._discover_devices({})
    return srv.device_group_states


def test_an_attached_unit_is_available_though_its_name_differs_from_the_config(
    monkeypatch
):
    backends = {
        "biopac": FakeBackend(
            {
                "BIOPAC MP36 USB Data Acquisition Unit": {
                    "name": "BIOPAC MP36 USB Data Acquisition Unit"
                }
            }
        )
    }
    assert _states(monkeypatch, backends)["BIOPAC"] == DeviceStatus.AVAILABLE.value


def test_no_biopac_attached_is_unavailable(monkeypatch):
    assert _states(monkeypatch, {"biopac": FakeBackend({})})["BIOPAC"] == (
        DeviceStatus.UNAVAILABLE.value
    )


def test_another_backends_devices_do_not_pass_for_biopac_hardware(monkeypatch):
    """Every backend's results land in one cache; availability must not be read
    from the combined pool, or a machine with only a virtual device would look
    like it had a BIOPAC unit attached."""
    backends = {
        "biopac": FakeBackend({}),
        "dummy": FakeBackend({"DummyVirtual": {"name": "DummyVirtual"}}),
    }
    assert _states(monkeypatch, backends)["BIOPAC"] == DeviceStatus.UNAVAILABLE.value


def test_a_backend_that_raises_leaves_biopac_unavailable_rather_than_crashing(
    monkeypatch
):
    class Exploding:
        @staticmethod
        def discover_devices():
            raise RuntimeError("driver on fire")

    backends = {"biopac": Exploding()}
    assert _states(monkeypatch, backends)["BIOPAC"] == DeviceStatus.UNAVAILABLE.value
