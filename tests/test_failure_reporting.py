"""The server must say *why* a device is unusable, not just that it is.

A GUI-spawned server has its output detached, so anything explained only in the
server's own log never reaches the user. Failure reasons therefore travel back
with the device status.
"""
import pytest
from bioview_common import Command, Configuration, DeviceStatus, Response

import bioview_server.server as server_mod
from bioview_server.device.biopac.constants import describe_biopac_code
from bioview_server.server import Server


class TestBiopacErrorCodes:
    def test_a_driver_failure_explains_itself_and_what_to_do(self):
        described = describe_biopac_code(2)
        assert "MPDRVERR" in described
        assert "code 2" in described
        assert "driver" in described.lower()

    def test_a_busy_unit_is_distinguished_from_a_missing_one(self):
        assert "already connected" in describe_biopac_code(3)
        assert "no MP unit is connected" in describe_biopac_code(5)

    def test_an_unrecognised_code_still_reports_its_number(self):
        assert "99" in describe_biopac_code(99)

    def test_connect_failure_raises_with_the_decoded_reason(self):
        from bioview_server.device.biopac import utils

        class FakeDll:
            @staticmethod
            def connectMPDev(*_args):
                return 2  # MPDRVERR

        with pytest.raises(Exception) as excinfo:
            utils.connect_biopac_device(FakeDll())

        assert "MPDRVERR" in str(excinfo.value)
        assert "reinstall" in str(excinfo.value).lower()


class TestDeviceErrorsReachTheClient:
    def test_device_status_carries_the_reason_a_group_failed(self, client):
        resp_type, payload = client.command(Command.GET_DEVICE_STATUS)
        assert resp_type == Response.SUCCESS.name
        assert "device_errors" in payload, "the client reads this to explain a failure"

    def test_discovery_records_why_a_group_is_unavailable(self, monkeypatch):
        srv = Server(local_only=True, control_port=0, data_port=0)
        srv.config = Configuration.from_dict(
            {
                "BIOPAC": {
                    "type": "BIOPAC",
                    "model": "MP36",
                    "samp_rate": 1000,
                    "channels": [1, 0, 0, 0],
                    "hardware": {"BIOPAC_MP36": {"channels": [1, 0, 0, 0]}},
                }
            }
        )

        class NoDevices:
            @staticmethod
            def discover_devices():
                return {}

        monkeypatch.setattr(server_mod, "AVAILABLE_BACKENDS", {"biopac": NoDevices()})
        srv._discover_devices({})

        assert srv.device_group_states["BIOPAC"] == DeviceStatus.UNAVAILABLE.value
        assert "no BIOPAC unit was found" in srv.device_group_errors["BIOPAC"]

    def test_an_unloadable_backend_is_named_as_the_reason(self, monkeypatch):
        srv = Server(local_only=True, control_port=0, data_port=0)
        srv.config = Configuration.from_dict(
            {
                "BIOPAC": {
                    "type": "BIOPAC",
                    "model": "MP36",
                    "samp_rate": 1000,
                    "channels": [1, 0, 0, 0],
                    "hardware": {},
                }
            }
        )
        monkeypatch.setattr(server_mod, "AVAILABLE_BACKENDS", {})
        monkeypatch.setattr(
            server_mod, "UNAVAILABLE_BACKENDS", {"biopac": "No module named 'wmi'"}
        )
        srv._discover_devices({})

        reason = srv.device_group_errors["BIOPAC"]
        assert "backend is not available" in reason
        assert "wmi" in reason

    def test_a_successful_discovery_records_no_error(self, monkeypatch):
        srv = Server(local_only=True, control_port=0, data_port=0)
        srv.config = Configuration.from_dict(
            {
                "BIOPAC": {
                    "type": "BIOPAC",
                    "model": "MP36",
                    "samp_rate": 1000,
                    "channels": [1, 0, 0, 0],
                    "hardware": {"BIOPAC_MP36": {"channels": [1, 0, 0, 0]}},
                }
            }
        )

        class OneUnit:
            @staticmethod
            def discover_devices():
                return {"BIOPAC MP36 USB Data Acquisition Unit": {"name": "unit"}}

        monkeypatch.setattr(server_mod, "AVAILABLE_BACKENDS", {"biopac": OneUnit()})
        srv._discover_devices({})

        assert srv.device_group_states["BIOPAC"] == DeviceStatus.AVAILABLE.value
        assert "BIOPAC" not in srv.device_group_errors


class TestMemoryIntegrityDetection:
    """Recognising the case where Windows itself is blocking the driver."""

    def test_the_server_attributes_a_driver_error_to_memory_integrity(self, monkeypatch):
        """The backend subprocess reports what the device said; only the server
        can see how the machine is configured, and it is the safe place to look
        -- the query has been seen to hang inside a subprocess and stall the
        initialization it was meant to explain."""
        from bioview_server.device.biopac import utils

        monkeypatch.setattr(utils, "memory_integrity_state", lambda: (True, True))
        srv = Server(local_only=True, control_port=0, data_port=0)

        enriched = srv._explain_device_failure(
            "BIOPAC connection failed: MPDRVERR (code 2)"
        )
        assert "Memory Integrity" in enriched

        # and the shared catalogue turns that into advice for the user
        from bioview_common import explain

        assert explain(enriched).id == "driver-blocked-by-memory-integrity"

    def test_a_failure_that_is_not_a_driver_error_is_left_alone(self):
        srv = Server(local_only=True, control_port=0, data_port=0)
        message = "no acquisition channels are enabled"
        assert srv._explain_device_failure(message) == message

    def test_the_connect_call_itself_stays_free_of_machine_queries(self, monkeypatch):
        """connect_biopac_device runs in the backend subprocess, so it must not
        reach for anything that can block."""
        from bioview_server.device.biopac import utils

        def _must_not_be_called():
            raise AssertionError("the subprocess must not query machine state")

        monkeypatch.setattr(utils, "memory_integrity_state", _must_not_be_called)

        class FakeDll:
            @staticmethod
            def connectMPDev(*_args):
                return 2  # MPDRVERR

        with pytest.raises(Exception) as excinfo:
            utils.connect_biopac_device(FakeDll())
        assert "MPDRVERR" in str(excinfo.value)

    def test_a_pending_reboot_is_called_out_rather_than_looking_solved(
        self, monkeypatch
    ):
        """Switched off but still running is the state that confuses people most:
        the toggle says off while the driver is still being refused."""
        from bioview_server.device.biopac import utils

        monkeypatch.setattr(utils, "memory_integrity_state", lambda: (True, False))
        hint = utils.driver_failure_hint()

        assert "still running" in hint
        assert "restart" in hint.lower()

    def test_no_memory_integrity_means_no_speculation(self, monkeypatch):
        from bioview_server.device.biopac import utils

        monkeypatch.setattr(utils, "memory_integrity_state", lambda: (False, False))
        assert utils.driver_failure_hint() == ""

    def test_the_state_probe_never_raises_and_never_hangs(self):
        """It runs while reporting a failure, so it must not become one."""
        import time

        from bioview_server.device.biopac import utils

        utils._hvci_state = None  # bypass the cache to time a real query
        started = time.monotonic()
        running, configured = utils.memory_integrity_state()
        elapsed = time.monotonic() - started

        assert isinstance(running, bool) and isinstance(configured, bool)
        assert elapsed < utils._HVCI_QUERY_TIMEOUT + 2, (
            f"the probe took {elapsed:.1f}s; it is bounded so a hanging WMI "
            "query cannot stall device initialization"
        )

    def test_the_probe_result_is_cached(self):
        """Memory Integrity cannot change without a reboot."""
        from bioview_server.device.biopac import utils

        utils._hvci_state = (True, False)
        assert utils.memory_integrity_state() == (True, False)
        utils._hvci_state = None
