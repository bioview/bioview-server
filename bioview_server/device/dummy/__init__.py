from .backend import DummyBackend, SineWaveWorker


def discover_devices():
    """Report the virtual dummy backend as present on the server."""
    return [
        {
            "name": "DummyVirtual",
            "type": "dummy",
            "device_type": "dummy",
            "serial": "virtual",
        }
    ]


def update_device_firmware(*args, **kwargs):
    # No-op for the virtual device; present for API parity with real backends.
    return True


__all__ = ["DummyBackend", "SineWaveWorker", "discover_devices", "update_device_firmware"]
