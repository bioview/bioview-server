from .backend import DummyBackend, SineWaveWorker


def discover_devices():
    """Virtual devices have nothing to physically probe. Discovery returns an
    empty list; the dummy backend always initializes successfully regardless of
    the reported availability, which is enough to exercise the full pipeline."""
    return []


def update_device_firmware(*args, **kwargs):
    # No-op for the virtual device; present for API parity with real backends.
    return True


__all__ = ["DummyBackend", "SineWaveWorker", "discover_devices", "update_device_firmware"]
