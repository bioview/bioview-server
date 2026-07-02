import types

import numpy as np

from bioview_common import (
    apply_filter,
    emit_signal,
    get_cache_file,
    get_filter,
    get_unique_path,
    suppress_stdout,
)
from bioview_server.device import AVAILABLE_BACKENDS


def test_available_backends_structure():
    # AVAILABLE_BACKENDS maps a backend type name -> backend module. Every module
    # must expose discover_devices() (the contract the server relies on) and the
    # always-available dummy backend must be present so hardware-free streaming
    # works everywhere.
    assert isinstance(AVAILABLE_BACKENDS, dict)
    assert "dummy" in AVAILABLE_BACKENDS
    for name, module in AVAILABLE_BACKENDS.items():
        assert isinstance(name, str)
        assert isinstance(module, types.ModuleType)
        assert hasattr(module, "discover_devices")


def test_suppress_stdout_and_emit_signal():
    with suppress_stdout():
        print("this should be suppressed")

    # emit_signal should silently ignore None
    emit_signal(None)

    # emit_signal should call the provided callable
    called = {"v": False}

    def cb(x):
        called["v"] = x

    emit_signal(cb, True)
    assert called["v"] is True


def test_get_cache_file_and_unique_path(tmp_path, monkeypatch):
    # Point HOME at a temp dir so the cache file is created somewhere writable.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows equivalent

    f = get_cache_file("testfile.txt")
    assert f.exists()

    up = get_unique_path(tmp_path, "file.txt")
    assert str(up).endswith("file.txt")


def test_filtering_roundtrip():
    samp_rate = 1000
    filt = get_filter([1, 100], samp_rate)
    data = np.random.randn(1000)
    filtered, _zf = apply_filter(data, filt)
    assert filtered.shape == data.shape
