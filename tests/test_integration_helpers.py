import json
import types

import numpy as np
import pytest
from bioview_common import ValidationError

from bioview_server.device import AVAILABLE_BACKENDS
from bioview_common import (
    apply_filter,
    emit_signal,
    get_cache_file,
    get_filter,
    get_unique_path,
    suppress_stdout,
)


def test_available_backends_structure():
    # AVAILABLE_BACKENDS should be a dict
    assert isinstance(AVAILABLE_BACKENDS, dict)
    # Keys (if present) should map to modules
    for k, v in AVAILABLE_BACKENDS.items():
        assert isinstance(k, str)
        assert isinstance(v, types.ModuleType)
        # Module should expose get_backend_handler or similar
        # We only assert presence of a couple of expected
        # attributes to avoid hardware access
        assert (
            hasattr(v, "get_backend_handler")
            or hasattr(v, "BIOPACBackend")
            or hasattr(v, "USRPBackend")
        )


def test_suppress_stdout_and_emit_signal(tmp_path):
    with suppress_stdout():
        print("this should be suppressed")

    # emit_signal should silently ignore None
    emit_signal(None)

    # emit_signal should call function
    called = {"v": False}

    def cb(x):
        called["v"] = x

    emit_signal(cb, True)
    assert called["v"] is True


def test_get_cache_file_and_unique_path(tmp_path, monkeypatch):
    # Use a temporary home directory
    monkeypatch.chdir(tmp_path)
    f = get_cache_file("testfile.txt")
    assert f.exists()

    up = get_unique_path(tmp_path, "file.txt")
    assert str(up).endswith("file.txt")

def test_filtering_roundtrip():
    samp_rate = 1000
    filt = get_filter([1, 100], samp_rate)
    data = np.random.randn(1000)
    filtered, zf = apply_filter(data, filt)
    assert filtered.shape == data.shape
