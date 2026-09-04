"""NumPy's C-API modules must be loaded before any USRP worker thread runs.

``libpyuhd`` is a pybind11 module built against NumPy 1.x, and pybind11
bootstraps the NumPy C-API lazily: the first ``tx_streamer.send()`` or
``rx_streamer.recv()`` in the process runs ``import numpy.core.multiarray`` from
inside the extension, with the GIL held.

Under NumPy 1.x that is a ``sys.modules`` hit. Under NumPy 2 ``numpy.core`` is a
compatibility shim that ``import numpy`` does not load, so the import machinery
genuinely runs -- and ``_start_streaming`` resumes the transmit and receive
workers in the same instant, so both threads enter that first import together
from inside a C extension and deadlock on the import lock without ever dropping
the GIL. The entire backend process freezes: even a bare ``time.sleep(0.35)`` on
the main thread never returns, START_STREAMING is never answered, and the server
times the device out and stops every other device in the session with it.

Observed as: USRP would not stream at all, alone or alongside a healthy BIOPAC,
after the environment picked up NumPy 2.
"""

import sys

import pytest


def test_importing_the_usrp_backend_warms_the_numpy_c_api():
    pytest.importorskip("uhd", reason="UHD bindings not installed")
    import bioview_server.device.usrp.backend  # noqa: F401

    # Whatever pybind11 reaches for on that first send()/recv() must already be
    # resolved, so no worker thread ever enters the import machinery.
    assert "numpy.core.multiarray" in sys.modules
    assert "numpy.core.umath" in sys.modules


def test_warm_up_survives_numpy_dropping_the_compat_shim(monkeypatch):
    """NumPy 3 may remove ``numpy.core`` outright; that must not break start-up."""
    import importlib

    from bioview_server.device.usrp import backend

    def _gone(name):
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", _gone)

    # Must not raise: by then the bindings that needed the shim are gone too.
    backend._warm_numpy_c_api()
