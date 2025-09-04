# Try to load all backends and provide
import sys

from bioview_server.utils import suppress_stdout


__all__ = []

AVAILABLE_BACKENDS = {}

try:
    # Ensure uhd is available

    # Ensure device is importable
    from . import usrp

    __all__.append["usrp"]
    AVAILABLE_BACKENDS["usrp"] = usrp
except Exception as e:
    print(f"USRP backend not available: {e}")

try:
    # Ensure platform is windows
    if sys.platform != "win32":
        raise OSError(f"Invalid platfrom {sys.platform}. Ensure you are using Windows")

    from . import biopac

    # Ensure mpdev.dll exists
    with suppress_stdout():
        if biopac.load_mpdev_dll() is None:
            raise ValueError("mpdev.dll not found")

    __all__.append["biopac"]
    AVAILABLE_BACKENDS["biopac"] = biopac
except Exception as e:
    print(f"BIOPAC backend not available: {e}")

__all__ = ["AVAILABLE_BACKENDS"]
