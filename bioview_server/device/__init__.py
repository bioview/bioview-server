# Try to load all backends and provide
import multiprocessing as mp
import sys

from bioview_common import DeviceType, log_print
from bioview_common.utils.logs import suppress_stdout


__all__ = []

AVAILABLE_BACKENDS = {}

# Backends that failed to load, mapped to why. Reported to the Configurator
# alongside the device list, since a GUI-spawned server has no visible stdout.
UNAVAILABLE_BACKENDS = {}


def _backend_unavailable(device_type, error):
    """Record why a backend could not be loaded, and say so on stdout."""
    UNAVAILABLE_BACKENDS[device_type] = str(error)
    print(f"{device_type} backend not available: {error}")


try:
    from . import usrp

    # usrp/__init__ resolves its heavy attributes lazily, so importing the
    # package alone touches no UHD. Import through to utils, which is what
    # actually loads the bindings: a broken or absent UHD then fails here, with
    # a reason, instead of at first use inside a device subprocess.
    from .usrp.utils import discover_devices  # noqa: F401

    __all__.append("usrp")
    AVAILABLE_BACKENDS[DeviceType.USRP.value] = usrp
except Exception as e:
    _backend_unavailable(DeviceType.USRP.value, e)

try:
    # Ensure platform is windows
    if sys.platform != "win32":
        raise OSError(f"Invalid platfrom {sys.platform}. Ensure you are using Windows")

    from . import biopac

    # Ensure mpdev.dll exists
    with suppress_stdout():
        if biopac.load_mpdev_dll() is None:
            raise ValueError("mpdev.dll not found")

    __all__.append("biopac")
    AVAILABLE_BACKENDS[DeviceType.BIOPAC.value] = biopac
except Exception as e:
    _backend_unavailable(DeviceType.BIOPAC.value, e)

try:
    from . import microphone

    # Importing the package alone touches no PortAudio: utils resolves
    # sounddevice lazily. Probe through to it so a missing sounddevice or an
    # unloadable PortAudio fails here, with a reason, rather than inside a
    # device subprocess at Connect.
    microphone.check_available()

    __all__.append("microphone")
    AVAILABLE_BACKENDS[DeviceType.MICROPHONE.value] = microphone
except Exception as e:
    _backend_unavailable(DeviceType.MICROPHONE.value, e)


def backend_report() -> dict:
    """Every backend and whether it loaded, as ``{type: {available, error}}``.

    The server hands this to a client the moment it authenticates, so a UHD
    that does not match its bindings or a backend whose driver is missing
    reaches the operator as one explained failure -- in the Monitor as much as
    in the Configurator -- instead of a line on a stdout nobody is reading.
    """
    report = {
        device_type: {"available": True, "error": ""}
        for device_type in AVAILABLE_BACKENDS
    }
    for device_type, reason in UNAVAILABLE_BACKENDS.items():
        report[device_type] = {"available": False, "error": str(reason)}
    return report


def _usrp_handler(backend, device_id, device_cfg, queues, discovered_devices):
    """The USRP group is the one backend whose constructor takes more than the
    group config: it is built per radio, so the hardware block is split out."""
    group_cfg = device_cfg.to_dict()
    hardware = group_cfg.get("hardware")
    devices = (
        {name: dict(hw) for name, hw in hardware.items()}
        if hardware
        else {device_id: group_cfg}
    )

    return backend.USRPBackend(
        group_id=device_id,
        samp_rate=device_cfg.get_param("samp_rate"),
        devices=devices,
        group_config=group_cfg,
        display_ds=device_cfg.get_param("disp_ds", 10),
        display_imaginary=device_cfg.get_param("display_imaginary", False),
        save_ds=device_cfg.get_param("save_ds", 1),
        save_iq=device_cfg.get_param("save_iq", False),
        save_imaginary=device_cfg.get_param("save_imaginary", True),
        discovered_devices=discovered_devices,
        **queues,
    )


def _group_config_handler(attribute):
    """Factory for a backend built from nothing but its group config.

    BIOPAC and the microphone are constructed identically; naming the class
    keeps that one shape rather than repeating the call per device type.
    """

    def build(backend, device_id, device_cfg, queues, discovered_devices):
        return getattr(backend, attribute)(
            group_id=device_id,
            group_config=device_cfg.to_dict(),
            discovered_devices=discovered_devices,
            **queues,
        )

    return build


#: device_type -> callable building that backend's handler. Registering here is
#: what makes a loaded backend usable; the test suite adds its own the same way.
HANDLER_FACTORIES = {
    DeviceType.USRP.value: _usrp_handler,
    DeviceType.BIOPAC.value: _group_config_handler("BIOPACBackend"),
    DeviceType.MICROPHONE.value: _group_config_handler("MicrophoneBackend"),
}


def get_device_handler(
    device_id,
    device_cfg,
    response_queue: mp.Queue,
    data_output_queue: mp.Queue,
    logger=None,
    discovered_devices: dict = None,
    save_output_queue: mp.Queue = None,
):
    device_type = device_cfg.get_param("device_type")

    build = HANDLER_FACTORIES.get(device_type)
    if build is None:
        log_print(logger, "error", f"Unsupported device type: {device_type}")
        return None

    backend = AVAILABLE_BACKENDS.get(device_type)
    if backend is None:
        log_print(logger, "warning", f"Backend not available for {device_type}")
        return None

    queues = {
        "response_queue": response_queue,
        "data_output_queue": data_output_queue,
        "save_output_queue": save_output_queue,
    }
    return build(backend, device_id, device_cfg, queues, discovered_devices)


__all__ = [
    "AVAILABLE_BACKENDS",
    "UNAVAILABLE_BACKENDS",
    "backend_report",
    "get_device_handler",
    "HANDLER_FACTORIES",
]
