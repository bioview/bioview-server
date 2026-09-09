# Try to load all backends and provide
import multiprocessing as mp
import sys

from bioview_common import SUPPORTED_DEVICES, DeviceType, log_print
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

try:
    # Virtual device: always available, no hardware or platform requirements.
    from . import dummy

    __all__.append("dummy")
    AVAILABLE_BACKENDS[DeviceType.DUMMY.value] = dummy
except Exception as e:
    _backend_unavailable(DeviceType.DUMMY.value, e)


def backend_report(include_virtual: bool = False) -> dict:
    """Every backend and whether it loaded, as ``{type: {available, error}}``.

    The server hands this to a client the moment it authenticates, so a UHD
    that does not match its bindings or a backend whose driver is missing
    reaches the operator as one explained failure -- in the Monitor as much as
    in the Configurator -- instead of a line on a stdout nobody is reading.
    """
    report = {}
    for device_type in AVAILABLE_BACKENDS:
        if device_type == DeviceType.DUMMY.value and not include_virtual:
            continue
        report[device_type] = {"available": True, "error": ""}
    for device_type, reason in UNAVAILABLE_BACKENDS.items():
        if device_type == DeviceType.DUMMY.value and not include_virtual:
            continue
        report[device_type] = {"available": False, "error": str(reason)}
    return report


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

    if device_type not in SUPPORTED_DEVICES:
        log_print(logger, "error", f"Unsupported device type: {device_type}")
        return None
    elif device_type not in AVAILABLE_BACKENDS:
        log_print(logger, "warning", f"Backend not available for {device_type}")
        return None

    match device_type:
        case DeviceType.USRP.value:
            group_cfg = device_cfg.to_dict()
            hardware = group_cfg.get("hardware")
            if hardware:
                devices = {name: dict(hw) for name, hw in hardware.items()}
            else:
                devices = {device_id: group_cfg}

            handler = AVAILABLE_BACKENDS.get(DeviceType.USRP.value).USRPBackend(
                group_id=device_id,
                samp_rate=device_cfg.get_param("samp_rate"),
                devices=devices,
                group_config=group_cfg,
                response_queue=response_queue,
                data_output_queue=data_output_queue,
                save_output_queue=save_output_queue,
                display_ds=device_cfg.get_param("disp_ds", 10),
                display_imaginary=device_cfg.get_param("display_imaginary", False),
                save_ds=device_cfg.get_param("save_ds", 1),
                save_iq=device_cfg.get_param("save_iq", False),
                save_imaginary=device_cfg.get_param("save_imaginary", True),
                discovered_devices=discovered_devices,
            )

        case DeviceType.BIOPAC.value:
            handler = AVAILABLE_BACKENDS.get(DeviceType.BIOPAC.value).BIOPACBackend(
                group_id=device_id,
                response_queue=response_queue,
                data_output_queue=data_output_queue,
                save_output_queue=save_output_queue,
                group_config=device_cfg.to_dict(),
                discovered_devices=discovered_devices,
            )

        case DeviceType.MICROPHONE.value:
            handler = AVAILABLE_BACKENDS.get(
                DeviceType.MICROPHONE.value
            ).MicrophoneBackend(
                group_id=device_id,
                response_queue=response_queue,
                data_output_queue=data_output_queue,
                save_output_queue=save_output_queue,
                group_config=device_cfg.to_dict(),
                discovered_devices=discovered_devices,
            )

        case DeviceType.DUMMY.value:
            handler = AVAILABLE_BACKENDS.get(DeviceType.DUMMY.value).DummyBackend(
                group_id=device_id,
                response_queue=response_queue,
                data_output_queue=data_output_queue,
                save_output_queue=save_output_queue,
                group_config=device_cfg.to_dict(),
            )

        case _:
            handler = None

    return handler


__all__ = [
    "AVAILABLE_BACKENDS",
    "UNAVAILABLE_BACKENDS",
    "backend_report",
    "get_device_handler",
]
