"""A fake device, registered into the server for the duration of a test run.

The server ships three real backends (USRP, BIOPAC, microphone), every one of
which needs hardware attached. This package supplies a fourth that needs none,
so the whole connect -> stream -> display -> save path stays covered on an
ordinary development machine.

It lives under ``tests/`` rather than in ``bioview_server`` on purpose: a test
double that ships is a device the operator can pick by mistake, and it showed up
in the Configurator's device list. Nothing in ``bioview_server`` imports this;
``install()`` below reaches in through the two registries the real backends use
-- ``AVAILABLE_BACKENDS`` / ``HANDLER_FACTORIES`` on the server, and the
configuration registry in bioview-common -- so the fake travels the same code
path a real device does, discovery included.
"""

from bioview_common.datatypes.configuration.config import (
    BaseConfig,
    merged_with_defaults,
    register_device_configuration,
)

from .backend import FakeBackend, SineWaveWorker


#: Wire-format ``type`` and backend ``device_type`` for the fake. Neither
#: appears in the shipped enums; the parser learns them from ``install()``.
FAKE_CFG_TYPE = "FAKE"
FAKE_DEVICE_TYPE = "fake"

#: What ``discover_devices`` reports. A config group (or a hardware key inside
#: one) named after any of these is found by the server's ordinary discovery
#: matching, so the fake needs no special case in the availability rules.
FAKE_DEVICE_NAMES = ("FakeDevice", "FakeDevice_1", "FakeDevice_2")


BASE_FAKE_CONFIG = {
    "samp_rate": 500,
    "num_channels": 4,
    "signal_freq": 1.0,
    "amplitude": 1.0,
    "noise_std": 0.0,
    "chunk_duration": 0.05,
    # RF simulation (optional -- enables MIMO / DPIC / calibration testing)
    "signal_scheme": "cw",
    "tx_gain": [30, 30],
    "tx_amplitude": [1, 1],
    "tx_phase": [0.0, 0.0],
    "if_freq": [100e3, 110e3],
    "if_filter_bw": 5e3,
    "save_ds": 100,
    "disp_ds": 10,
    "calibration": {
        "enabled": False,
        "shape": "triangle",
        "num_pulses": 5,
        "pulse_duration_s": 0.1,
        "packet_spacing_s": 1.0,
        "envelope_freq_hz": 10.0,
        "modulation_depth": 0.2,
        "envelope_offset": 0.0,
        "inject_channels": [0],
        "record_reference": True,
    },
    "dpic_balance": {
        "auto_on_start": False,
        "amp_target": 0.5,
        "coarse_phase_step_deg": 6.0,
        "coarse_amp_step": 0.05,
        "coarse_probe_amplitude": 0.1,
        "phase_step_deg": 0.2,
        "amp_step": 0.001,
        "settle_time_s": 0.1,
        "parallel_devices": True,
    },
    "channel_map": None,
    "hardware": None,
    "rf_simulation": {
        "cross_coupling": 0.08,
        "on_axis_gain": 0.4,
        "direct_leak": 0.4,
        "dpic_coupling": 0.4,
    },
}


class FakeConfiguration(BaseConfig):
    """Reads a ``"type": "FAKE"`` block.

    Without ``hardware``/``channel_map`` the backend synthesizes phase-shifted
    sine waves; with them it runs the full USRP pipeline against a virtual MIMO
    channel.
    """

    def __init__(self, config_dict: dict):
        super().__init__(BASE_FAKE_CONFIG)
        self.cfg_type = FAKE_CFG_TYPE

        for key, value in merged_with_defaults(
            BASE_FAKE_CONFIG, config_dict or {}
        ).items():
            setattr(self, key, value)

        self.device_type = FAKE_DEVICE_TYPE


def discover_devices(logger=None):
    """Report the fake units as attached, the way a real backend would."""
    return [
        {
            "name": name,
            "type": FAKE_DEVICE_TYPE,
            "device_type": FAKE_DEVICE_TYPE,
            "serial": f"fake-{index}",
        }
        for index, name in enumerate(FAKE_DEVICE_NAMES)
    ]


#: No Configurator-editable properties, matching the simplest real backend.
EDITABLE_PROPERTIES = {}


def set_device_config(device_info: dict, new_config: dict, logger=None):
    return False, "This device type has no editable properties."


def _build_handler(backend, device_id, device_cfg, queues, discovered_devices):
    return FakeBackend(
        group_id=device_id,
        group_config=device_cfg.to_dict(),
        **queues,
    )


def install():
    """Register the fake with the server and the configuration parser.

    Idempotent, so an autouse fixture can call it per test.
    """
    import sys

    from bioview_server.device import AVAILABLE_BACKENDS, HANDLER_FACTORIES

    module = sys.modules[__name__]
    AVAILABLE_BACKENDS[FAKE_DEVICE_TYPE] = module
    HANDLER_FACTORIES[FAKE_DEVICE_TYPE] = _build_handler
    register_device_configuration(FAKE_DEVICE_TYPE, FAKE_CFG_TYPE, FakeConfiguration)


__all__ = [
    "BASE_FAKE_CONFIG",
    "EDITABLE_PROPERTIES",
    "FAKE_CFG_TYPE",
    "FAKE_DEVICE_NAMES",
    "FAKE_DEVICE_TYPE",
    "FakeBackend",
    "FakeConfiguration",
    "SineWaveWorker",
    "discover_devices",
    "install",
    "set_device_config",
]
