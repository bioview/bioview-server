"""Host audio-input discovery and device resolution (PortAudio, via sounddevice)."""

from __future__ import annotations

import contextlib

from bioview_common import log_print


def _sounddevice():
    """Import sounddevice on demand."""
    import sounddevice as sd

    return sd


def check_available() -> None:
    """Raise unless PortAudio can actually be reached."""
    sd = _sounddevice()
    sd.query_devices()


def sanitize_name(name: str, index: int) -> str:
    """A config-safe key for a host device name."""
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in (name or ""))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or f"Microphone_{index}"


def list_input_devices(logger=None) -> list[dict]:
    """Every host device with at least one input channel."""
    try:
        sd = _sounddevice()
        raw = sd.query_devices()
    except Exception as e:
        log_print(logger, "warning", f"Audio device enumeration failed: {e}")
        return []

    devices = []
    for index, entry in enumerate(raw):
        max_in = int(entry.get("max_input_channels", 0) or 0)
        if max_in <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": entry.get("name", f"Microphone_{index}"),
                "max_input_channels": max_in,
                "default_samplerate": float(entry.get("default_samplerate", 0) or 0),
                "hostapi": int(entry.get("hostapi", 0) or 0),
            }
        )
    return devices


_LOOPBACK_MARKERS = ("stereo mix", "what u hear", "loopback", "wave out mix")


def _looks_like_loopback(name: str) -> bool:
    lowered = (name or "").lower()
    return any(marker in lowered for marker in _LOOPBACK_MARKERS)


def _input_opens(entry: dict, logger=None) -> bool:
    """True when this input opens at any rate worth trying."""
    rates = []
    native = float(entry.get("default_samplerate", 0) or 0)
    if native > 0:
        rates.append(native)
    rates.extend(float(rate) for rate in FALLBACK_SAMPLE_RATES)

    for rate in rates:
        if supports_input(entry["index"], 1, rate):
            return True
    log_print(
        logger,
        "debug",
        f"Audio input {entry['name']!r} enumerates but will not open at any "
        "supported rate; skipping it",
    )
    return False


def _default_input_index(logger=None):
    with contextlib.suppress(Exception):
        sd = _sounddevice()
        default = sd.default.device
        index = default[0] if isinstance(default, list | tuple) else default
        if index is not None and int(index) >= 0:
            return int(index)
    log_print(logger, "debug", "No PortAudio default input device")
    return None


def discover_devices(logger=None) -> dict:
    """Discover host audio inputs; returns ``{hardware_key: device_info}``."""
    discovered = {}
    default_index = _default_input_index(logger)

    for entry in list_input_devices(logger):
        key = sanitize_name(entry["name"], entry["index"])
        if key in discovered:
            key = f"{key}_{entry['index']}"
        discovered[key] = {
            "name": entry["name"],
            "type": "microphone",
            "device_type": "microphone",
            "serial": f"portaudio:{entry['index']}",
            "index": entry["index"],
            "max_input_channels": entry["max_input_channels"],
            "default_samplerate": entry["default_samplerate"],
            "is_default": entry["index"] == default_index,
        }
    return discovered


def resolve_input_device(requested, logger=None):
    """Turn a configured ``device`` value into a PortAudio device index."""
    devices = list_input_devices(logger)
    if not devices:
        raise RuntimeError(
            "no audio input device was found. Check that a microphone is "
            "attached and enabled in the operating system's sound settings"
        )

    if requested is None or (
        isinstance(requested, str) and requested.strip().lower() in {"", "default"}
    ):
        default_index = _default_input_index(logger)
        if default_index is not None:
            return default_index
        candidates = [e for e in devices if not _looks_like_loopback(e["name"])]
        chosen = next(
            (e for e in candidates if _input_opens(e, logger)),
            None,
        )
        if chosen is None:
            skipped = ", ".join(repr(e["name"]) for e in candidates)
            raise RuntimeError(
                "no usable audio input was found. "
                + (
                    f"These enumerate but refuse to open: {skipped} -- on "
                    "Windows that usually means nothing is plugged into the "
                    "jack, or the device is disabled in Sound settings. "
                    if skipped
                    else ""
                )
                + "Attach a microphone, or name an input explicitly with "
                '"device" in the configuration. Available inputs: '
                + str([e["name"] for e in devices])
            )
        log_print(
            logger,
            "warning",
            "No default audio input is set on this machine; using "
            f"{chosen['name']!r}. Name the input explicitly in the "
            "configuration to be sure of what is being recorded",
        )
        return chosen["index"]

    if isinstance(requested, int) or (
        isinstance(requested, str) and requested.strip().lstrip("-").isdigit()
    ):
        index = int(requested)
        for entry in devices:
            if entry["index"] == index:
                return index
        raise RuntimeError(
            f"audio device index {index} is not an input device. "
            f"Available: {sorted((e['index'], e['name']) for e in devices)}"
        )

    needle = str(requested).strip().lower()
    for entry in devices:
        if sanitize_name(entry["name"], entry["index"]).lower() == needle:
            return entry["index"]
    for entry in devices:
        if needle in entry["name"].lower():
            return entry["index"]

    raise RuntimeError(
        f"no audio input device matches {requested!r}. "
        f"Available: {[e['name'] for e in devices]}"
    )


FALLBACK_SAMPLE_RATES = (48000, 44100, 32000, 22050, 16000, 11025, 8000)


def describe_input(device_index) -> str:
    """``"FrontMic (...)" (index 10)`` for an error message, or just the index."""
    with contextlib.suppress(Exception):
        sd = _sounddevice()
        name = sd.query_devices(device_index).get("name")
        if name:
            return f"{name!r} (index {device_index})"
    return f"audio input index {device_index}"


def supports_input(device_index, channels: int, samp_rate: float) -> bool:
    """True when PortAudio will open this input at these settings."""
    try:
        sd = _sounddevice()
        stream = sd.InputStream(
            device=device_index,
            channels=int(channels),
            samplerate=float(samp_rate),
            dtype="float32",
            callback=lambda *_args: None,
        )
    except Exception:
        return False

    with contextlib.suppress(Exception):
        stream.close()
    return True


def negotiate_samplerate(device_index, channels: int, requested: float, logger=None):
    """Return a rate this input will actually run at, preferring ``requested``."""
    requested = float(requested)
    if supports_input(device_index, channels, requested):
        return requested

    candidates = []
    with contextlib.suppress(Exception):
        sd = _sounddevice()
        native = float(sd.query_devices(device_index).get("default_samplerate", 0) or 0)
        if native > 0:
            candidates.append(native)
    candidates.extend(float(rate) for rate in FALLBACK_SAMPLE_RATES)

    for rate in candidates:
        if rate == requested:
            continue
        if supports_input(device_index, channels, rate):
            log_print(
                logger,
                "warning",
                f"Audio input will not run at {requested:.0f} Hz; capturing at "
                f"{rate:.0f} Hz instead. The recording carries the rate it was "
                "actually captured at",
            )
            return rate

    return requested


def build_hardware_dict_from_group(group_config: dict, group_id: str) -> dict:
    """Normalize a group to the nested ``hardware`` shape the backend reads."""
    hardware = group_config.get("hardware")
    if hardware:
        return {name: dict(entry) for name, entry in hardware.items()}

    entry = {
        k: v
        for k, v in group_config.items()
        if k in {"samp_rate", "channels", "device", "blocksize", "gain", "labels"}
    }
    device_name = group_config.get("device_name") or group_id
    return {device_name: entry}


def resolve_hardware_entry(hardware: dict, discovered_devices: dict | None = None):
    """Pick the hardware entry to drive, and the key it was found under."""
    if not hardware:
        return None, {}
    if len(hardware) == 1:
        key = next(iter(hardware))
        return key, dict(hardware[key])
    if discovered_devices:
        for key, entry in hardware.items():
            if key in discovered_devices:
                return key, dict(entry)
    key = next(iter(hardware))
    return key, dict(hardware[key])
