"""``check_input_settings`` does not predict whether a stream will open.

Measured on a Realtek machine whose only inputs enumerate under WDM-KS:

    FrontMic    1ch @ 44100   check:OK   open:NO  (Invalid device)
    Stereo Mix  1ch @ 44100   check:NO   open:OK

It is wrong in both directions, so the rate negotiated from it was a rate the
device would refuse, and the input chosen from it was an input that could never
be opened. The probe now opens the stream, which is the only question anyone
was ever asking.
"""

import pytest

from bioview_server.device.microphone import utils
from bioview_server.device.microphone.utils import (
    describe_input,
    negotiate_samplerate,
    supports_input,
)


class _FakeStream:
    def __init__(self, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


def _patch_open(monkeypatch, opens):
    """``opens(device, channels, rate) -> bool`` decides what Pa_OpenStream does."""
    created = []

    class _FakeSd:
        @staticmethod
        def InputStream(**kwargs):  # noqa: N802 - mirrors sounddevice's name
            created.append(kwargs)
            if not opens(kwargs["device"], kwargs["channels"], kwargs["samplerate"]):
                raise RuntimeError("Error opening InputStream: Invalid device")
            return _FakeStream(**kwargs)

        @staticmethod
        def check_input_settings(**kwargs):
            raise AssertionError("check_input_settings must not be consulted")

        @staticmethod
        def query_devices(index):
            return {"name": f"Device {index}", "default_samplerate": 44100.0}

    monkeypatch.setattr(utils, "_sounddevice", lambda: _FakeSd)
    return created


def test_the_probe_opens_rather_than_asking(monkeypatch):
    created = _patch_open(monkeypatch, lambda d, c, r: r == 44100.0)

    assert supports_input(10, 1, 44100) is True
    assert supports_input(10, 1, 16000) is False
    # It really went through Pa_OpenStream, with the settings it was asked about.
    assert created[0]["device"] == 10
    assert created[0]["channels"] == 1
    assert created[0]["samplerate"] == 44100.0


def test_the_probe_closes_what_it_opened(monkeypatch):
    """A probe that leaked the stream would hold the device against the open
    that actually matters."""
    opened = []

    class _FakeSd:
        @staticmethod
        def InputStream(**kwargs):  # noqa: N802
            stream = _FakeStream(**kwargs)
            opened.append(stream)
            return stream

    monkeypatch.setattr(utils, "_sounddevice", lambda: _FakeSd)

    assert supports_input(1, 1, 48000) is True
    assert opened and all(stream.closed for stream in opened)


def test_a_close_that_fails_does_not_make_the_probe_lie(monkeypatch):
    class _Stubborn(_FakeStream):
        def close(self):
            raise RuntimeError("device busy")

    class _FakeSd:
        @staticmethod
        def InputStream(**kwargs):  # noqa: N802
            return _Stubborn(**kwargs)

    monkeypatch.setattr(utils, "_sounddevice", lambda: _FakeSd)
    assert supports_input(1, 1, 48000) is True


def test_negotiation_now_follows_what_opens(monkeypatch):
    """The Stereo Mix case: check said no at 44.1 kHz, the device says yes."""
    _patch_open(monkeypatch, lambda d, c, r: r in (44100.0, 48000.0))

    # 16 kHz is refused, the device's native rate is tried first and taken.
    assert negotiate_samplerate(9, 1, 16000) == 44100.0
    # A rate that opens is kept as asked.
    assert negotiate_samplerate(9, 1, 48000) == 48000.0


def test_a_device_that_opens_at_nothing_hands_the_request_back(monkeypatch):
    """So the real open fails naming the rate that was configured."""
    _patch_open(monkeypatch, lambda d, c, r: False)
    assert negotiate_samplerate(10, 1, 16000) == 16000.0


@pytest.mark.parametrize(
    ("index", "expected"),
    [(10, "'Device 10' (index 10)"), (3, "'Device 3' (index 3)")],
)
def test_errors_name_the_device_not_just_its_index(monkeypatch, index, expected):
    _patch_open(monkeypatch, lambda d, c, r: True)
    assert describe_input(index) == expected


def test_describe_input_survives_an_unqueryable_device(monkeypatch):
    class _FakeSd:
        @staticmethod
        def query_devices(index):
            raise RuntimeError("gone")

    monkeypatch.setattr(utils, "_sounddevice", lambda: _FakeSd)
    assert describe_input(10) == "audio input index 10"
