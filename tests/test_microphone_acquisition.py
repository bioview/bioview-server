"""How the microphone backend turns PortAudio callbacks into pipeline chunks.

Everything here runs against a fake PortAudio: the machines these tests run on
have no guaranteed audio input, and a test that silently passes because no
frames ever arrived would be worse than no test. The fake drives the very
callback ``sd.InputStream`` would call, so the path under test is the real one
from the callback inward.
"""

import multiprocessing as mp
import queue
import time

import numpy as np
import pytest

from bioview_server.device.microphone.acquire import (
    CAPTURE_QUEUE_DEPTH,
    MicrophoneAcquisitionWorker,
)
from bioview_server.device.microphone.backend import MicrophoneBackend
from bioview_server.device.microphone.utils import (
    negotiate_samplerate,
    resolve_input_device,
    sanitize_name,
)


class FakeStatus:
    """The ``CallbackFlags`` object PortAudio hands the callback."""

    def __init__(self, input_overflow=False):
        self.input_overflow = input_overflow

    def __bool__(self):
        return self.input_overflow


def drain(q, timeout=1.0):
    """Every chunk currently on a queue, waiting briefly for the first."""
    out = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            out.append(q.get(timeout=0.05))
        except queue.Empty:
            if out:
                break
    return out


@pytest.fixture
def worker():
    w = MicrophoneAcquisitionWorker(
        channels=2,
        samp_rate=16000,
        display_queue=queue.Queue(),
        save_queue=queue.Queue(),
    )
    w.start()
    w.resume()
    yield w
    w.stop()
    w.join(timeout=2)


def test_callback_frames_are_transposed_to_channel_rows(worker):
    """PortAudio hands back (frames, channels); the pipeline wants the transpose.

    A missed transpose is not an error anywhere downstream -- it produces a
    plausible-looking array with the axes swapped, which only shows up as a
    recording whose channel count changes with its chunk size.
    """
    indata = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    worker.callback(indata, 3, None, FakeStatus())

    chunks = drain(worker.display_queue)
    assert len(chunks) == 1
    assert chunks[0].shape == (2, 3)
    np.testing.assert_allclose(chunks[0][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(chunks[0][1], [10.0, 20.0, 30.0])


def test_callback_copies_the_buffer_it_is_given(worker):
    """PortAudio reuses its input buffer between callbacks.

    Keeping a view of it means every queued chunk ends up holding whatever the
    newest callback wrote -- a recording of the last 100 ms, repeated.
    """
    indata = np.array([[1.0, 1.0]], dtype=np.float32)
    worker.callback(indata, 1, None, FakeStatus())
    indata[:] = 99.0  # PortAudio overwriting its buffer

    chunks = drain(worker.display_queue)
    assert len(chunks) == 1
    np.testing.assert_allclose(chunks[0], [[1.0], [1.0]])


def test_gain_is_applied_to_both_the_saved_and_displayed_copy(worker):
    worker.gain = 4.0
    worker.callback(np.array([[0.25, 0.5]], dtype=np.float32), 1, None, FakeStatus())

    displayed = drain(worker.display_queue)
    saved = drain(worker.save_queue)
    np.testing.assert_allclose(displayed[0], [[1.0], [2.0]])
    np.testing.assert_allclose(saved[0], [[1.0], [2.0]])


def test_saved_chunk_is_not_the_displayed_array(worker):
    """The two queues must not hand out the same object.

    The display path is allowed to mutate what it is given; sharing the array
    would corrupt the recording rather than the plot.
    """
    worker.callback(np.array([[1.0, 2.0]], dtype=np.float32), 1, None, FakeStatus())
    displayed = drain(worker.display_queue)[0]
    saved = drain(worker.save_queue)[0]

    displayed[:] = 0.0
    np.testing.assert_allclose(saved, [[1.0], [2.0]])


def test_a_full_capture_queue_drops_rather_than_blocking():
    """The callback runs on PortAudio's own thread and must never block there.

    Blocking it stalls the audio device itself, which loses far more than the
    chunk that could not be queued.
    """
    # Never started, so nothing drains the capture queue.
    w = MicrophoneAcquisitionWorker(
        channels=1, samp_rate=16000, display_queue=queue.Queue()
    )
    frame = np.zeros((1, 1), dtype=np.float32)
    for _ in range(CAPTURE_QUEUE_DEPTH + 5):
        w.callback(frame, 1, None, FakeStatus())

    assert w.capture_queue.full()
    assert w.dropped_captures == 5


def test_host_overflows_are_counted(worker):
    worker.callback(
        np.zeros((1, 2), dtype=np.float32), 1, None, FakeStatus(input_overflow=True)
    )
    drain(worker.display_queue)
    assert worker.overflows == 1


def test_extra_hardware_channels_are_trimmed_to_the_configured_count():
    """Some inputs will only open at their full channel count.

    Emitting the extra rows would reshape every chunk against the source list
    the pipeline was told to expect.
    """
    w = MicrophoneAcquisitionWorker(
        channels=1, samp_rate=16000, display_queue=queue.Queue()
    )
    w.start()
    w.resume()
    try:
        w.callback(np.array([[0.5, 0.9]], dtype=np.float32), 1, None, FakeStatus())
        chunks = drain(w.display_queue)
        assert chunks[0].shape == (1, 1)
        np.testing.assert_allclose(chunks[0], [[0.5]])
    finally:
        w.stop()
        w.join(timeout=2)


# --------------------------------------------------------------- resolution


def test_sanitize_name_collapses_host_punctuation():
    assert (
        sanitize_name("Microphone (Realtek(R) Audio)", 3) == "Microphone_Realtek_R_Audio"
    )
    assert sanitize_name("", 3) == "Microphone_3"


def _patch_inputs(monkeypatch, devices, openable=None):
    """Enumerate ``devices`` with no host default; ``openable`` indices open."""
    monkeypatch.setattr(
        "bioview_server.device.microphone.utils.list_input_devices",
        lambda logger=None: devices,
    )
    monkeypatch.setattr(
        "bioview_server.device.microphone.utils._default_input_index",
        lambda logger=None: None,
    )
    allowed = {e["index"] for e in devices} if openable is None else set(openable)
    monkeypatch.setattr(
        "bioview_server.device.microphone.utils.supports_input",
        lambda device, channels, rate: device in allowed,
    )


def test_resolve_input_device_matches_key_then_substring(monkeypatch):
    devices = [
        {"index": 4, "name": "Stereo Mix (Realtek)", "max_input_channels": 2},
        {"index": 7, "name": "Headset Microphone (USB)", "max_input_channels": 1},
    ]
    _patch_inputs(monkeypatch, devices)

    assert resolve_input_device("Headset_Microphone_USB") == 7
    assert resolve_input_device("headset") == 7
    assert resolve_input_device(4) == 4
    # No host default: a loopback is never the right guess, so the real input
    # wins even though it enumerates second.
    assert resolve_input_device("default") == 7

    with pytest.raises(RuntimeError, match="no audio input device matches"):
        resolve_input_device("webcam")


def test_an_input_that_will_not_open_is_not_chosen(monkeypatch):
    """A Realtek front-panel jack enumerates whether or not anything is in it.

    An unpopulated one refuses every open with ``Invalid device``. Picking it
    because it came first in enumeration order failed the whole group at
    Connect while a working input sat further down the list.
    """
    devices = [
        {
            "index": 10,
            "name": "FrontMic (Realtek HD Audio Front Mic input)",
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        },
        {
            "index": 12,
            "name": "Headset Microphone (USB)",
            "max_input_channels": 1,
            "default_samplerate": 48000.0,
        },
    ]
    _patch_inputs(monkeypatch, devices, openable={12})

    assert resolve_input_device("default") == 12


def test_an_input_named_by_hand_is_honoured_even_if_it_will_not_open(monkeypatch):
    """The open probe only ranks the automatic choice.

    A named device is the operator's decision, and the error PortAudio raises
    at Connect names it -- second-guessing it here would make a typo and a
    deliberate choice indistinguishable.
    """
    devices = [
        {"index": 10, "name": "FrontMic (Realtek)", "max_input_channels": 2},
        {"index": 12, "name": "Headset Microphone (USB)", "max_input_channels": 1},
    ]
    _patch_inputs(monkeypatch, devices, openable={12})

    assert resolve_input_device("FrontMic") == 10


def test_nothing_usable_says_what_was_rejected_and_why(monkeypatch):
    """The whole point: PortAudio's own message names neither the device nor
    the reason, and this machine hits exactly this path."""
    devices = [
        {
            "index": 9,
            "name": "Stereo Mix (Realtek HD Audio Stereo input)",
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
        },
        {
            "index": 10,
            "name": "FrontMic (Realtek HD Audio Front Mic input)",
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        },
    ]
    # Only the loopback opens -- which is never an automatic choice.
    _patch_inputs(monkeypatch, devices, openable={9})

    with pytest.raises(RuntimeError) as excinfo:
        resolve_input_device("default")

    message = str(excinfo.value)
    assert "FrontMic" in message
    assert "plugged into the jack" in message
    # And it lists what could have been named instead.
    assert "Stereo Mix" in message


def test_a_loopback_is_still_never_chosen_automatically(monkeypatch):
    """Even when it is the only input that opens.

    A loopback records the instruction audio being played back instead of the
    participant, and nothing about the recording says so until it is opened.
    """
    devices = [
        {
            "index": 9,
            "name": "Stereo Mix (Realtek)",
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
        },
    ]
    _patch_inputs(monkeypatch, devices, openable={9})

    with pytest.raises(RuntimeError, match="no usable audio input"):
        resolve_input_device("default")

    # Named by hand, it is honoured.
    assert resolve_input_device("Stereo Mix") == 9


def test_negotiate_samplerate_falls_back_to_a_supported_rate(monkeypatch):
    """MME commonly offers a device only at its native rate.

    Refusing the session outright over a rate the driver merely dislikes would
    be worse than recording at a rate the file then declares.
    """
    supported = {44100.0}
    monkeypatch.setattr(
        "bioview_server.device.microphone.utils.supports_input",
        lambda device, channels, rate: float(rate) in supported,
    )

    class FakeSd:
        @staticmethod
        def query_devices(index):
            return {"default_samplerate": 44100.0}

    monkeypatch.setattr(
        "bioview_server.device.microphone.utils._sounddevice", lambda: FakeSd
    )

    # The device's own native rate is tried before the generic ladder, so a
    # card that only runs at 44.1 kHz is found in one probe rather than four.
    assert negotiate_samplerate(0, 1, 16000) == 44100.0
    assert negotiate_samplerate(0, 1, 44100) == 44100.0


def test_negotiate_samplerate_returns_the_request_when_nothing_works(monkeypatch):
    """So opening the stream fails naming the rate that was asked for."""
    monkeypatch.setattr(
        "bioview_server.device.microphone.utils.supports_input",
        lambda device, channels, rate: False,
    )
    monkeypatch.setattr(
        "bioview_server.device.microphone.utils._sounddevice",
        lambda: (_ for _ in ()).throw(RuntimeError("no portaudio")),
    )
    assert negotiate_samplerate(0, 1, 16000) == 16000.0


# ------------------------------------------------------------------ backend


def _backend(monkeypatch, group_config, negotiated=16000.0):
    monkeypatch.setattr(
        "bioview_server.device.microphone.backend.resolve_input_device",
        lambda device, logger=None: 0,
    )
    monkeypatch.setattr(
        "bioview_server.device.microphone.backend.negotiate_samplerate",
        lambda index, channels, requested, logger=None: negotiated,
    )
    return MicrophoneBackend(
        group_id="MIC", response_queue=mp.Queue(), group_config=group_config
    )


def test_sources_carry_the_negotiated_rate_not_the_requested_one(monkeypatch):
    """``disp_freq`` is the timebase written into the recording header.

    Advertising the rate that was asked for rather than the one the input
    actually runs at makes every recorded timestamp wrong by that ratio.
    """
    backend = _backend(
        monkeypatch,
        {"type": "MICROPHONE", "samp_rate": 16000, "channels": 1},
        negotiated=44100.0,
    )
    assert backend.samp_rate == 44100
    (source,) = backend.data_sources
    assert source.get_disp_freq() == 44100.0
    assert source.label == "Audio"


def test_multiple_channels_get_numbered_labels(monkeypatch):
    backend = _backend(
        monkeypatch, {"type": "MICROPHONE", "samp_rate": 16000, "channels": 2}
    )
    labels = sorted(s.label for s in backend.data_sources)
    assert labels == ["Audio1", "Audio2"]


def test_configured_labels_win(monkeypatch):
    backend = _backend(
        monkeypatch,
        {
            "type": "MICROPHONE",
            "samp_rate": 16000,
            "channels": 2,
            "labels": ["Speech", "Room"],
        },
    )
    assert sorted(s.label for s in backend.data_sources) == ["Room", "Speech"]


def test_a_biopac_shaped_channel_mask_still_loads(monkeypatch):
    """``channels`` is a count here and a mask for BIOPAC.

    A config copied from a BIOPAC block should open one input per set entry
    rather than failing on ``int([1, 1, 0, 0])``.
    """
    backend = _backend(
        monkeypatch, {"type": "MICROPHONE", "samp_rate": 16000, "channels": [1, 1, 0, 0]}
    )
    assert backend.channel_count == 2


def test_chunk_size_is_capped_at_a_tenth_of_a_second(monkeypatch):
    """The chunk is the acquisition latency, so a huge blocksize delays the plot."""
    backend = _backend(
        monkeypatch,
        {"type": "MICROPHONE", "samp_rate": 16000, "channels": 1, "blocksize": 100000},
        negotiated=16000.0,
    )
    assert backend._frames_per_read() == 1600
