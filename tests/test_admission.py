"""Who gets in, and what happens to whoever does not."""

import pytest
from bioview_common import get_local_addresses

from bioview_server.admission import (
    AdmissionController,
    AdmissionError,
    normalize_policy,
    resolve_hostnames,
)


CAMPUS_PUBLIC = "128.253.4.17"


@pytest.fixture
def asked():
    """An ``ask`` controller whose prompt is a recorded, scripted answer."""

    class Recorder(AdmissionController):
        def __init__(self, answer=True, **kwargs):
            super().__init__(policy="ask", **kwargs)
            self.answer = answer
            self.asked = []

        def _ask(self, peer_ip, client_info):
            self.asked.append(peer_ip)
            return self.answer

    return Recorder


@pytest.mark.parametrize("policy", ["loopback", "local", "ask", "any"])
def test_this_machine_is_never_asked_about(policy):
    """A window and the server it started are the same machine, always."""
    controller = AdmissionController(policy=policy)
    for address in ["127.0.0.1", *get_local_addresses()]:
        allowed, reason = controller.decide(address, {})
        assert allowed, f"{policy} refused {address}: {reason}"


def test_loopback_refuses_the_lan():
    controller = AdmissionController(policy="loopback")
    allowed, reason = controller.decide("192.168.1.50", {})
    assert not allowed
    assert "own machine" in reason


def test_local_admits_a_private_address_without_asking():
    controller = AdmissionController(policy="local")
    allowed, _ = controller.decide("10.0.0.9", {})
    assert allowed


def test_local_is_exactly_what_broke_multi_pc_on_a_public_network():
    """The regression the prompt exists to replace, kept as documentation."""
    controller = AdmissionController(policy="local")
    allowed, reason = controller.decide(CAMPUS_PUBLIC, {"hostname": "lab-pc-2"})
    assert not allowed
    assert "--allow local" in reason


def test_any_admits_everyone(asked):
    controller = AdmissionController(policy="any")
    allowed, _ = controller.decide(CAMPUS_PUBLIC, {})
    assert allowed


def test_an_unknown_policy_is_refused_at_startup_not_at_connect_time():
    with pytest.raises(AdmissionError):
        AdmissionController(policy="sometimes")
    assert normalize_policy("ASK ") == "ask"


def test_ask_admits_a_public_peer_the_operator_says_yes_to(asked):
    controller = asked(answer=True)
    allowed, reason = controller.decide(CAMPUS_PUBLIC, {"hostname": "lab-pc-2"})
    assert allowed
    assert controller.asked == [CAMPUS_PUBLIC]
    assert "operator" in reason


def test_ask_refuses_a_peer_the_operator_says_no_to(asked):
    controller = asked(answer=False)
    allowed, reason = controller.decide(CAMPUS_PUBLIC, {})
    assert not allowed
    assert "refused" in reason


def test_an_answer_is_remembered_so_reconnecting_does_not_re_ask(asked):
    """A second window on the same machine must not raise a second dialog."""
    controller = asked(answer=True)
    controller.decide(CAMPUS_PUBLIC, {})
    controller.decide(CAMPUS_PUBLIC, {})
    controller.decide(CAMPUS_PUBLIC, {})
    assert controller.asked == [CAMPUS_PUBLIC]


def test_a_refusal_is_remembered_too(asked):
    controller = asked(answer=False)
    assert controller.decide(CAMPUS_PUBLIC, {})[0] is False
    assert controller.decide(CAMPUS_PUBLIC, {})[0] is False
    assert controller.asked == [CAMPUS_PUBLIC]


def test_forget_makes_the_next_connection_ask_again(asked):
    controller = asked(answer=True)
    controller.decide(CAMPUS_PUBLIC, {})
    controller.forget(CAMPUS_PUBLIC)
    controller.decide(CAMPUS_PUBLIC, {})
    assert controller.asked == [CAMPUS_PUBLIC, CAMPUS_PUBLIC]


def test_a_trusted_address_is_never_asked_about(asked):
    controller = asked(answer=False, trusted=[CAMPUS_PUBLIC])
    allowed, _ = controller.decide(CAMPUS_PUBLIC, {})
    assert allowed
    assert controller.asked == [], "--trust still raised a prompt"


def test_concurrent_connections_from_one_peer_raise_one_prompt(asked):
    """Two windows opening at once must not stack two dialogs."""
    import threading

    controller = asked(answer=True)
    barrier = threading.Barrier(4)
    results = []

    def connect():
        barrier.wait()
        results.append(controller.decide(CAMPUS_PUBLIC, {})[0])

    threads = [threading.Thread(target=connect) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert results == [True] * 4
    assert controller.asked == [CAMPUS_PUBLIC]


def test_an_empty_peer_address_is_refused_rather_than_asked_about(asked):
    controller = asked(answer=True)
    allowed, _ = controller.decide("", {})
    assert not allowed
    assert controller.asked == []


def test_no_way_to_ask_means_refuse(monkeypatch):
    """A headless server with no console admits nobody it has not been told to."""
    controller = AdmissionController(policy="ask")
    monkeypatch.setattr(controller, "_ask_desktop", lambda who, ip: None)
    monkeypatch.setattr(controller, "_ask_console", lambda who, ip: None)
    allowed, _ = controller.decide(CAMPUS_PUBLIC, {})
    assert not allowed


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("loopback", False), ("local", False), ("ask", True), ("any", True)],
)
def test_discovery_probes_follow_the_policy_but_never_prompt(policy, expected):
    """Under ask, a probe is answered unasked: it is how a rig is found at all."""
    controller = AdmissionController(policy=policy)
    assert controller.may_probe(CAMPUS_PUBLIC) is expected
    assert controller.may_probe("127.0.0.1") is True


def test_local_answers_probes_from_private_addresses():
    controller = AdmissionController(policy="local")
    assert controller.may_probe("192.168.1.50") is True


def test_resolve_hostnames_keeps_literal_addresses():
    assert resolve_hostnames(["10.0.0.1", " ", "10.0.0.2"]) == {"10.0.0.1", "10.0.0.2"}


def test_resolve_hostnames_resolves_localhost_to_its_addresses():
    resolved = resolve_hostnames(["localhost"])
    assert "localhost" in resolved
    assert "127.0.0.1" in resolved
