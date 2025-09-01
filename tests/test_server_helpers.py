import hashlib
import time

import pytest
from bioview_common import AuthenticationError, Command, Response

from bioview_server.server import Server


def test_compute_response_matches_hashlib():
    s = Server()
    challenge = "abc123"
    token = "secrettoken"
    expected = hashlib.sha256(f"{challenge}:{token}".encode()).hexdigest()
    assert s._compute_response(challenge, token) == expected


def test_auth_validate_timestamp_pass():
    s = Server()
    payload = {"timestamp": time.time()}
    # Should not raise
    s._auth_validate_timestamp(payload)


def test_auth_validate_timestamp_fail():
    s = Server()
    payload = {"timestamp": 0}
    with pytest.raises(AuthenticationError):
        s._auth_validate_timestamp(payload)


def test_auth_validate_command_type_pass():
    s = Server()
    parsed = {"type": Command.AUTHENTICATE_CLIENT.value}
    # Should not raise
    s._auth_validate_command_type(parsed)


def test_auth_validate_command_type_fail():
    s = Server()
    parsed = {"type": "SOMETHING_ELSE"}
    with pytest.raises(AuthenticationError):
        s._auth_validate_command_type(parsed)


def test_dispatch_unknown_command_returns_error():
    s = Server()
    resp = s._dispatch_command("NON_EXISTENT", {})
    assert resp["type"] == Response.ERROR.value
    assert "Unknown command" in resp["payload"]["message"]


def test_sanitize_for_json_converts_nonserializable():
    s = Server()

    class Foo:
        def __repr__(self):
            return "<foo>"

    data = {"a": Foo(), "b": [1, 2, Foo()], "c": (3, Foo())}
    sanitized = s._sanitize_for_json(data)
    # Ensure sanitized contains strings for non-serializable items
    assert isinstance(sanitized["a"], str)
    assert isinstance(sanitized["b"][2], str)
    assert isinstance(sanitized["c"][1], str)
