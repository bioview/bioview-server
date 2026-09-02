"""A --local server must accept a client from its own machine.

The client dials the address the server advertised in its discovery info, which
is a NIC address rather than loopback. On a network that hands out public
addresses (many campus and corporate networks do) that address is in no private
range, and a locality check based on private ranges alone closes the connection
before the challenge is sent -- the client then reports "Server did not provide
authentication token".
"""
import pytest
from bioview_common import get_local_addresses

from bioview_server.server import Server


@pytest.fixture
def server():
    return Server(local_only=True, control_port=0, data_port=0)


@pytest.mark.parametrize("peer", ["127.0.0.1", "192.168.1.10", "10.0.0.4"])
def test_loopback_and_private_peers_are_local(server, peer):
    assert server._is_local_client((peer, 51000))


def test_this_machines_own_addresses_are_local_whatever_their_range(server):
    for addr in get_local_addresses():
        assert server._is_local_client((addr, 51000)), addr


def test_a_public_address_that_is_not_ours_is_still_rejected(server):
    outside = "8.8.8.8"
    assert outside not in get_local_addresses()
    assert not server._is_local_client((outside, 51000))


@pytest.mark.parametrize("address", [None, "", (), "127.0.0.1", 42])
def test_malformed_peer_addresses_are_rejected_rather_than_raising(server, address):
    assert not server._is_local_client(address)
