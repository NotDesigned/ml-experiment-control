"""One stdlib HTTPS transport; a fixed TCP relay preserves origin TLS and auth."""
from __future__ import annotations

import http.client
import ipaddress
import os
import socket
from urllib.parse import urlsplit

CONTRACT = "api-tcp-relay.v1"


class ApiTransportError(ValueError):
    """Fixed diagnostic codes; no endpoints, credentials or response bodies."""


def error_code(error):
    allowed = {"API_RELAY_CONFIG_INVALID", "API_RELAY_TARGET_MISMATCH", "API_RELAY_UNREACHABLE"}
    return str(error) if isinstance(error, ApiTransportError) and str(error) in allowed else type(error).__name__


def relay_environment(value):
    if value is None:
        return {}
    try:
        if not isinstance(value, dict) or set(value) != {"endpoint", "origin"}:
            raise ValueError()
        relay, origin = urlsplit(value["endpoint"]), urlsplit(value["origin"])
        address = ipaddress.ip_address(relay.hostname)
        if (relay.scheme != "tcp" or not address.is_private or address.is_unspecified or address.is_multicast
                or relay.port is None or origin.scheme != "https" or not origin.hostname
                or any(u.username or u.password or u.path or u.query or u.fragment for u in (relay, origin))):
            raise ValueError()
        # Accessing .port also validates the upper bound; zero is invalid.
        if not relay.port or origin.port == 0:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ApiTransportError("API_RELAY_CONFIG_INVALID") from None
    return {"ML_EXPD_API_RELAY": value["endpoint"], "ML_EXPD_API_ORIGIN": value["origin"]}


def https_connection(target, *, timeout, environment=None):
    env = os.environ if environment is None else environment
    endpoint, origin = env.get("ML_EXPD_API_RELAY"), env.get("ML_EXPD_API_ORIGIN")
    connection = http.client.HTTPSConnection(target.hostname, target.port or 443, timeout=timeout)
    if endpoint is None and origin is None:
        return connection
    relay_environment({"endpoint": endpoint, "origin": origin})
    allowed = urlsplit(origin)
    if (target.hostname, target.port or 443) != (allowed.hostname, allowed.port or 443):
        raise ApiTransportError("API_RELAY_TARGET_MISMATCH")
    relay = urlsplit(endpoint)
    def connect(_address, seconds, source_address):
        try:
            return socket.create_connection((relay.hostname, relay.port), seconds, source_address)
        except OSError:
            raise ApiTransportError("API_RELAY_UNREACHABLE") from None
    # HTTPConnection's socket factory changes only the TCP destination. The
    # standard HTTPSConnection still wraps it with origin SNI/certificate checks.
    connection._create_connection = connect
    return connection
