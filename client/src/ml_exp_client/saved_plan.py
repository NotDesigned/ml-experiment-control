"""Small durable save/observe/request helpers; protocols retain their own policy."""
from __future__ import annotations

from .api import ClientError


def bound_response(value, expected, *, message="response differs from the requested exact identity"):
    """Require explicit response identity before saving it or dispatching effects."""
    if not isinstance(value, dict) or any(value.get(key) != item for key, item in expected.items()):
        raise ClientError(message)
    return value


class SavedPlan:
    """Save replies and write effect markers before a single HTTP dispatch.

    No retries, status transitions, authorization decisions or reconciliation
    live here. Each caller retains its existing identities and validators.
    """

    def __init__(self, client, state, path, *, persist):
        self.client, self.state, self.path, self.persist = client, state, path, persist

    def store(self, key, value, *, validate=None):
        value = validate(value) if validate is not None else value
        self.state[key] = value
        self.persist(self.path, self.state)
        return value

    def observe(self, key, endpoint, *, validate=None):
        return self.store(key, self.client.call(endpoint), validate=validate)

    def request(self, key, endpoint, *, data, marker=None, validate=None,
                uncertain="request outcome is uncertain; inspect the original identity before continuing"):
        if marker is not None:
            if self.state.get(marker):
                raise ClientError(uncertain)
            self.state[marker] = True
            self.persist(self.path, self.state)
        return self.store(key, self.client.call(endpoint, data=data), validate=validate)
