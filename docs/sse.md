# Index event stream

`GET /api/stream` uses `sse-starlette` for SSE framing, a 15-second heartbeat,
disconnect detection, and a 15-second send timeout. Authentication and protocol
headers are unchanged. Ordinary messages keep their existing JSON envelope:

```text
data: {"type": "index_updated", "project": "demo", "run_id": "run-a"}

```

The stream contains best-effort invalidations, not a durable event history.
Fetch an HTTP snapshot on initial connection and every reconnect. There are no
replay IDs and `Last-Event-ID` does not recover missed invalidations.

Both the worker-thread handoff and each subscriber queue hold at most 128
updates. If either fills, affected subscribers receive this ordinary SSE data
message and their stream closes:

```text
data: {"type": "resync_required"}

```

Reconnect and reload the authoritative HTTP snapshot on `resync_required`.
Clients should tolerate unknown event types and reconnects. A blocked socket
may time out before receiving the resync message, so resync on reconnect is
required too. A slow subscriber never blocks delivery to other subscribers.
Generators close explicitly after disconnects, timeouts, and shutdown.

The implementation uses the [sse-starlette response API](https://github.com/sysid/sse-starlette).
