BEGIN TRANSACTION;
CREATE TABLE events (
    seq INTEGER PRIMARY KEY,
    action_id TEXT NOT NULL REFERENCES executions(action_id),
    event_key TEXT NOT NULL,
    revision INTEGER,
    payload TEXT NOT NULL CHECK (json_valid(payload) AND json_type(payload)='object'),
    UNIQUE (action_id, event_key),
    UNIQUE (action_id, revision)
);
INSERT INTO "events" VALUES(1,'action-0123456789abcdef','transition:00000000000000000000000000000001',1,'{"event": "action_prepared", "payload": {"operation": "WRITE_CAMPAIGN", "ready": true}, "revision": 1, "timestamp": "2026-10-27T00:00:00Z", "transition_id": "00000000000000000000000000000001"}');
INSERT INTO "events" VALUES(2,'action-0123456789abcdef','transition:00000000000000000000000000000002',2,'{"event": "execution_authorized", "payload": {"error": null, "status": "AUTHORIZED"}, "revision": 2, "timestamp": "2026-10-27T00:00:00Z", "transition_id": "00000000000000000000000000000002"}');
INSERT INTO "events" VALUES(3,'action-0123456789abcdef','transition:00000000000000000000000000000003',3,'{"event": "execution_started", "payload": {"error": null, "intent_digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "status": "EXECUTING"}, "revision": 3, "timestamp": "2026-10-27T00:00:00Z", "transition_id": "00000000000000000000000000000003"}');
CREATE TABLE executions (
    action_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    payload TEXT NOT NULL CHECK (json_valid(payload) AND json_type(payload)='object')
);
INSERT INTO "executions" VALUES('action-0123456789abcdef',3,'{"authorization_actor": "actor:test", "authorization_expires_at": "2100-01-01T00:00:00Z", "authorization_note": "approved explicitly", "authorized_at": "2026-10-27T00:00:00Z", "authorized_gate_bundle_digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "authorized_intent_digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "error": null, "finished_at": null, "next_action": "WAIT", "resolution": "STILL_IN_PROGRESS", "result": null, "revision": 3, "safe_to_retry": false, "started_at": "2026-10-27T00:00:00Z", "status": "EXECUTING"}');
CREATE INDEX events_action ON events(action_id, seq);
COMMIT;
PRAGMA user_version=1;
