# Project Lifecycle

`ml-expd` maintains a workspace-local registry of research Projects. The
registry controls which Projects this daemon indexes and polls. It does not
replace a Project's authored `research_project.yaml`, and it never owns the
Project repository, scheduler jobs, Runs, Attempts, logs, checkpoints, or
artifacts.

## Identity and ownership

Each registry record binds a stable Project identity to an absolute
`research_project.yaml` path and records its lifecycle state, registration
source, timestamps, and operator reason. The authored file remains the source
of Project title, run roots, controller configuration, Campaign catalog, and
research-question locations.

One daemon workspace has one registry. Two daemon workspaces may register the
same Project independently and assign different lifecycle states. Operators
must avoid enabling live collectors in two workspaces against the same
Project unless the controller explicitly supports that arrangement.

## State machine

```text
                  pause                   archive
        ACTIVE ------------> PAUSED ----------------> ARCHIVED
           ^                   |  ^                      |
           |      resume       |  | restore              |
           +-------------------+  +----------------------+
                    archive from ACTIVE is also allowed
```

| State | Indexed as active | Live collector polls | Meaning |
|---|---:|---:|---|
| `ACTIVE` | yes | yes | Normal observation and operation eligibility |
| `PAUSED` | no | no | Temporarily removed from this live workspace |
| `ARCHIVED` | no | no | Retained as workspace history; restore first returns it to `PAUSED` |

Valid transitions are:

- `ACTIVE -> PAUSED` or `ACTIVE -> ARCHIVED`;
- `PAUSED -> ACTIVE` or `PAUSED -> ARCHIVED`;
- `ARCHIVED -> PAUSED`.

Archive requires an operator reason. Restore is deliberately two-step:
`ARCHIVED -> PAUSED -> ACTIVE`, so restoring a record cannot silently resume
backend polling.

## Operations

The HTTP API is the canonical transport boundary:

```text
GET  /api/project-lifecycle
POST /api/project-lifecycle/register
POST /api/project-lifecycle/{project}/pause
POST /api/project-lifecycle/{project}/resume
POST /api/project-lifecycle/{project}/archive
POST /api/project-lifecycle/{project}/restore
POST /api/project-lifecycle/{project}/unregister
POST /api/project-lifecycle/unregister-all
```

Registration validates and loads the authored Project, persists the registry
record, adds it to the live runtime, and indexes its existing Run roots.
Pause, archive, and unregister remove the Project from the runtime's shared
active list, so the collector observes the change between calls without a
daemon restart. Resume reloads the authored catalog before returning the
Project to that list; identity drift fails closed.

Unregister removes only the workspace registry record. Archive keeps the
record but makes it inactive. Neither operation deletes or modifies:

- the Project repository or `research_project.yaml`;
- Campaign, Run, Attempt, or evidence files;
- scheduler jobs or backend storage;
- logs, metrics, checkpoints, or artifacts.

Project-file writes and scheduler mutations remain governed by the separate
Action runtime gates. A lifecycle transition does not authorize either.

## Bootstrap and durability

`ServerConfig.projects` seeds an empty registry for compatibility and initial
deployment. After bootstrap, the workspace-owned registry is durable under
`project_registry_root`; normal registration and transitions use the API.
The active in-memory Project list is reconstructed from that registry at
daemon startup.

Registry writes are atomic and lifecycle events are retained for audit. A
registered file whose declared Project identity changes is rejected as
identity drift instead of being silently rebound.

## Client contract

Research Console and other clients render lifecycle state and send explicit
commands, but they do not maintain a second registry or collector. Clients
must not interpret an absent Project from the active list as deleted: consult
`GET /api/project-lifecycle` to distinguish `PAUSED`, `ARCHIVED`, unregistered,
and unknown Projects.
