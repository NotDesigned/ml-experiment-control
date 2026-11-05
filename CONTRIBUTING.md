# Contributing

Follow [development setup and validation](docs/development.md). Start with
affected tests. Tests must not contact live schedulers.

Keep the [crate boundaries](docs/architecture.md): pure core, confined I/O,
standalone client, server control plane and in-image worker. Scientific commands,
metrics and success criteria are project-owned. Follow the
[retained Python contract](docs/downstream_contract.md) when changing its exports.

Maintain [current documentation](docs/README.md) alongside behavior changes.
Use one primary page per concept; remove obsolete instructions and label known
limitations. Keep credentials and deployment evidence outside Git.
