# Contributing

Follow [development setup and validation](docs/development.md). Start with
affected tests; required CI includes independent full line/branch gates and
installed-wheel checks. Tests must not contact live schedulers.

Keep the [package boundaries](docs/architecture.md): standalone HTTP client,
server control plane and reusable backend core. Scientific commands, metrics
and success criteria are project-owned. Follow the
[public integration contract](docs/downstream_contract.md) when changing exports.

Maintain [current documentation](docs/README.md) alongside behavior changes.
Use one primary page per concept; remove obsolete instructions and label known
limitations. Keep credentials and deployment evidence outside Git. Generate the
redactor CLI reference with its tool rather than editing it by hand.
