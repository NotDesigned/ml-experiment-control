# Contributing

Install the package and development dependencies, then run the repository gates:

```bash
uv sync --locked
uv run python tools/coverage_gate.py
uv run python tools/generate_cli_reference.py --check
uv run python -m compileall -q src tests tools examples
uv build
```

Add runtime dependencies with `uv add <package>` and development dependencies
with `uv add --dev <package>`. Commit the resulting `pyproject.toml` and
`uv.lock` changes together.

The coverage gate checks repository-wide line and branch coverage independently:
at least 90% line coverage and 80% branch coverage. See
[`docs/development.md`](docs/development.md) for the testing and generated CLI
documentation policy.

Backend tests use injected command runners and must not access a live scheduler.
Keep project-specific configuration, launch commands, metrics, and assets in a
host-owned `ProjectAdapter`; use `examples/minimal_project_adapter.py` as the
contract checklist.
