# Contributing

Install the package and its test dependency, then run the standalone checks:

```bash
python -m pip install -e . pytest
python -m pytest -q
python -m pip wheel --no-deps .
```

Backend tests use injected command runners and must not access a live scheduler.
Keep project-specific configuration, launch commands, metrics, and assets in a
host-owned `ProjectAdapter`; use `examples/minimal_project_adapter.py` as the
contract checklist.
