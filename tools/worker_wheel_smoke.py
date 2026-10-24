"""Verify the installed execution wheel from outside the repository."""
import importlib
import importlib.metadata
import importlib.util
import pkgutil


def main():
    assert not importlib.metadata.requires("ml-experiment-worker")
    for name in ("ml_exp_server", "ml_exp_client", "experiment_control", "yaml", "fastapi", "wandb"):
        assert importlib.util.find_spec(name) is None, f"unexpected dependency: {name}"
    import ml_exp_worker
    modules = list(pkgutil.iter_modules(ml_exp_worker.__path__))
    assert len(modules) == 13
    for module in modules:
        importlib.import_module("ml_exp_worker." + module.name)
        assert ml_exp_worker.source_path(module.name + ".py").is_file()
    print("Standalone worker wheel: 13 execution programs; no runtime dependencies")


if __name__ == "__main__":
    main()
