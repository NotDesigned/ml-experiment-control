"""The execution package imports without the control plane or cloud SDKs."""
from pathlib import Path
import subprocess
import sys


def test_execution_programs_import_with_only_the_standard_library(tmp_path):
    script = """
import importlib, importlib.abc, pkgutil, sys
sys.path.insert(0, sys.argv[1])
class RejectControlPlane(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'ml_exp_server', 'experiment_control', 'ml_exp_client', 'fastapi',
            'pydantic', 'yaml', 'boto3', 'botocore', 'wandb', 'opentelemetry',
            'alibabacloud_tea_openapi', 'alibabacloud_oss_v2',
        }:
            raise ImportError('control-plane dependency: ' + fullname)
sys.meta_path.insert(0, RejectControlPlane())
import ml_exp_worker
for module in pkgutil.iter_modules(ml_exp_worker.__path__):
    importlib.import_module('ml_exp_worker.' + module.name)
    assert ml_exp_worker.source_path(module.name + '.py').is_file()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(__file__).parents[1] / "src")],
        cwd=tmp_path, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
