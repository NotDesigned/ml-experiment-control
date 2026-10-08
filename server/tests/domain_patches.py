"""Patch shared dependencies where domain operations import them."""
from ml_exp_server import application
from ml_exp_server.runs import context, evidence, queries, validation, run_operations, failures


from ml_exp_server.projects import lifecycle


def patch_application_dependency(monkeypatch, name, value):
    for module in (application, context, evidence, queries, validation, run_operations, lifecycle, failures):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
