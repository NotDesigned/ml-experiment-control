"""Inspect the optional SDK without importing it into the API process."""
from __future__ import annotations

import importlib
import importlib.util
import sys


SDK_MISSING = "WANDB_SDK_NOT_INSTALLED"


class WandbUnavailable(ImportError):
    """The optional publisher cannot load its SDK; local records remain usable."""


def sdk_status():
    available = (sys.modules["wandb"] is not None if "wandb" in sys.modules
                 else importlib.util.find_spec("wandb") is not None)
    return {"available": available, "error": None if available else SDK_MISSING}


def load_sdk():
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise WandbUnavailable(SDK_MISSING) from exc
