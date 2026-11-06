"""Per-request state shared by imported builders and the module entry point."""
from contextvars import ContextVar

BUILD_LOG = ContextVar("ml_exp_build_log", default=None)
BUILD_PROGRESS = ContextVar("ml_exp_build_progress", default=None)
BUILD_CACHE = ContextVar("ml_exp_build_cache", default=None)
