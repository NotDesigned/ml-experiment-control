"""Per-request state shared by imported builders and the module entry point."""
from contextvars import ContextVar

BUILD_LOG = ContextVar("ml_exp_build_log", default=None)
BUILD_PROGRESS = ContextVar("ml_exp_build_progress", default=None)
BUILD_CACHE = ContextVar("ml_exp_build_cache", default=None)
BUILD_REMOTE_CONTEXT = ContextVar("ml_exp_remote_context", default=None)
BUILD_CONTEXT_BYTES = ContextVar("ml_exp_remote_context_bytes", default=0)
