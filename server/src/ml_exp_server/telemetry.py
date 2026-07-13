"""Privacy-preserving OpenTelemetry support for ml-expd.

Telemetry is deliberately optional.  Importing this module does not import the
OpenTelemetry SDK and the disabled path has no global or background-thread side
effects.  Callers must use the fixed span names and attribute allowlist below;
research content is never accepted as telemetry metadata.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import math
import re
from typing import Any, Iterator, Mapping, Protocol

from . import __version__


INSTRUMENTATION_SCOPE = "ml_exp_server"
DEFAULT_OTLP_HTTP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
# Export remains asynchronous.  This short deadline only bounds background
# delivery and process shutdown when the default local collector is absent.
OTLP_EXPORT_TIMEOUT_SECONDS = 1.0
BATCH_EXPORT_TIMEOUT_MILLIS = 1_000

# Keeping names closed prevents a prompt, object title, or command line from
# accidentally becoming a span name.
SAFE_SPAN_NAMES = frozenset(
    {
        "research.operation",
        "research.intent.prepare",
        "research.action.prepare",
        "research.action.authorize",
        "research.action.execute",
        "research.action.reconcile",
        "research.controller",
    }
)

_IDENTITY_ATTRIBUTES = frozenset(
    {
        "research.project",
        "research.scope_type",
        "research.object_id",
        "research.operation_id",
        "research.intent_id",
        "research.action_id",
        "research.run_id",
        "research.attempt_id",
        "backend.job_id",
        "research.status",
        "error.type",
        "error.category",
    }
)
_NUMBER_ATTRIBUTES = frozenset({"research.retry_count", "research.duration_ms"})
SAFE_ATTRIBUTE_KEYS = _IDENTITY_ATTRIBUTES | _NUMBER_ATTRIBUTES

# IDs, enum values, and class/category names only.  Free-form text is rejected,
# even when supplied under an otherwise allowed key.
_SAFE_TOKEN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}\Z")


class TelemetryInitializationError(RuntimeError):
    """Raised when explicitly enabled telemetry cannot be initialized."""


@dataclass(frozen=True)
class TelemetrySettings:
    """Minimal settings accepted by :func:`initialize_telemetry`.

    Application config models may be passed directly too; initialization reads
    these three attributes (or mapping keys) without importing a schema module.
    """

    enabled: bool = True
    otlp_http_endpoint: str | None = DEFAULT_OTLP_HTTP_ENDPOINT
    service_name: str = "ml-expd"


class _Span(Protocol):
    def set_attribute(self, key: str, value: Any) -> Any: ...


class _Tracer(Protocol):
    def start_as_current_span(self, name: str, **kwargs: Any) -> Any: ...


class _NoOpSpan:
    """A tiny span-shaped object so disabled call sites need no branches."""

    def set_attribute(self, key: str, value: Any) -> None:
        del key, value


class _SafeSpan:
    """Restrict post-creation attributes as strictly as initial attributes."""

    def __init__(self, span: _Span):
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        clean = safe_attributes({key: value})
        if key in clean:
            self._span.set_attribute(key, clean[key])


def _setting(settings: object, name: str, default: Any) -> Any:
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


def safe_attributes(attributes: Mapping[str, object] | None) -> dict[str, object]:
    """Return only allowlisted, non-content span attributes.

    Unknown keys are dropped.  String values must be identifier-like tokens;
    multiline/free-form values are never exported.  Booleans are not accepted
    as numbers because they commonly result from accidental content flags.
    """

    clean: dict[str, object] = {}
    for key, value in (attributes or {}).items():
        if key in _IDENTITY_ATTRIBUTES:
            if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value):
                clean[key] = value
        elif key in _NUMBER_ATTRIBUTES:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                clean[key] = value
            elif (
                key == "research.duration_ms"
                and isinstance(value, float)
                and math.isfinite(value)
                and value >= 0
            ):
                clean[key] = value
    return clean


@dataclass
class Telemetry:
    """Configured tracer facade with a safe, renderer-neutral span API."""

    enabled: bool = False
    tracer: _Tracer | None = None
    provider: object | None = None
    _owns_provider: bool = False

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[_Span]:
        """Create a safe span, recording only an exception's class name.

        The OpenTelemetry defaults that record exception messages and stack
        traces are explicitly disabled because either may contain research
        evidence, prompts, command output, or credentials.
        """

        if name not in SAFE_SPAN_NAMES:
            raise ValueError(f"unsupported telemetry span name: {name!r}")
        if not self.enabled or self.tracer is None:
            yield _NoOpSpan()
            return

        with self.tracer.start_as_current_span(
            name,
            attributes=safe_attributes(attributes),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            safe_span = _SafeSpan(span)
            try:
                yield safe_span
            except BaseException as exc:
                safe_span.set_attribute("error.type", type(exc).__name__)
                raise

    def shutdown(self) -> None:
        """Flush owned SDK resources; injected providers remain caller-owned."""

        if self._owns_provider and self.provider is not None:
            shutdown = getattr(self.provider, "shutdown", None)
            if callable(shutdown):
                shutdown()


def initialize_telemetry(
    settings: object | None = None,
    *,
    tracer_provider: object | None = None,
    exporter: object | None = None,
    span_processor_factory: Any | None = None,
) -> Telemetry:
    """Initialize optional OTLP/HTTP tracing without changing global providers.

    A preconfigured provider is the intended injection seam for tests and
    embedding applications.  Otherwise the SDK provider, batch processor, and
    OTLP/HTTP exporter are constructed lazily.  Explicit enablement fails with
    a useful error when the optional packages are absent.
    """

    settings = settings or TelemetrySettings()
    if not bool(_setting(settings, "enabled", False)):
        return Telemetry()

    service_name = _setting(settings, "service_name", "ml-expd")
    if not isinstance(service_name, str) or not _SAFE_TOKEN.fullmatch(service_name):
        raise TelemetryInitializationError("telemetry service_name must be an identifier")

    owns_provider = tracer_provider is None
    if tracer_provider is None:
        endpoint = _setting(settings, "otlp_http_endpoint", DEFAULT_OTLP_HTTP_ENDPOINT)
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise TelemetryInitializationError(
                "telemetry otlp_http_endpoint is required when telemetry is enabled"
            )
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise TelemetryInitializationError(
                "telemetry is enabled but OpenTelemetry SDK and OTLP HTTP exporter "
                "are not installed"
            ) from exc

        tracer_provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        # Keep exporter availability observable through daemon health/logging
        # rather than unsolicited root-handler output.
        exporter_logger = logging.getLogger(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        )
        if not exporter_logger.handlers:
            exporter_logger.addHandler(logging.NullHandler())
        exporter_logger.propagate = False
        exporter = exporter or OTLPSpanExporter(
            endpoint=endpoint,
            timeout=OTLP_EXPORT_TIMEOUT_SECONDS,
        )
        if span_processor_factory is None:
            processor = BatchSpanProcessor(
                exporter,
                export_timeout_millis=BATCH_EXPORT_TIMEOUT_MILLIS,
            )
        else:
            # Preserve the simple one-argument injection contract for tests and
            # embedding applications; timeout policy belongs to the SDK path.
            processor = span_processor_factory(exporter)
        tracer_provider.add_span_processor(processor)
    elif exporter is not None:
        try:
            if span_processor_factory is None:
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                span_processor_factory = BatchSpanProcessor
            tracer_provider.add_span_processor(span_processor_factory(exporter))
        except (AttributeError, ImportError) as exc:
            raise TelemetryInitializationError(
                "injected tracer provider cannot accept the configured exporter"
            ) from exc

    get_tracer = getattr(tracer_provider, "get_tracer", None)
    if not callable(get_tracer):
        raise TelemetryInitializationError("tracer provider does not provide get_tracer()")
    tracer = get_tracer(INSTRUMENTATION_SCOPE, __version__)
    return Telemetry(
        enabled=True,
        tracer=tracer,
        provider=tracer_provider,
        _owns_provider=owns_provider,
    )


__all__ = [
    "BATCH_EXPORT_TIMEOUT_MILLIS",
    "DEFAULT_OTLP_HTTP_ENDPOINT",
    "INSTRUMENTATION_SCOPE",
    "OTLP_EXPORT_TIMEOUT_SECONDS",
    "SAFE_ATTRIBUTE_KEYS",
    "SAFE_SPAN_NAMES",
    "Telemetry",
    "TelemetryInitializationError",
    "TelemetrySettings",
    "initialize_telemetry",
    "safe_attributes",
]
