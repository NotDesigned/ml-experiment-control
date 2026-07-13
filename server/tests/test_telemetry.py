from __future__ import annotations

from contextlib import contextmanager
import builtins
import logging
import socket
import sys
import time
from types import ModuleType

import pytest

from ml_exp_server.telemetry import (
    BATCH_EXPORT_TIMEOUT_MILLIS,
    DEFAULT_OTLP_HTTP_ENDPOINT,
    OTLP_EXPORT_TIMEOUT_SECONDS,
    TelemetryInitializationError,
    TelemetrySettings,
    initialize_telemetry,
    safe_attributes,
)


class FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class FakeTracer:
    def __init__(self):
        self.started = []

    @contextmanager
    def start_as_current_span(self, name, **kwargs):
        span = FakeSpan()
        self.started.append((name, kwargs, span))
        yield span


class FakeProvider:
    def __init__(self):
        self.tracer = FakeTracer()
        self.requests = []
        self.shutdown_calls = 0

    def get_tracer(self, name, version):
        self.requests.append((name, version))
        return self.tracer

    def shutdown(self):
        self.shutdown_calls += 1


def test_disabled_has_zero_import_and_provider_side_effects(monkeypatch):
    attempted = []
    original_import = builtins.__import__

    def observe(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            attempted.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observe)
    telemetry = initialize_telemetry(TelemetrySettings(enabled=False))

    assert telemetry.enabled is False
    with telemetry.span("research.operation", {"prompt": "secret"}) as span:
        span.set_attribute("draft", "not recorded")
    telemetry.shutdown()
    assert attempted == []


def test_safe_attributes_strictly_filters_content_and_invalid_values():
    attributes = safe_attributes(
        {
            "research.project": "elf-v2",
            "research.scope_type": "attempt",
            "research.retry_count": 2,
            "research.duration_ms": 3.5,
            "prompt": "system prompt",
            "research.prompt": "secret",
            "draft": "intent body",
            "evidence": "metric records",
            "note": "authorization note",
            "api_key": "sk-secret",
            "stdout": "process output",
            "error.type": "bad error message with spaces",
            "research.object_id": "line\nsecret",
            "research.run_id": True,
            "research.duration_ms.invalid": float("nan"),
        }
    )

    assert attributes == {
        "research.project": "elf-v2",
        "research.scope_type": "attempt",
        "research.retry_count": 2,
        "research.duration_ms": 3.5,
    }
    assert safe_attributes(None) == {}
    assert safe_attributes({"research.retry_count": -1}) == {}
    assert safe_attributes({"research.retry_count": False}) == {}
    assert safe_attributes({"research.duration_ms": -0.1}) == {}
    assert safe_attributes({"research.duration_ms": float("nan")}) == {}


def test_injected_provider_records_only_safe_metadata_and_exception_class():
    provider = FakeProvider()
    telemetry = initialize_telemetry(
        {"enabled": True, "service_name": "ml-expd"},
        tracer_provider=provider,
    )

    with pytest.raises(RuntimeError, match="sensitive value"):
        with telemetry.span(
            "research.intent.prepare",
            {
                "research.project": "elf",
                "research.operation_id": "research.recommend",
                "prompt": "do not emit",
            },
        ) as span:
            span.set_attribute("stdout", "more secret content")
            span.set_attribute("research.status", "RUNNING")
            raise RuntimeError("sensitive value")

    name, kwargs, span = provider.tracer.started[0]
    assert name == "research.intent.prepare"
    assert kwargs == {
        "attributes": {
            "research.project": "elf",
            "research.operation_id": "research.recommend",
        },
        "record_exception": False,
        "set_status_on_exception": False,
    }
    assert span.attributes == {
        "research.status": "RUNNING",
        "error.type": "RuntimeError",
    }
    assert provider.requests[0][0] == "ml_exp_server"
    telemetry.shutdown()
    assert provider.shutdown_calls == 0


def test_invalid_span_service_provider_and_missing_sdk_fail_closed():
    provider = FakeProvider()
    telemetry = initialize_telemetry({"enabled": True}, tracer_provider=provider)
    with pytest.raises(ValueError, match="unsupported telemetry span"):
        with telemetry.span("research.prompt.secret"):
            pass

    with pytest.raises(TelemetryInitializationError, match="service_name"):
        initialize_telemetry(
            {"enabled": True, "service_name": "research console"},
            tracer_provider=provider,
        )
    with pytest.raises(TelemetryInitializationError, match="get_tracer"):
        initialize_telemetry({"enabled": True}, tracer_provider=object())
    with pytest.raises(TelemetryInitializationError, match="otlp_http_endpoint"):
        initialize_telemetry({"enabled": True, "otlp_http_endpoint": None})


def test_missing_sdk_failure_is_explicit_even_with_default_endpoint(monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(TelemetryInitializationError, match="not installed"):
        initialize_telemetry({"enabled": True})


def test_injected_exporter_uses_processor_factory_and_rejects_bad_provider():
    provider = FakeProvider()
    provider.processors = []
    provider.add_span_processor = provider.processors.append
    exporter = object()
    processor = object()

    telemetry = initialize_telemetry(
        {"enabled": True},
        tracer_provider=provider,
        exporter=exporter,
        span_processor_factory=lambda current: processor if current is exporter else None,
    )
    assert telemetry.enabled
    assert provider.processors == [processor]

    with pytest.raises(TelemetryInitializationError, match="cannot accept"):
        initialize_telemetry(
            {"enabled": True},
            tracer_provider=FakeProvider(),
            exporter=exporter,
            span_processor_factory=lambda current: processor,
        )


def test_otlp_http_initialization_owns_and_shuts_down_provider(monkeypatch):
    resources = []
    exporters = []
    processors = []
    providers = []

    class Resource:
        @classmethod
        def create(cls, attributes):
            resources.append(attributes)
            return ("resource", attributes)

    class TracerProvider(FakeProvider):
        def __init__(self, *, resource):
            super().__init__()
            self.resource = resource
            self.processors = []
            providers.append(self)

        def add_span_processor(self, processor):
            self.processors.append(processor)

    class OTLPSpanExporter:
        def __init__(self, *, endpoint, timeout):
            self.endpoint = endpoint
            self.timeout = timeout
            exporters.append(self)

    class BatchSpanProcessor:
        def __init__(self, exporter, *, export_timeout_millis=None):
            self.exporter = exporter
            self.export_timeout_millis = export_timeout_millis
            processors.append(self)

    modules = {
        "opentelemetry": ModuleType("opentelemetry"),
        "opentelemetry.exporter": ModuleType("opentelemetry.exporter"),
        "opentelemetry.exporter.otlp": ModuleType("opentelemetry.exporter.otlp"),
        "opentelemetry.exporter.otlp.proto": ModuleType("opentelemetry.exporter.otlp.proto"),
        "opentelemetry.exporter.otlp.proto.http": ModuleType(
            "opentelemetry.exporter.otlp.proto.http"
        ),
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": ModuleType(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        ),
        "opentelemetry.sdk": ModuleType("opentelemetry.sdk"),
        "opentelemetry.sdk.resources": ModuleType("opentelemetry.sdk.resources"),
        "opentelemetry.sdk.trace": ModuleType("opentelemetry.sdk.trace"),
        "opentelemetry.sdk.trace.export": ModuleType("opentelemetry.sdk.trace.export"),
    }
    modules[
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    ].OTLPSpanExporter = OTLPSpanExporter
    modules["opentelemetry.sdk.resources"].Resource = Resource
    modules["opentelemetry.sdk.trace"].TracerProvider = TracerProvider
    modules["opentelemetry.sdk.trace.export"].BatchSpanProcessor = BatchSpanProcessor
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    telemetry = initialize_telemetry(
        TelemetrySettings(
            enabled=True,
            otlp_http_endpoint="http://collector:4318/v1/traces",
            service_name="ml-expd",
        )
    )
    assert resources == [{"service.name": "ml-expd"}]
    assert exporters[0].endpoint == "http://collector:4318/v1/traces"
    assert exporters[0].timeout == OTLP_EXPORT_TIMEOUT_SECONDS
    assert processors[0].exporter is exporters[0]
    assert processors[0].export_timeout_millis == BATCH_EXPORT_TIMEOUT_MILLIS
    assert providers[0].processors == [processors[0]]
    telemetry.shutdown()
    assert providers[0].shutdown_calls == 1

    injected_provider = FakeProvider()
    injected_provider.processors = []
    injected_provider.add_span_processor = injected_provider.processors.append
    injected_exporter = object()
    configured = initialize_telemetry(
        {"enabled": True},
        tracer_provider=injected_provider,
        exporter=injected_exporter,
    )
    assert isinstance(injected_provider.processors[0], BatchSpanProcessor)
    assert injected_provider.processors[0].exporter is injected_exporter
    configured.shutdown()
    assert injected_provider.shutdown_calls == 0

    custom_processors = []

    def custom_processor_factory(current):
        custom_processors.append(("custom", current))
        return custom_processors[-1]

    custom = initialize_telemetry(
        TelemetrySettings(
            otlp_http_endpoint="http://collector:4318/v1/traces",
        ),
        span_processor_factory=custom_processor_factory,
    )
    assert providers[-1].processors == [custom_processors[0]]
    assert custom_processors[0][1] is exporters[-1]
    custom.shutdown()


def test_default_endpoint_and_missing_collector_do_not_block_or_leak(caplog):
    pytest.importorskip("opentelemetry.sdk.trace")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        unavailable_port = probe.getsockname()[1]

    telemetry = initialize_telemetry(
        TelemetrySettings(
            otlp_http_endpoint=f"http://127.0.0.1:{unavailable_port}/v1/traces"
        )
    )
    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        with telemetry.span(
            "research.intent.prepare",
            {
                "research.project": "elf",
                "prompt": "SENSITIVE_PROMPT_VALUE",
                "draft": "SENSITIVE_DRAFT_VALUE",
            },
        ) as span:
            span.set_attribute("stdout", "SENSITIVE_STDOUT_VALUE")
        span_elapsed = time.monotonic() - start
        shutdown_start = time.monotonic()
        telemetry.shutdown()
        shutdown_elapsed = time.monotonic() - shutdown_start

    assert TelemetrySettings().enabled is True
    assert TelemetrySettings().otlp_http_endpoint == DEFAULT_OTLP_HTTP_ENDPOINT
    assert span_elapsed < 0.5
    assert shutdown_elapsed < 3.0
    emitted_logs = caplog.text
    assert "SENSITIVE_PROMPT_VALUE" not in emitted_logs
    assert "SENSITIVE_DRAFT_VALUE" not in emitted_logs
    assert "SENSITIVE_STDOUT_VALUE" not in emitted_logs
