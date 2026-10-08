"""Shared diagnostic redaction, including platform-generated credentials."""

from experiment_control.redaction import redact_line


def test_prefixed_credentials_and_bearer_are_redacted():
    value = (
        "APPTAINER_DOCKER_PASSWORD=registry-secret "
        "ML_EXPD_BOOTSTRAP_TOKEN='bootstrap-secret' "
        'API_TOKEN="api-secret" Authorization: Bearer bearer-secret '
        "loss=2.0 seed=42"
    )
    result = redact_line(value)
    for secret in ("registry-secret", "bootstrap-secret", "api-secret", "bearer-secret"):
        assert secret not in result
    assert "loss=2.0 seed=42" in result
    assert result.count("<redacted>") >= 4


def test_signed_urls_remove_all_query_credentials_but_keep_location():
    for key in ("X-Amz-Signature", "X-Goog-Signature", "signature", "sig"):
        result = redact_line(
            f"GET https://user:password@objects.test/archive.zip?"
            f"X-Amz-Credential=private-owner&{key}=private-signature&Expires=600"
        )
        assert result == "GET https://<redacted>@objects.test/archive.zip?<redacted>"


def test_unsigned_queries_preserve_safe_parameters_and_remove_credentials():
    value = "https://objects.test/files?page=2&X-Amz-Credential=private&token=secret"
    assert redact_line(value) == (
        "https://objects.test/files?page=2&X-Amz-Credential=<redacted>&token=<redacted>"
    )
