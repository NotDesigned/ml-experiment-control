from __future__ import annotations

import pytest

from experiment_control.submission import (
    require_submission_intent,
    submission_marker,
    validate_submission_token,
)


TOKEN = "d" * 32


def test_submission_intent_accepts_nested_and_legacy_flat_requests():
    nested = {"submission_token": TOKEN, "request": {"scheduler_name": "job"}}
    assert require_submission_intent(nested) == (TOKEN, nested["request"])

    flat = {"submission_token": TOKEN, "scheduler_name": "job"}
    assert require_submission_intent(flat) == (TOKEN, flat)
    assert submission_marker(TOKEN) == f"ml-exp-{TOKEN}"


@pytest.mark.parametrize(
    ("intent", "error"),
    [
        (None, "requires a durable submission intent"),
        ({"submission_token": "short"}, "128-bit hexadecimal"),
        ({"submission_token": TOKEN, "request": []}, "request must be a mapping"),
    ],
)
def test_submission_intent_rejects_unrecoverable_identity(intent, error):
    with pytest.raises(RuntimeError, match=error):
        require_submission_intent(intent)

    if intent and intent.get("submission_token") == "short":
        with pytest.raises(RuntimeError, match="128-bit hexadecimal"):
            validate_submission_token(intent["submission_token"])
