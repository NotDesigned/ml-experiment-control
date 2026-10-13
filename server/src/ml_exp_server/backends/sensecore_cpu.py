"""Shared CPU resource and exact-job checks; operation state stays with callers."""
from __future__ import annotations

import re

from experiment_control.backends.sensecore_rest import create_document

from ..projects.source_imports import IDENTITY


def validate_profile(profile, *, timeouts=(), debug=False):
    """Validate without changing a frozen profile or its quota selection."""
    if (any(type(profile.get(key)) is not int or profile[key] != expected
            for key, expected in (("gpus", 0), ("cpus", 2), ("memory_gb", 4)))
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.2c4g", profile.get("worker_spec", ""))
            or not IDENTITY.fullmatch(profile.get("aec2", ""))
            or profile.get("quota_type", "reserved") not in {"reserved", "spot"}
            or debug and "debug" not in profile["aec2"].casefold()
            or any(type(profile.get(key)) is not int or not 5 <= profile[key] <= 3600 for key in timeouts)):
        raise ValueError("CPU operations require the verified 2CPU/4GiB/0GPU profile and bounded time")


def verify_spec(rest, profile):
    specs = [row for row in rest.specs(profile) if row["name"] == profile["worker_spec"]]
    if len(specs) != 1:
        raise ValueError("ACP CPU-only specification is unavailable")
    spec = specs[0]
    if (spec["device"]["number"], spec["cpu"]["vcpu_allocatable"], spec["memory"]["allocatable"]) != (0, 2, 4):
        raise ValueError("ACP CPU specification allocates different resources")


def document(profile, name, image, command):
    body = create_document(profile, name, image, command, cpu_copy=True)
    body["roles"][0]["resource_spec"][0].update(
        requests={"cpu": "2", "memory": "3Gi"}, limits={"cpu": "2", "memory": "4Gi"})
    return body


def checked_job(job, name, uid=None):
    """Reject a replacement even when a provider reuses the original name."""
    if (not isinstance(job, dict) or job.get("name") != name
            or not isinstance(job.get("uid"), str) or not job["uid"]
            or uid is not None and job["uid"] != uid):
        raise ValueError("CPU job name or confirmed UID differs")
    return job


def find_job(rest, profile, name, uid=None):
    jobs = rest.find(profile, name)
    if len(jobs) > 1:
        raise ValueError("ambiguous CPU scheduler identity")
    return checked_job(jobs[0], name, uid) if jobs else None


def describe_job(rest, profile, name, uid):
    return checked_job(rest.describe(profile, name), name, uid)


def stop_job(rest, profile, name, uid):
    # The provider's existing batchStop is name based. This guard refuses a
    # known replacement; it does not claim an atomic compare-and-stop API.
    describe_job(rest, profile, name, uid)
    rest.stop(profile, name)
