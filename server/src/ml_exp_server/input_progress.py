"""Bounded exact-Attempt data evidence; logs never grant execution authority."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from .artifacts import open_directory

MARKER = 'ML_EXPD_INPUT_PROGRESS='
PHASES = {'INITIALIZING', 'WAITING_FOR_CACHE', 'DOWNLOADING', 'EXTRACTING', 'VERIFYING', 'PUBLISHING', 'MOUNTING', 'READY', 'FAILED'}


def read_input_progress(attempt: Path):
    latest = None
    legacy_failed = False
    try:
        descriptor = open_directory(attempt)
    except (OSError, ValueError):
        return None
    try:
        for name in ('stdout.log', 'stderr.log'):
            try:
                file = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
                with os.fdopen(file, 'rb') as body:
                    body.seek(0, 2)
                    body.seek(max(0, body.tell() - 128 * 1024))
                    lines = body.read(128 * 1024).decode(errors='replace').splitlines()
            except OSError:
                continue
            for line in lines:
                legacy_failed |= line == 'ML_EXPD_INPUT_DELIVERY=FAILED'
                if not line.startswith(MARKER):
                    continue
                try:
                    value = json.loads(line[len(MARKER):])
                    # Project code can print arbitrary log text. Expose only a
                    # bounded, validated observational schema, never raw values.
                    at = datetime.fromisoformat(value['at'])
                    if (at.tzinfo is None or value['phase'] not in PHASES
                            or not re.fullmatch(r'asset\.[0-9a-f]{64}|none', value['asset_id'])):
                        continue
                    result = {key: value[key] for key in ('at', 'phase', 'asset_id')}
                    for key in ('received_bytes', 'expected_bytes', 'elapsed_seconds', 'bytes_per_second', 'http_status', 'errno'):
                        number = value.get(key)
                        if type(number) in (int, float) and 0 <= number < 10 ** 20:
                            result[key] = number
                    for key in ('code', 'error_class', 'failed_phase'):
                        word = value.get(key)
                        if isinstance(word, str) and re.fullmatch(r'[A-Za-z0-9_]{1,80}', word):
                            result[key] = word
                    if latest is None or at >= datetime.fromisoformat(latest['at']):
                        latest = result
                except (ValueError, TypeError, KeyError):
                    continue
    finally:
        os.close(descriptor)
    if latest is not None:
        if latest['phase'] == 'FAILED':
            latest.setdefault('code', 'INPUT_DELIVERY_FAILED_DETAILS_UNAVAILABLE')
        latest['evidence_source'] = 'exact_attempt_worker_log'
        latest['seconds_since_progress'] = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(latest['at'])).total_seconds())
        return latest
    if legacy_failed:
        return {'phase': 'FAILED', 'code': 'INPUT_DELIVERY_FAILED_DETAILS_UNAVAILABLE',
                'evidence_source': 'exact_attempt_legacy_log', 'at': None,
                'message': 'This historical worker recorded no internal exception; inspect or reproduce without resubmitting training'}
    return None
