"""Freeze capabilities of the exact configured worker, not the server version.

Older images keep their original wire contract. Reconciliation reads only the
saved declaration; it cannot infer new capabilities from a replacement binary.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from ..storage import atomic_json


def load_features(path, expected):
    if not path.exists():
        return []
    value = json.loads(path.read_text())
    features = value.get("features")
    if (set(value) != {"worker_sha256", "features"}
            or value["worker_sha256"] != expected
            or not isinstance(features, list)
            or features not in ([], ["storage-layout.v1"])):
        raise ValueError("native worker capability binding differs")
    return features


def freeze_features(binary, expected, path):
    binary = Path(binary)
    if path.exists():
        return load_features(path, expected)
    def matches():
        return not binary.is_symlink() and "sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest() == expected
    if not matches():
        raise ValueError("native worker capability binary differs")
    result = subprocess.run([str(binary), "--capabilities"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            timeout=5, check=False)
    features = []
    if result.returncode == 0:
        if len(result.stdout) > 4096:
            raise ValueError("native worker capability response exceeds bound")
        value = json.loads(result.stdout)
        if set(value) != {"contract", "features"} or value["contract"] != "native-worker-capabilities.v1":
            raise ValueError("native worker capability contract differs")
        features = value["features"]
        if features not in ([], ["storage-layout.v1"]):
            raise ValueError("native worker capabilities are unsupported")
    if not matches():
        raise ValueError("native worker changed during capability inspection")
    atomic_json(path, {"worker_sha256": expected, "features": features})
    return features
