"""Client updates never contact a service or run an unverified installer."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client import updater


def release(release_version="1.5.2", client_version="1.5.2"):
    names = [f"ml_experiment_client-{client_version}-py3-none-any.whl", "SHA256SUMS"]
    return {"draft": False, "prerelease": False, "tag_name": "v" + release_version,
            "assets": [{"name": name, "browser_download_url":
                f"https://github.com/{updater.REPOSITORY}/releases/download/v{release_version}/{name}"} for name in names]}


def wheel(name=updater.CLIENT, version="1.5.2", metadata=None):
    body = BytesIO()
    with ZipFile(body, "w") as archive:
        archive.writestr(f"ml_experiment_client-{version}.dist-info/METADATA",
                        metadata if metadata is not None else f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n")
        archive.writestr("ml_exp_client/__init__.py", "# Verified fixture; never installed.\n")
    return body.getvalue()


@pytest.fixture
def setup(monkeypatch):
    state = {"release": release(), "wheel": wheel(), "current": "1.5.1", "server": None,
             "downloads": [], "commands": []}
    def download(url, limit):
        state["downloads"].append((url, limit))
        if url.startswith("https://api.github.com/"):
            return json.dumps(state["release"]).encode()
        if url.endswith("SHA256SUMS"):
            return state.get("checksums", (sha256(state["wheel"]).hexdigest() + "  " + state["release"]["assets"][0]["name"] + "\n").encode())
        assert url.endswith(".whl")
        return state["wheel"]
    def install(command, **kwargs):
        path = Path(command[-1])
        assert path.read_bytes() == state["wheel"]
        state["commands"].append((command, kwargs, path))
        state["current"] = "1.5.2"
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(updater, "_download", download)
    monkeypatch.setattr(updater, "_installed", lambda name: state["server"] if name == "ml-experiment-server" else state["current"])
    monkeypatch.setattr(updater.util, "find_spec", lambda name: object())
    monkeypatch.setattr(updater.subprocess, "run", install)
    monkeypatch.setattr(updater, "build_opener", lambda *args: pytest.fail("real network is forbidden"))
    return state


def test_check_only_fetches_release_metadata_and_uses_independent_client_version(setup):
    setup["release"] = release("1.5.3", "1.5.2")
    setup["server"] = "1.5.0"
    assert updater.update_client(check=True) == {"current_version": "1.5.1", "target_version": "1.5.2",
        "release_version": "1.5.3", "update_available": True, "updated": False, "check_only": True}
    assert len(setup["downloads"]) == 1 and not setup["commands"]
    assert setup["downloads"][0][0].endswith("/latest")


def test_stable_version_order_is_numeric_instead_of_lexicographic(setup):
    setup["current"] = "1.9.9"
    setup["release"] = release("1.10.0", "1.10.0")
    assert updater.update_client(check=True)["update_available"]


def test_already_current_does_not_download_or_install_assets(setup):
    setup["current"] = "1.5.2"
    result = updater.update_client("1.5.2")
    assert not result["updated"] and not result["update_available"]
    assert len(setup["downloads"]) == 1 and setup["downloads"][0][0].endswith("/tags/v1.5.2")
    assert not setup["commands"]


def test_verified_wheel_uses_isolated_same_interpreter_and_removes_temporary_files(setup, monkeypatch):
    monkeypatch.setenv("ML_EXPD_API_TOKEN", "private-not-for-child")
    monkeypatch.setenv("PIP_TARGET", "/wrong-target")
    monkeypatch.setenv("PIP_PREFIX", "/wrong-prefix")
    monkeypatch.setenv("PIP_USER", "true")
    monkeypatch.setenv("PIP_CONFIG_FILE", "/wrong-config")
    monkeypatch.setenv("UV_PREFIX", "/wrong-prefix")
    monkeypatch.setenv("PYTHONPATH", "/wrong-modules")
    result = updater.update_client()
    command, kwargs, path = setup["commands"][0]
    assert command[:3] == [sys.executable, "-m", "pip"]
    assert all(flag in command for flag in ("--isolated", "--no-input", "--no-index", "--no-deps", "--disable-pip-version-check"))
    assert kwargs["capture_output"] and not kwargs["check"] and kwargs["timeout"] == 300
    assert not any(key in kwargs["env"] for key in ("ML_EXPD_API_TOKEN", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "UV_PREFIX", "PYTHONPATH"))
    assert {key: value for key, value in kwargs["env"].items() if key.startswith("PIP_")} == {"PIP_CONFIG_FILE": os.devnull}
    assert not path.parent.exists()
    assert result["updated"] and result["installed_version"] == "1.5.2"
    assert [limit for _, limit in setup["downloads"]] == [updater.METADATA_LIMIT, updater.WHEEL_LIMIT, updater.METADATA_LIMIT]


def test_uv_fallback_is_offline_no_config_and_has_explicit_interpreter(setup, monkeypatch):
    monkeypatch.setattr(updater.util, "find_spec", lambda name: None)
    monkeypatch.setattr(updater.shutil, "which", lambda name: "/known/uv")
    assert updater.update_client()["updated"]
    command = setup["commands"][0][0]
    assert command[:4] == ["/known/uv", "--no-config", "--offline", "--no-cache"]
    assert command[command.index("--python") + 1] == sys.executable
    assert "--no-index" in command and "--no-deps" in command


def test_no_installer_fails_before_downloading_assets(setup, monkeypatch):
    monkeypatch.setattr(updater.util, "find_spec", lambda name: None)
    monkeypatch.setattr(updater.shutil, "which", lambda name: None)
    with pytest.raises(ClientError, match="no installer"):
        updater.update_client()
    assert len(setup["downloads"]) == 1 and not setup["commands"]


def test_server_environment_cannot_self_update_but_core_and_worker_are_not_a_block(setup):
    setup["server"] = "1.5.0"
    with pytest.raises(ClientError, match="server deployment"):
        updater.update_client()
    assert len(setup["downloads"]) == 1 and not setup["commands"]
    setup["server"] = None
    assert updater.update_client()["updated"]


def test_downgrade_is_not_installed_even_for_an_explicit_release(setup):
    setup["current"] = "1.6.0"
    assert not updater.update_client(check=True)["update_available"]
    with pytest.raises(ClientError, match="downgrade"):
        updater.update_client("1.5.2")
    assert not setup["commands"] and len(setup["downloads"]) == 2


@pytest.mark.parametrize("value", ["v1.5.2", "1.5", "1.5.2rc1", "01.5.2", "1.5.2/SECRET", 12])
def test_invalid_requested_release_is_rejected_without_network(setup, value):
    with pytest.raises(ClientError, match="stable") as failure:
        updater.update_client(value)
    assert "SECRET" not in str(failure.value) and not setup["downloads"]


@pytest.mark.parametrize("change", ["draft", "prerelease", "tag", "missing_wheel", "duplicate_wheel", "missing_sums", "duplicate_sums", "foreign_asset", "wrong_release"])
def test_invalid_or_ambiguous_official_release_fails_closed(setup, change):
    value = setup["release"]
    if change in {"draft", "prerelease"}: value[change] = True
    if change == "tag": value["tag_name"] = "v1.5.2rc1"
    if change == "missing_wheel": value["assets"] = value["assets"][1:]
    if change == "duplicate_wheel": value["assets"].append(deepcopy(value["assets"][0]))
    if change == "missing_sums": value["assets"] = value["assets"][:1]
    if change == "duplicate_sums": value["assets"].append(deepcopy(value["assets"][1]))
    if change == "foreign_asset": value["assets"][0]["browser_download_url"] = "https://user:SECRET@example.org/file"
    with pytest.raises(ClientError) as failure:
        updater.update_client("1.5.3" if change == "wrong_release" else None)
    assert "SECRET" not in str(failure.value) and not setup["commands"] and len(setup["downloads"]) == 1


@pytest.mark.parametrize("sums", [b"bad SECRET", b"\xff", b"", b"0" * 64 + b"  other.whl\n", b"0" * 64 + b"  ml_experiment_client-1.5.2-py3-none-any.whl\n"])
def test_bad_checksums_never_reach_installer(setup, sums):
    setup["checksums"] = sums
    with pytest.raises(ClientError, match="SHA256") as failure:
        updater.update_client()
    assert "SECRET" not in str(failure.value) and not setup["commands"]


def test_checksum_binary_format_and_relative_basename_are_supported(setup):
    setup["checksums"] = (sha256(setup["wheel"]).hexdigest() + " *./" + setup["release"]["assets"][0]["name"] + "\n").encode()
    assert updater.update_client()["updated"]


def test_duplicate_checksum_for_target_is_ambiguous(setup):
    line = sha256(setup["wheel"]).hexdigest() + "  " + setup["release"]["assets"][0]["name"] + "\n"
    setup["checksums"] = (line * 2).encode()
    with pytest.raises(ClientError, match="SHA256"):
        updater.update_client()
    assert not setup["commands"]


@pytest.mark.parametrize("bad", ["not_zip", "foreign_name", "wrong_version", "duplicate_name", "oversized_metadata"])
def test_wheel_metadata_must_match_exact_client_and_version(setup, bad):
    if bad == "not_zip": setup["wheel"] = b"SECRET not a zip"
    if bad == "foreign_name": setup["wheel"] = wheel(name="other-project-SECRET")
    if bad == "wrong_version": setup["wheel"] = wheel(version="1.5.1")
    if bad == "duplicate_name": setup["wheel"] = wheel(metadata="Name: ml-experiment-client\nName: SECRET\nVersion: 1.5.2\n")
    if bad == "oversized_metadata": setup["wheel"] = wheel(metadata="x" * (updater.METADATA_LIMIT + 1))
    with pytest.raises(ClientError, match="identity") as failure:
        updater.update_client()
    assert "SECRET" not in str(failure.value) and not setup["commands"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "oserror", "wrong_installed_version"])
def test_installer_failure_never_claims_success_or_exposes_output(setup, monkeypatch, failure):
    paths = []
    def install(command, **kwargs):
        paths.append(Path(command[-1]))
        if failure == "timeout": raise subprocess.TimeoutExpired(command, 300, output=b"SECRET")
        if failure == "oserror": raise OSError("SECRET")
        return SimpleNamespace(returncode=1 if failure == "exit" else 0, stdout=b"SECRET", stderr=b"SECRET")
    monkeypatch.setattr(updater.subprocess, "run", install)
    with pytest.raises(ClientError) as error:
        updater.update_client()
    assert "SECRET" not in str(error.value) and all(not path.parent.exists() for path in paths)


def test_download_is_bounded_unauthenticated_and_restricts_redirects(setup, monkeypatch):
    # Restore the actual downloader; all urllib operations are still fake.
    requests = []
    class Response(BytesIO):
        def geturl(self): return "https://release-assets.githubusercontent.com/asset"
    class Opener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return Response(b"12345")
    monkeypatch.setattr(updater, "build_opener", lambda *args: Opener())
    original = DOWNLOAD
    assert original("https://github.com/official", 5) == b"12345"
    request, timeout = requests[0]
    assert request.get_method() == "GET" and timeout == 30
    assert set(key.lower() for key in request.headers) == {"user-agent", "accept"}
    assert original("https://api.github.com/releases/latest", 5) == b"12345"
    assert requests[-1][0].get_header("Accept") == "application/vnd.github+json"
    with pytest.raises(ClientError, match="size limit"):
        original("https://github.com/official", 4)
    with pytest.raises(ClientError, match="outside GitHub"):
        updater._GithubRedirect().redirect_request(None, None, 302, "", {}, "https://user:SECRET@foreign.example")


@pytest.mark.parametrize("url", ["http://github.com/private", "https://SECRET@github.com/private", "https://foreign.example/SECRET"])
def test_foreign_or_credentialed_download_url_never_opens_network(setup, url):
    with pytest.raises(ClientError, match="invalid official") as error:
        DOWNLOAD(url, 5)
    assert "SECRET" not in str(error.value)


def test_invalid_release_json_has_fixed_error_and_never_installs(setup, monkeypatch):
    monkeypatch.setattr(updater, "_download", lambda *args: b"SECRET invalid JSON")
    with pytest.raises(ClientError, match="metadata is invalid") as error:
        updater.update_client()
    assert "SECRET" not in str(error.value) and not setup["commands"]


@pytest.mark.parametrize("failure", [OSError("SECRET"), ValueError("SECRET")])
def test_network_failure_diagnostics_do_not_echo_credentials(setup, monkeypatch, failure):
    class Opener:
        def open(self, *args, **kwargs): raise failure
    monkeypatch.setattr(updater, "build_opener", lambda *args: Opener())
    with pytest.raises(ClientError, match="connectivity") as error:
        DOWNLOAD("https://github.com/official", 5)
    assert "SECRET" not in str(error.value)


DOWNLOAD = updater._download
