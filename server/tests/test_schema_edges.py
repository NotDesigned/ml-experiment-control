"""Validation edges for daemon network and publication configuration."""

from __future__ import annotations

import pytest

from ml_exp_server.schemas import LocalWandbConfig, WandbCloudConfig


def test_local_wandb_accepts_dns_bind_host_and_explicit_none_url():
    config = LocalWandbConfig(bind_host="wandb.local", external_url=None)
    assert config.bind_host == "wandb.local"
    assert config.external_url is None


@pytest.mark.parametrize(("values", "message"), [
    ({"bind_host": "bad host"}, "bind_host"),
    ({"bind_host": "bad_host"}, "bind_host"),
    ({"external_url": "https://example.com:bad"}, "invalid port"),
    ({"enabled": True, "external_url": "https://example.com"},
     "managed local W&B cannot use external_url"),
    ({"enabled": True, "image": "bad image"}, "image contains unsupported"),
    ({"enabled": True, "docker_executable": "docker"}, "must be an absolute path"),
    ({"enabled": True, "managed": False}, "requires external_url"),
])
def test_local_wandb_rejects_unsafe_network_or_process_contract(values, message):
    with pytest.raises(ValueError, match=message):
        LocalWandbConfig(**values)


@pytest.mark.parametrize(("values", "message"), [
    ({"enabled": True}, "requires default_credential_ref"),
    ({"api_url": "http://api.example.com"}, "absolute HTTPS"),
    ({"dashboard_url": "https://user@example.com"}, "must not contain"),
    ({"dashboard_url": "https://example.com/path?token=x"}, "must not contain"),
])
def test_wandb_cloud_rejects_incomplete_or_credential_bearing_urls(values, message):
    with pytest.raises(ValueError, match=message):
        WandbCloudConfig(**values)
