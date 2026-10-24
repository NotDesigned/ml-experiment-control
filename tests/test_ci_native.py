"""A manifest may describe only the exact compiler selected by this process."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import ci_native as native


HOST = "x86_64-unknown-linux-gnu"
VALUE = "1.85.0 (4d91de4e4 2025-02-17)"


def installation(root, *, channel="stable", value=VALUE):
    compiler = root / (channel + "-" + HOST) / "bin/rustc"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("selected installed binary")
    compiler.chmod(0o755)
    manifest = compiler.parent.parent / "lib/rustlib/multirust-channel-manifest.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('manifest-version="2"\ndate="2025-02-20"\n'
                        f'[pkg.rust]\nversion="{value}"\n[pkg.rustc]\nversion="{value}"\n'
                        f'[pkg.rustc.target.{HOST}]\navailable=true\n')
    return compiler, manifest


@pytest.fixture
def selected(tmp_path, monkeypatch):
    compiler, manifest = installation(tmp_path)
    rustup = tmp_path / "rustup"
    rustup.write_text("rustup shim")
    shim = tmp_path / "rustc"
    shim.symlink_to(rustup)
    monkeypatch.delenv("RUSTC", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda name: str(shim if name == "rustc" else rustup))
    commands = []
    def command(argv):
        commands.append(argv)
        return str(compiler) if argv[-2:] == ["which", "rustc"] else "rustc " + VALUE
    monkeypatch.setattr(native, "command", command)
    return compiler, manifest, shim, rustup, commands


def test_exact_selected_shim_manifest_avoids_loading_compiler_and_check_requires_real_version(selected):
    compiler, manifest, shim, rustup, commands = selected
    result = native.identify()
    assert result["toolchain"] == "1.85.0-4d91de4e4-2025-02-17"
    assert len(commands) == 1 and commands[0] == [str(rustup), "which", "rustc"]
    assert "\n" not in result["abi"] and " " not in result["toolchain"]
    assert native.identify(check=True) == result
    assert commands[-1] == [str(shim), "--version"]


def test_actual_selected_binary_on_path_and_toolchain_override_are_preserved(selected, tmp_path, monkeypatch):
    compiler, _, _, rustup, commands = selected
    monkeypatch.setattr(native.shutil, "which", lambda name: str(compiler if name == "rustc" else rustup))
    assert native.identify()["toolchain"].startswith("1.85.0-")
    other, _ = installation(tmp_path, channel="1.86.0", value="1.86.0 (abcdef012 2025-02-18)")
    monkeypatch.setenv("RUSTUP_TOOLCHAIN", "1.86.0")
    monkeypatch.setattr(native.shutil, "which", lambda name: str(other if name == "rustc" else rustup))
    monkeypatch.setattr(native, "command", lambda args: str(other))
    assert native.identify()["toolchain"] == "1.86.0-abcdef012-2025-02-18"


@pytest.mark.parametrize("change", ["no-manifest", "bad-toml", "version", "component", "commit", "host",
                                  "date", "channel", "relative", "rustup-failure", "no-rustup", "non-executable"])
def test_unverifiable_manifest_falls_back_to_the_real_path_compiler(selected, monkeypatch, change):
    compiler, manifest, shim, rustup, commands = selected
    text = manifest.read_text()
    if change == "no-manifest": manifest.unlink()
    elif change == "non-executable": compiler.chmod(0o600)
    elif change == "bad-toml": manifest.write_text("not [valid")
    elif change == "version": manifest.write_text(text.replace(VALUE, "1.85.0 unknown"))
    elif change == "component": manifest.write_text(text.replace('[pkg.rust]\nversion="' + VALUE, '[pkg.rust]\nversion="1.86.0 (abcdef012 2025-02-18)'))
    elif change == "commit": manifest.write_text(text.replace('[pkg.rustc]\n', '[pkg.rustc]\ngit_commit_hash="' + "a" * 40 + '"\n'))
    elif change == "host": manifest.write_text(text.replace(HOST, "aarch64-unknown-linux-gnu"))
    elif change == "date": manifest.write_text(text.replace('date="2025-02-20"', 'date="2025-02-01"'))
    elif change == "channel": manifest.write_text(text.replace(VALUE, "1.85.0-nightly (4d91de4e4 2025-02-17)"))
    elif change == "no-rustup": monkeypatch.setattr(native.shutil, "which", lambda name: str(shim) if name == "rustc" else None)
    elif change in {"relative", "rustup-failure"}:
        def command(argv):
            commands.append(argv)
            if argv[-1] == "--version": return "rustc " + VALUE
            if change == "relative": return "relative/bin/rustc"
            raise subprocess.CalledProcessError(1, argv)
        monkeypatch.setattr(native, "command", command)
    assert native.identify()["toolchain"] == "1.85.0-4d91de4e4-2025-02-17"
    assert commands[-1] == [str(shim), "--version"]


@pytest.mark.parametrize("override", [False, True])
def test_custom_compiler_never_uses_another_toolchains_manifest(selected, tmp_path, monkeypatch, override):
    _, _, shim, rustup, commands = selected
    custom = tmp_path / "custom-rustc"
    custom.write_text("custom compiler wrapper")
    if override:
        monkeypatch.setenv("RUSTC", str(custom))
    else:
        monkeypatch.setattr(native.shutil, "which", lambda name: str(custom if name == "rustc" else rustup))
    assert native.identify()["toolchain"].startswith("1.85.0-")
    assert commands[-1] == [str(custom), "--version"]


def test_real_compiler_disagreement_or_invalid_output_cannot_emit_a_cache_key(selected, monkeypatch):
    compiler, _, _, _, _ = selected
    monkeypatch.setattr(native, "command", lambda argv: str(compiler) if argv[-1] == "rustc" else "rustc 1.86.0 (abcdef012 2025-02-18)")
    with pytest.raises(ValueError, match="disagrees"):
        native.identify(check=True)
    for text in ("1.85.0", "rustc 1.85.0 (unknown)", "rustc 1.85.0 (abcdef012 2025-02-31)",
                 "rustc 1.85.0 (abcdef012 2025-02-18)\nabi=forged", "rustc 01.85.0 (abcdef012 2025-02-18)"):
        monkeypatch.setattr(native, "command", lambda argv, text=text: text)
        with pytest.raises(ValueError): native.real_version(compiler)


def test_cli_emits_only_safe_output_and_real_command_uses_no_shell(selected, monkeypatch, capsys):
    native.main([])
    assert capsys.readouterr().out.startswith("toolchain=1.85.0-4d91de4e4-2025-02-17\nabi=")


def test_real_command_uses_an_argv_timeout_and_no_shell(monkeypatch):
    calls = []
    monkeypatch.setattr(native.subprocess, "check_output", lambda argv, **kwargs: calls.append((argv, kwargs)) or "version\n")
    assert native.command(["rustc", "--version"]) == "version"
    assert calls == [(["rustc", "--version"], {"text": True, "timeout": 15})]
