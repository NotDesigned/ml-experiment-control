#!/usr/bin/env python3
"""Which client artifacts a release needs, without building or running any of them."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_native_release as tool  # noqa: E402


def elf_arm64():
    return b"\x7fELF\x02\x01" + bytes(12) + struct.pack("<H", 183) + bytes(44)


def pe_x64():
    return b"MZ" + bytes(58) + struct.pack("<I", 64) + b"PE\0\0\x64\x86"


def macho(cpu):
    return b"\xcf\xfa\xed\xfe" + struct.pack("<I", cpu) + bytes(56)


CONTENT = {
    "aarch64-unknown-linux-musl": elf_arm64(),
    "x86_64-pc-windows-msvc": pe_x64(),
    "x86_64-apple-darwin": macho(0x01000007),
    "aarch64-apple-darwin": macho(0x0100000c),
}


class Clients(unittest.TestCase):
    def build(self, targets):
        """Runs as far as the client checks. Nothing here is executed as a program."""
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            binaries, clients = temp / "bin", temp / "clients"
            binaries.mkdir()
            clients.mkdir()
            version = tool.tomllib.loads((ROOT / "rust/Cargo.toml").read_text())["workspace"]["package"]["version"]
            for name in tool.BINARIES:
                path = binaries / name
                path.write_text(f"#!/bin/sh\necho '{name} {version}'\n")
                path.chmod(0o755)
            for target in targets:
                suffix = ".exe" if "windows" in target else ""
                (clients / f"ml-exp-{target}{suffix}").write_bytes(CONTENT[target])
            original = tool.static_worker
            tool.static_worker = lambda path: True
            try:
                return tool.build(binaries, temp / "out", version, clients)
            finally:
                tool.static_worker = original

    def targets(self, manifest):
        return {asset["target"] for asset in manifest["assets"]}

    def test_a_release_ships_whichever_mac_clients_finished_and_never_runs_a_client(self):
        required = list(tool.REQUIRED_CLIENT_TARGETS)
        for optional in ([], ["x86_64-apple-darwin"], list(tool.OPTIONAL_CLIENT_TARGETS)):
            try:
                manifest = self.build(required + optional)
            except subprocess.CalledProcessError as error:
                self.fail(f"it tried to run something: {error}")
            self.assertTrue(set(required + optional) <= self.targets(manifest))
            for target in set(tool.OPTIONAL_CLIENT_TARGETS) - set(optional):
                self.assertNotIn(target, self.targets(manifest))

    def test_a_missing_required_client_is_refused(self):
        with self.assertRaises(ValueError):
            self.build(["aarch64-unknown-linux-musl"])

    def test_a_client_in_the_wrong_format_is_refused(self):
        original = CONTENT["x86_64-pc-windows-msvc"]
        CONTENT["x86_64-pc-windows-msvc"] = elf_arm64()
        try:
            with self.assertRaises(ValueError):
                self.build(list(tool.REQUIRED_CLIENT_TARGETS))
        finally:
            CONTENT["x86_64-pc-windows-msvc"] = original


if __name__ == "__main__":
    unittest.main()
