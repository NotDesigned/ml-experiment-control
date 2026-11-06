#!/usr/bin/env python3
"""Structural guard around the no-replay code. It reads files and compiles nothing.

It catches two kinds of drift: a sealed file changed, or a new way to reach a
provider create appeared. It does not parse reconcile bodies and it cannot prove
ordering. Behaviour tests remain the proof: `server/tests/cloud_no_replay.rs`,
`server/tests/cloud_runs.rs`, `backends/slurm/tests/submission_tests.rs` and
`backends/cloud/outbox/tests.rs`.

A pin changes only when the owner has approved that file's change. To print the
current digests, run this file with --print-pins.
"""
from collections import Counter
import hashlib
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CRATES = Path("rust/crates")
SERVER = CRATES / "server/src"

PINS = {
    "rust/crates/server/src/backends/cloud/outbox.rs": "ff7c6a30b9d04c530da27e63fe9d81f0fdb52b07ed9469a5ef01aab729653f73",
    "rust/crates/server/src/backends/cloud/outbox/tests.rs": "b32c6152feb9d3950fe73200448bc64076904f77f43a2c0839e6e583a1be9bab",
    "rust/crates/server/src/backends/cloud/action.rs": "a09c4a78734245d578dbbbaa75c62a86e0ee2a49f81737c100e4cc5e2d7123dc",
    "rust/crates/server/src/backends/slurm/action.rs": "15ad4038d8525dde13d1439c97faa1475bc371e078603f3c8eb1ab0444b609ba",
    "rust/crates/server/src/backends/slurm/transport.rs": "1be44752888a9c93cbd7d4ac81643ab3c2977f7c345e61f8e36410ca60c9af5c",
    "rust/crates/server/src/backends/slurm/tests/submission_tests.rs": "f0bec35ffa76196fe0d8845acf561a9026d779ec81a56274172b64e39f53cc85",
    "rust/crates/server/src/backends/slurm/tests/action_tests.rs": "4947f58528888cf04ab64acaa39f8b16e6adeeb02445de9c582b3a7c99ed4bcc",
    "rust/crates/server/src/operations/store.rs": "1038a3f5be12140276318ece41b24432f18d8f87ed3d5df9c404a5168832661f",
    "rust/crates/server/tests/cloud_runs.rs": "b279b151a28e9168599c902cb490df99d8340d26ed054765a32fe85546afe926",
    "rust/crates/core/src/operations.rs": "f956e2c25597de20d76b42b04cebbb11b52083e6ca85275ab751fbb279cc3ec2",
    "rust/crates/core/src/slurm/action.rs": "61d6fb7e14248434b39f5a73208412d232d9134c784512e114c12892b8de9548",
    "rust/crates/core/src/cloud/action.rs": "5829a8a03252f9100341b46217e37e9f40f202630f251377d3777f6faba3d13f",
    "rust/crates/core/src/cloud/exchange.rs": "0d8257ef48ca52d3bb560d4bab582d6df28af98988cc45c100df60f74e4b664a",
    "rust/crates/worker/src/slurm/ledger.rs": "4c945a908455aa68bdcdadbc5fbfaa3c9e4cc0f6bb373c56511423ce68138910",
    "rust/crates/worker/src/slurm/ledger/artifacts.rs": "0ef666d31c8a89f19c595ad19e75974eb67d220cb9f5c50efdf8efe6cee53995",
    "rust/crates/worker/src/slurm/ledger/checkpoints.rs": "f54f98d9a36bf7d47ad75e79202e69c7b56035af2ec62048b08b3a774a6e082c",
    "rust/crates/worker/src/slurm/ledger/history.rs": "95a9db48c2cb38e8020510f6d3f4a15c5bf33ef0dc62bb56f22854d0ded7c254",
    "rust/crates/worker/src/slurm/ledger/jobs.rs": "695018e2b31111048c48eac6581140aa6f8f14b8c98561bd6f4f4efa2a0909f0",
    "rust/crates/worker/src/slurm/ledger/observation.rs": "19b00eb7ac7bbc8e5023e39a05a29f9f39beb0e32bfd6c1700916620c495ad04",
    "rust/crates/worker/src/slurm/ledger/staging.rs": "002a35855102960c2349606b2cb2e993eb49e298a46ac02ae86faf1ded035a2a",
}

# Every line in the server crate's production code that calls a method named
# `create`, by file name and text. Most compile a provider request and send
# nothing. A new site must be reviewed against the no-replay rules, then added.
KNOWN_CREATE_SITES = Counter(
    {
        ("adapter.rs", "Profile::Pai(p) => self.call(p.create(job, allocation), cancelled)?,"): 1,
        ("adapter.rs", "self.call(p.create(job), cancelled)?"): 1,
        ("outbox.rs", "p.create(job, allocation)"): 1,
        ("outbox.rs", "p.create(job).map_err(why!(Error::configuration()))?;"): 1,
        ("outbox.rs", "match guard().and_then(|_| adapter.create(job, allocation, cancelled)) {"): 1,
        ("actions.rs", "cloud::adapter::Profile::Pai(p) => p.create(&execution.job, execution.allocation),"): 1,
        ("actions.rs", "cloud::adapter::Profile::Sensecore(p) => p.create(&execution.job),"): 1,
        ("preparations.rs", "cloud::adapter::Profile::Pai(p) => p.create("): 1,
        ("preparations.rs", "cloud::adapter::Profile::Sensecore(p) => p.create(&job),"): 1,
        ("script_preparation.rs", "p.create(&job, ml_exp_core::cloud::pai::Allocation::Guaranteed)"): 1,
        ("script_preparation.rs", "cloud::adapter::Profile::Sensecore(p) => p.create(&job),"): 1,
        # Debug sessions have their own durable claim, saved before this request.
        ("debug.rs", ".create(&saved.recipe)"): 1,
    }
)
ADAPTER_CREATE_HOME = SERVER / "backends/cloud/outbox.rs"
SLURM_SUBMIT_HOME = SERVER / "backends/slurm/action.rs"
RUNS_MODULE = SERVER / "runs"
RUNS_FORBIDDEN = re.compile(
    r"\b(Adapter|Outbox|Transport|Runner)\b|\b(adapter|outbox|transport)::|::(adapter|outbox|transport)\b"
)


def is_test_file(path):
    return (
        "tests" in path.parts
        or path.name == "tests.rs"
        or path.name.endswith("_tests.rs")
    )


def rust_files(root, relative):
    return sorted(p.relative_to(root) for p in (root / relative).rglob("*.rs"))


def code_lines(root, path):
    """Stripped lines, without line comments."""
    for line in (root / path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            yield line


def production_lines(root, path):
    """Code lines outside test files and outside a top-level inline test module.

    rustfmt, which CI enforces, puts that module's closing brace in column zero.
    A string that also does so ends the skip early, which can only add lines.
    """
    if is_test_file(path):
        return
    lines = (root / path).read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        if lines[index] == "#[cfg(test)]" and re.fullmatch(
            r"mod \w+ \{", lines[index + 1] if index + 1 < len(lines) else ""
        ):
            while index < len(lines) and lines[index] != "}":
                index += 1
        else:
            line = lines[index].strip()
            if line and not line.startswith("//"):
                yield line
        index += 1


def digest(root, relative):
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


def pin_failures(root, pins=PINS):
    failures = []
    for relative, expected in pins.items():
        if not (root / relative).is_file():
            failures.append(f"{relative}: sealed file is missing")
        elif digest(root, relative) != expected:
            failures.append(f"{relative}: sealed file changed")
    return failures


def adapter_create_failures(root):
    failures = []
    for path in rust_files(root, CRATES):
        if path == ADAPTER_CREATE_HOME:
            continue
        if any("adapter.create(" in line for line in code_lines(root, path)):
            failures.append(f"{path}: adapter.create( belongs only in {ADAPTER_CREATE_HOME}")
    return failures


def slurm_submit_failures(root):
    failures = []
    found = False
    for path in rust_files(root, CRATES):
        if any("Request::submit(" in line for line in production_lines(root, path)):
            if path == SLURM_SUBMIT_HOME:
                found = True
            else:
                failures.append(f"{path}: Request::submit( belongs only in {SLURM_SUBMIT_HOME}")
    if not found:
        failures.append(f"{SLURM_SUBMIT_HOME}: the one Request::submit( site has moved")
    return failures


def create_site_failures(root, known=KNOWN_CREATE_SITES):
    observed = Counter()
    for path in rust_files(root, SERVER):
        for line in production_lines(root, path):
            if ".create(" in line:
                observed[(path.name, line)] += 1
    failures = []
    for (name, line), count in sorted((observed - known).items()):
        failures.append(f"{name}: unreviewed create site ({count}): {line}")
    for (name, line), count in sorted((known - observed).items()):
        failures.append(f"{name}: known create site is gone ({count}): {line}")
    return failures


def runs_module_failures(root):
    """Applies once `runs/` exists: it decides above the wrappers, never effects."""
    if not (root / RUNS_MODULE).is_dir():
        return []
    failures = []
    for path in rust_files(root, RUNS_MODULE):
        for line in code_lines(root, path):
            if RUNS_FORBIDDEN.search(line):
                failures.append(f"{path}: runs/ must not reach an effect: {line}")
    return failures


def tree_crossing_failures(root):
    """Handlers that serve either backend sit above both trees, not inside one."""
    failures = []
    for home, pattern in [
        ("backends/slurm", r"\bOwner::Cloud\b|\bapi\.cloud\(\)|\bapi\.cloud_only\(\)|\bbackends::cloud\b"),
        ("backends/cloud", r"\bOwner::Slurm\b|\bapi\.submissions\(\)|\bapi\.slurm_only\(\)"),
    ]:
        if not (root / SERVER / home).is_dir():
            failures.append(f"{home}: the tree has moved, so this rule reads nothing")
        for path in rust_files(root, SERVER / home):
            for line in production_lines(root, path):
                if re.search(pattern, line):
                    failures.append(f"{path}: {home} must not reach the other backend: {line}")
    return failures


def service_crossing_failures(root):
    """Neither service names the other's `Service`. Slurm asks a builder through the
    `RuntimeBuilder` trait, and anything else reaches a service as data from its caller."""
    failures = []
    for home, other in [
        ("backends/cloud/service", r"(?:submissions|slurm::service)"),
        ("backends/slurm/service", r"(?:cloud_runs|cloud::service)"),
    ]:
        pattern = re.compile(rf"\b{other}::Service\b")
        if not (root / SERVER / home).is_dir():
            failures.append(f"{home}: the service has moved, so this rule reads nothing")
        for path in rust_files(root, SERVER / home):
            for line in code_lines(root, path):
                if pattern.search(line):
                    failures.append(f"{path}: {home} must not name the other service: {line}")
    return failures


class Structure(unittest.TestCase):
    def test_sealed_files_are_byte_identical(self):
        self.assertEqual(pin_failures(ROOT), [])

    def test_adapter_create_occurs_only_in_the_outbox(self):
        self.assertEqual(adapter_create_failures(ROOT), [])
        self.assertTrue(
            any("adapter.create(" in line for line in code_lines(ROOT, ADAPTER_CREATE_HOME))
        )

    def test_slurm_submit_occurs_only_in_its_runner(self):
        self.assertEqual(slurm_submit_failures(ROOT), [])

    def test_every_create_site_is_known(self):
        self.assertEqual(create_site_failures(ROOT), [])

    def test_runs_module_reaches_no_effect(self):
        self.assertTrue((ROOT / RUNS_MODULE).is_dir(), "the rule must have a module to check")
        self.assertEqual(runs_module_failures(ROOT), [])

    def test_neither_service_names_the_other(self):
        self.assertEqual(service_crossing_failures(ROOT), [])

    def test_neither_backend_tree_reaches_the_other(self):
        self.assertEqual(tree_crossing_failures(ROOT), [])


class GuardDetects(unittest.TestCase):
    """The guard itself is checked against a small tree of its own."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.write(ADAPTER_CREATE_HOME, "adapter.create(job);\n")
        self.write(SLURM_SUBMIT_HOME, "Request::submit(intent);\n")
        self.write(SERVER / "debug.rs", ".create(&saved.recipe)\n")
        self.known = Counter(
            {("outbox.rs", "adapter.create(job);"): 1, ("debug.rs", ".create(&saved.recipe)"): 1}
        )

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_a_clean_tree_passes(self):
        self.assertEqual(adapter_create_failures(self.root), [])
        self.assertEqual(slurm_submit_failures(self.root), [])
        self.assertEqual(create_site_failures(self.root, self.known), [])
        self.assertEqual(runs_module_failures(self.root), [])

    def test_a_changed_or_missing_sealed_file_fails(self):
        relative = str(ADAPTER_CREATE_HOME)
        pins = {relative: digest(self.root, relative)}
        self.assertEqual(pin_failures(self.root, pins), [])
        self.write(ADAPTER_CREATE_HOME, "adapter.create(job); // edited\n")
        self.assertEqual(len(pin_failures(self.root, pins)), 1)
        (self.root / relative).unlink()
        self.assertEqual(len(pin_failures(self.root, pins)), 1)

    def test_adapter_create_elsewhere_fails_even_in_a_test_file(self):
        self.write(SERVER / "tests.rs", "adapter.create(job);\n")
        self.assertEqual(len(adapter_create_failures(self.root)), 1)

    def test_slurm_submit_elsewhere_fails_outside_tests_only(self):
        self.write(CRATES / "worker/src/slurm/tests.rs", "Request::submit(intent);\n")
        self.write(CRATES / "worker/tests/flow.rs", "Request::submit(intent);\n")
        self.write(
            CRATES / "core/src/helper.rs",
            "fn real() {}\n#[cfg(test)]\nmod tests {\n    Request::submit(intent);\n}\n",
        )
        self.assertEqual(slurm_submit_failures(self.root), [])
        self.write(SERVER / "runs/apply.rs", "Request::submit(intent);\n")
        self.assertEqual(len(slurm_submit_failures(self.root)), 1)

    def test_code_after_an_inline_test_module_still_counts(self):
        self.write(
            CRATES / "core/src/helper.rs",
            "#[cfg(test)]\nmod tests {\n}\nfn real() {\n    Request::submit(intent);\n}\n",
        )
        self.assertEqual(len(slurm_submit_failures(self.root)), 1)

    def test_a_moved_slurm_submit_fails(self):
        (self.root / SLURM_SUBMIT_HOME).unlink()
        self.write(SERVER / "other.rs", "Request::submit(intent);\n")
        self.assertEqual(len(slurm_submit_failures(self.root)), 2)

    def test_a_new_or_removed_create_site_fails(self):
        self.write(SERVER / "apply.rs", "client\n    .create(&job)\n")
        failures = create_site_failures(self.root, self.known)
        self.assertEqual(len(failures), 1)
        self.assertIn("apply.rs", failures[0])
        (self.root / SERVER / "apply.rs").unlink()
        self.write(SERVER / "debug.rs", "// .create(&saved.recipe)\n")
        failures = create_site_failures(self.root, self.known)
        self.assertEqual(len(failures), 1)
        self.assertIn("gone", failures[0])

    def test_a_create_site_duplicated_in_one_file_fails(self):
        self.write(SERVER / "debug.rs", ".create(&saved.recipe)\n.create(&saved.recipe)\n")
        self.assertEqual(len(create_site_failures(self.root, self.known)), 1)

    def test_runs_module_may_not_import_an_effect(self):
        self.write(
            RUNS_MODULE / "apply.rs", "// Outbox is named only in this comment.\nuse crate::why;\n"
        )
        self.assertEqual(runs_module_failures(self.root), [])
        for line in [
            "use crate::backends::cloud::adapter::Adapter;",
            "use crate::backends::cloud::outbox;",
            "use crate::backends::slurm::transport::call;",
            "let runner: Runner<'_> = todo();",
        ]:
            self.write(RUNS_MODULE / "apply.rs", line + "\n")
            self.assertEqual(len(runs_module_failures(self.root)), 1, line)

    def test_a_backend_tree_reaching_the_other_fails(self):
        self.write(SERVER / "backends/slurm/service/http.rs", "// Owner::Cloud is a comment.\nfn f() {}\n")
        self.write(SERVER / "backends/cloud/service/http.rs", "fn f() {}\n")
        self.assertEqual(tree_crossing_failures(self.root), [])
        for line in [
            "Owner::Cloud(cloud) => cloud.tracking_status(),",
            "let cloud = api.cloud();",
            "let cloud = api.cloud_only()?;",
            "use crate::backends::cloud::service::ExternalExecutors;",
        ]:
            self.write(SERVER / "backends/slurm/service/http.rs", line + "\n")
            self.assertEqual(len(tree_crossing_failures(self.root)), 1, line)
        self.write(SERVER / "backends/slurm/service/http.rs", "fn f() {}\n")
        for line in [
            "Owner::Slurm(slurm) => slurm.tracking_status(),",
            "let slurm = api.submissions();",
            "let slurm = api.slurm_only()?;",
        ]:
            self.write(SERVER / "backends/cloud/service/http.rs", line + "\n")
            self.assertEqual(len(tree_crossing_failures(self.root)), 1, line)
        (self.root / SERVER / "backends/cloud/service/http.rs").unlink()
        (self.root / SERVER / "backends/cloud/service").rmdir()
        (self.root / SERVER / "backends/cloud/outbox.rs").unlink()
        (self.root / SERVER / "backends/cloud").rmdir()
        self.assertIn("has moved", tree_crossing_failures(self.root)[0])

    def test_a_service_naming_the_other_fails(self):
        self.write(SERVER / "backends/cloud/service/preparations.rs", "// submissions::Service\nuse crate::why;\n")
        self.write(SERVER / "backends/slurm/service/engine.rs", "pub trait RuntimeBuilder {}\n")
        self.assertEqual(service_crossing_failures(self.root), [])
        self.write(
            SERVER / "backends/cloud/service/preparations.rs", "slurm: Option<&crate::submissions::Service>,\n"
        )
        self.write(
            SERVER / "backends/slurm/service/engine.rs", "cloud: Arc<crate::cloud_runs::Service>,\n"
        )
        self.assertEqual(len(service_crossing_failures(self.root)), 2)
        self.write(
            SERVER / "backends/cloud/service/preparations.rs",
            "slurm: Option<&crate::backends::slurm::service::Service>,\n",
        )
        self.write(
            SERVER / "backends/slurm/service/engine.rs",
            "cloud: Arc<crate::backends::cloud::service::Service>,\n",
        )
        self.assertEqual(len(service_crossing_failures(self.root)), 2)
        (self.root / SERVER / "backends/cloud/service/preparations.rs").unlink()
        (self.root / SERVER / "backends/cloud/service").rmdir()
        self.assertIn("has moved", service_crossing_failures(self.root)[0])


if __name__ == "__main__":
    if "--print-pins" in sys.argv:
        for relative in PINS:
            print(f'    "{relative}": "{digest(ROOT, relative)}",')
        sys.exit(0)
    unittest.main()
