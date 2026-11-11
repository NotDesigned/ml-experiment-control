#!/usr/bin/env python3
"""Structural guard around the no-replay code. It reads files and compiles nothing.

It catches two kinds of drift: a sealed file changed, or a new way to reach a
provider create appeared. It does not parse reconcile bodies and it cannot prove
ordering. Behaviour tests remain the proof: `server/tests/cloud_no_replay.rs`,
`server/tests/cloud_runs.rs`, `backends/slurm/tests/submission_tests.rs` and
`backends/cloud/outbox/tests.rs`.

Pinned are only the files where a job is actually asked for and where the record
that forbids asking twice is kept, one pair per backend:

- `backends/cloud/outbox.rs` writes the durable `started` mark before a create and
  never creates again once it is set; `backends/cloud/adapter.rs` is the one place
  that sends CreateJob, and holds `find`, which may recover a handle but never
  permits a create. Its read-only diagnostics live in `adapter/diagnostics.rs`.
- `backends/slurm/action.rs` holds the one `Request::submit(`; the helper's
  `worker/src/slurm/ledger.rs` keeps the durable intent and runs sbatch once.

Everything else (Action stores, request compilation, cancellation records, the
tests themselves) is covered by the behaviour tests above, and by the scans below,
which find any new path to a create wherever it appears.

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
    "rust/crates/server/src/backends/cloud/outbox.rs": "98e5e6b2e3818085c2d3d78dc32856be8a90f48d1ae0da27c48d00fce6b34bc7",
    "rust/crates/server/src/backends/cloud/adapter.rs": "92bf028ab2b9441c2e760629cb3df0281df26eff78cbf14b70128b21e344362a",
    "rust/crates/server/src/backends/slurm/action.rs": "15ad4038d8525dde13d1439c97faa1475bc371e078603f3c8eb1ab0444b609ba",
    "rust/crates/worker/src/slurm/ledger.rs": "0ec0b45b7a7001c69fa29a13e8f8abbcc26c9db29b787c2650b09ba8b13afa75",
}

# Every line in the server crate's production code that calls a method named
# `create`, by file name and text. Compiling a PAI request is `create_request`
# and sends nothing, so it is not listed. A new site must be reviewed against the
# no-replay rules, then added.
KNOWN_CREATE_SITES = Counter(
    {
        ("outbox.rs", "match guard().and_then(|_| adapter.create(job, allocation, cancelled)) {"): 1,
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
