import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oss_verify", ROOT / "rust/crates/server/src/backends/cloud/service/engine/oss_verify.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def archive(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as output:
        for name, content, kind in entries:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.size = len(content) if kind == tarfile.REGTYPE else 0
            entry.linkname = "outside" if kind == tarfile.SYMTYPE else ""
            output.addfile(entry, io.BytesIO(content) if kind == tarfile.REGTYPE else None)
    return stream.getvalue()


def inputs(data):
    config = {"sha256": hashlib.sha256(data).hexdigest(), "archive_bytes": len(data),
              "part_bytes": 1024, "part_count": (len(data) + 1023) // 1024}
    calls = []
    def fetch(number):
        calls.append(number)
        return data[number * 1024:(number + 1) * 1024]
    return config, fetch, calls


class Verification(unittest.TestCase):
    def test_streams_each_destination_once_and_hashes_files_across_part_boundaries(self):
        body = b"model" * 1000
        data = archive([("weights/model.bin", body, tarfile.REGTYPE), ("empty", b"", tarfile.REGTYPE)])
        config, fetch, calls = inputs(data)
        report = VERIFY.verify(config, fetch)
        self.assertEqual(calls, list(range(config["part_count"])))
        self.assertEqual(report["definition"]["sha256"], config["sha256"])
        self.assertEqual(report["definition"]["files"][1]["sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(len(report["parts"]), config["part_count"])
        self.assertEqual(report["parts"]["0"]["sha256"], hashlib.sha256(data[:1024]).hexdigest())

    def test_hash_includes_every_byte_after_tar_end_marker(self):
        data = archive([("file", b"ok", tarfile.REGTYPE)]) + b"trailing bytes"
        config, fetch, calls = inputs(data)
        VERIFY.verify(config, fetch)
        self.assertEqual(len(calls), config["part_count"])
        config["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ARCHIVE_SHA256_MISMATCH"):
            VERIFY.verify(config, fetch)

    def test_rejects_duplicate_links_directories_parent_and_absolute_paths(self):
        for entries in [
            [("file", b"a", tarfile.REGTYPE), ("file", b"b", tarfile.REGTYPE)],
            [("link", b"", tarfile.SYMTYPE)], [("directory", b"", tarfile.DIRTYPE)],
            [("../escape", b"bad", tarfile.REGTYPE)], [("/escape", b"bad", tarfile.REGTYPE)],
            [("a", b"file", tarfile.REGTYPE), ("a/b", b"other", tarfile.REGTYPE)],
        ]:
            with self.subTest(entries=entries):
                config, fetch, _ = inputs(archive(entries))
                with self.assertRaises(ValueError):
                    VERIFY.verify(config, fetch)

    def test_pax_long_unicode_paths_remain_exact(self):
        path = "训练/" + "long" * 40 + ".bin"
        config, fetch, _ = inputs(archive([(path, b"content", tarfile.REGTYPE)]))
        self.assertEqual(VERIFY.verify(config, fetch)["definition"]["files"][0]["path"], path)

    def test_truncated_destination_never_produces_a_report(self):
        config, fetch, _ = inputs(archive([("file", b"ok", tarfile.REGTYPE)]))
        with self.assertRaisesRegex(ValueError, "PART_SIZE_MISMATCH"):
            VERIFY.verify(config, lambda n: fetch(n)[:-1])

    def test_empty_archive_is_refused(self):
        config, fetch, _ = inputs(archive([]))
        with self.assertRaises(ValueError):
            VERIFY.verify(config, fetch)


if __name__ == "__main__":
    unittest.main()
