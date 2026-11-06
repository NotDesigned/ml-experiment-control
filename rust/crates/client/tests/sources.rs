//! Actual temporary source trees and finite loopback uploads. No repository
//! programs, provider calls, production state, or credentials are used.
use ml_exp_client::{Client, Error, sources};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};

fn health() -> Value {
    json!({"capabilities":["source-archive.v1"]})
}
fn result(digest: &str) -> Value {
    json!({"project":"demo","source_id":format!("source.{}","b".repeat(64)),
        "tree_digest":format!("sha256:{}","b".repeat(64)),
        "observation":{"kind":"archive","sha256":digest},
        "registration":{"existing":true}})
}
fn metadata() -> Value {
    let mut value = result(&"a".repeat(64));
    value.as_object_mut().unwrap().remove("registration");
    value["schema_version"] = json!(1);
    value["tree"] = json!("tree");
    value["created_at"] = json!("2026-10-27T00:00:00+00:00");
    value
}
fn payload() -> sources::PreparedArchive {
    let mut file = tempfile::tempfile().unwrap();
    // Multiple chunks and a partial final chunk exercise fixed-length streaming.
    for block in 0..17 {
        file.write_all(&vec![block; 64 * 1024]).unwrap();
    }
    file.write_all(b"final-partial-chunk").unwrap();
    sources::inspect_archive(file).unwrap()
}
fn accept(listener: &TcpListener) -> TcpStream {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        match listener.accept() {
            Ok((stream, _)) => return stream,
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(error) => panic!("expected source request was not received: {error}"),
        }
    }
}
fn request(stream: &mut TcpStream) -> (String, u64) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    while !bytes.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        stream.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
        assert!(bytes.len() < 16 * 1024);
    }
    let head = String::from_utf8(bytes).unwrap().to_lowercase();
    assert_eq!(
        head.matches("authorization: bearer source-test-secret\r\n")
            .count(),
        1
    );
    let length = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length: "))
        .map_or(0, |value| value.parse().unwrap());
    (head, length)
}
fn receive(stream: &mut TcpStream, length: u64) -> String {
    let mut reader = stream.take(length);
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    let mut bytes = 0;
    loop {
        let length = reader.read(&mut buffer).unwrap();
        if length == 0 {
            break;
        }
        digest.update(&buffer[..length]);
        bytes += length as u64;
    }
    assert_eq!(bytes, length);
    format!("{:x}", digest.finalize())
}
fn reply(stream: &mut TcpStream, status: &str, body: &[u8]) {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .unwrap();
    stream.write_all(body).unwrap();
}
fn no_repeat(listener: &TcpListener) {
    listener.set_nonblocking(true).unwrap();
    let deadline = Instant::now() + Duration::from_millis(100);
    loop {
        match listener.accept() {
            Ok(_) => panic!("source request was automatically replayed"),
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return,
            Err(error) => panic!("listener failed: {error}"),
        }
    }
}

#[test]
fn pure_identity_rules_keep_archive_sha_separate_from_the_source_tree() {
    let digest = "a".repeat(64);
    let plan = sources::plan_archive("demo", &digest, 10).unwrap();
    assert_eq!(plan.archive_sha256(), digest);
    assert_eq!(plan.bytes(), 10);
    assert_eq!(
        sources::validate_result(&plan, result(&digest)).unwrap()["project"],
        "demo"
    );
    assert!(
        sources::validate_metadata("demo", &format!("source.{}", "b".repeat(64)), metadata())
            .is_ok()
    );
    for (project, digest, bytes) in [
        ("../outside", "a".repeat(64), 10),
        ("demo", "A".repeat(64), 10),
        ("demo", "a".repeat(63), 10),
        ("demo", "a".repeat(64), 0),
        (
            "demo",
            "a".repeat(64),
            ml_exp_core::sources::Policy::default().max_archive_bytes + 1,
        ),
    ] {
        assert!(sources::plan_archive(project, &digest, bytes).is_err());
    }
    for (pointer, value) in [
        ("/project", json!("foreign")),
        ("/source_id", json!(format!("source.{}", "c".repeat(64)))),
        ("/tree_digest", json!(format!("sha256:{}", "c".repeat(64)))),
        ("/observation/kind", json!("git")),
        ("/observation/sha256", json!("c".repeat(64))),
        ("/registration/existing", json!(false)),
        ("/registration/existing", json!(1)),
    ] {
        let mut bad = result(&digest);
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(matches!(
            sources::validate_result(&plan, bad),
            Err(Error::Response)
        ));
    }
    for (pointer, value) in [
        ("/project", json!("foreign")),
        ("/schema_version", json!(true)),
        ("/schema_version", json!(1.0)),
        ("/tree", json!("../other")),
        ("/observation/sha256", json!("not-a-sha")),
    ] {
        let mut bad = metadata();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            sources::validate_metadata("demo", &format!("source.{}", "b".repeat(64)), bad).is_err()
        );
    }
}

#[test]
fn uploads_real_file_once_with_exact_length_digest_prefix_auth_and_protocol() {
    let archive = payload();
    let plan = sources::plan_archive("demo", &archive.sha256, archive.bytes).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
    let digest = archive.sha256.clone();
    let length = archive.bytes;
    let expected = result(&digest);
    let response = expected.clone();
    let thread = std::thread::spawn(move || {
        let mut stream = accept(&listener);
        let (head, actual) = request(&mut stream);
        assert!(head.starts_with(&format!(
            "post /ml-expd/api/source-imports/archive?project=demo&sha256={digest} http/1.1\r\n"
        )));
        assert_eq!(actual, length);
        assert!(!head.contains("transfer-encoding:"));
        assert_eq!(receive(&mut stream, length), digest);
        reply(
            &mut stream,
            "200 OK",
            &serde_json::to_vec(&response).unwrap(),
        );
        drop(stream);
        no_repeat(&listener);
    });
    let client = Client::new(&base, "source-test-secret".into()).unwrap();
    assert_eq!(
        sources::upload(&client, &plan, archive.file, &health()).unwrap(),
        expected
    );
    thread.join().unwrap();
}

#[test]
fn lost_malformed_and_foreign_publication_responses_remain_unknown_without_replay() {
    for case in [
        "lost",
        "invalid-json",
        "foreign",
        "server-error",
        "redirect",
    ] {
        let archive = payload();
        let plan = sources::plan_archive("demo", &archive.sha256, archive.bytes).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let length = archive.bytes;
        let digest = archive.sha256.clone();
        let thread = std::thread::spawn(move || {
            let mut stream = accept(&listener);
            let (_, actual) = request(&mut stream);
            assert_eq!(actual, length);
            assert_eq!(receive(&mut stream, length), digest);
            match case {
                "lost" => {}
                "invalid-json" => reply(&mut stream, "200 OK", b"sensitive-not-json"),
                "foreign" => {
                    let mut response = result(&digest);
                    response["project"] = json!("foreign");
                    reply(&mut stream, "200 OK", &serde_json::to_vec(&response).unwrap());
                }
                "server-error" => reply(&mut stream, "503 Unavailable", br#"{"detail":"sensitive"}"#),
                "redirect" => stream.write_all(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/token-leak\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap(),
                _ => unreachable!(),
            }
            drop(stream);
            no_repeat(&listener);
        });
        let client = Client::new(&base, "source-test-secret".into()).unwrap();
        let error = sources::upload(&client, &plan, archive.file, &health()).unwrap_err();
        if case == "redirect" {
            assert!(matches!(error, Error::Http(value) if value.status == 302));
        } else {
            assert!(
                matches!(error, Error::SourceOutcomeUnknown(_)),
                "{case}: {error}"
            );
        }
        thread.join().unwrap();
    }
}

#[test]
fn exact_metadata_get_preserves_source_binding_and_never_retries() {
    for foreign in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let source_id = format!("source.{}", "b".repeat(64));
        let expected_path = source_id.clone();
        let thread = std::thread::spawn(move || {
            let mut stream = accept(&listener);
            let (head, length) = request(&mut stream);
            assert_eq!(length, 0);
            assert!(head.starts_with(&format!(
                "get /ml-expd/api/projects/demo/sources/{expected_path} http/1.1\r\n"
            )));
            let mut response = metadata();
            if foreign {
                response["source_id"] = json!(format!("source.{}", "c".repeat(64)));
                response["tree_digest"] = json!(format!("sha256:{}", "c".repeat(64)));
            }
            reply(
                &mut stream,
                "200 OK",
                &serde_json::to_vec(&response).unwrap(),
            );
            drop(stream);
            no_repeat(&listener);
        });
        let client = Client::new(&base, "source-test-secret".into()).unwrap();
        let value = sources::read(&client, "demo", &source_id, &health());
        assert_eq!(value.is_err(), foreign);
        thread.join().unwrap();
    }
}

#[test]
fn lost_reply_lookup_uses_original_archive_identity_without_reupload_or_absence_claim() {
    for case in ["found", "foreign-sha", "missing"] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/ml-expd", listener.local_addr().unwrap());
        let digest = "a".repeat(64);
        let expected = digest.clone();
        let thread = std::thread::spawn(move || {
            let mut stream = accept(&listener);
            let (head, length) = request(&mut stream);
            assert_eq!(length, 0);
            assert!(head.starts_with(&format!(
                "get /ml-expd/api/projects/demo/source-imports/{expected} http/1.1\r\n"
            )));
            if case == "missing" {
                reply(&mut stream, "404 Not Found", br#"{"code":"SOURCE_RECEIPT_UNOBSERVED","detail":"No receipt observed; publication absence is not established"}"#);
            } else {
                let mut response = result(&expected);
                if case == "foreign-sha" {
                    response["observation"]["sha256"] = json!("c".repeat(64));
                }
                reply(
                    &mut stream,
                    "200 OK",
                    &serde_json::to_vec(&response).unwrap(),
                );
            }
            drop(stream);
            no_repeat(&listener);
        });
        let client = Client::new(&base, "source-test-secret".into()).unwrap();
        let value = sources::lookup_archive(&client, "demo", &digest, &health());
        match case {
            "found" => assert_eq!(value.unwrap(), result(&digest)),
            "foreign-sha" => assert!(matches!(value, Err(Error::Response))),
            "missing" => assert!(matches!(value, Err(Error::Http(error)) if error.status == 404)),
            _ => unreachable!(),
        }
        thread.join().unwrap();
    }
}

#[test]
fn invalid_health_or_changed_local_archive_never_sends_upload() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let client = Client::new(&base, "source-test-secret".into()).unwrap();
    for health in [json!({"capabilities":["source-import.v1"]}), json!({})] {
        let archive = payload();
        let plan = sources::plan_archive("demo", &archive.sha256, archive.bytes).unwrap();
        assert!(sources::upload(&client, &plan, archive.file, &health).is_err());
    }
    let mut archive = payload();
    let plan = sources::plan_archive("demo", &archive.sha256, archive.bytes).unwrap();
    archive.file.write_all(b"changed").unwrap();
    assert!(sources::upload(&client, &plan, archive.file, &health()).is_err());
    no_repeat(&listener);
}

#[cfg(unix)]
#[test]
fn group_writable_directories_in_a_users_tree_pack_the_same_as_private_ones() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let nested = temp.path().join("nested");
    std::fs::create_dir(&nested).unwrap();
    std::fs::write(nested.join("train.py"), "print('not executed')\n").unwrap();
    for mode in [0o775, 0o777] {
        std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(mode)).unwrap();
        std::fs::set_permissions(&nested, std::fs::Permissions::from_mode(mode)).unwrap();
        let writable = sources::pack_directory(temp.path()).unwrap();
        std::fs::set_permissions(temp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::set_permissions(&nested, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(
            writable.sha256,
            sources::pack_directory(temp.path()).unwrap().sha256
        );
    }
}

#[cfg(unix)]
#[test]
fn packs_deterministic_regular_files_and_preserves_only_executable_intent() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("nested")).unwrap();
    std::fs::create_dir(temp.path().join("empty")).unwrap();
    std::fs::write(
        temp.path().join("nested/train.py"),
        "print('not executed')\n",
    )
    .unwrap();
    std::fs::write(temp.path().join("run.sh"), "#!/bin/sh\nexit 99\n").unwrap();
    std::fs::set_permissions(
        temp.path().join("run.sh"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    for skip in [".git", ".venv", "__pycache__"] {
        std::fs::create_dir(temp.path().join(skip)).unwrap();
        std::fs::write(temp.path().join(skip).join(".env"), "not uploaded").unwrap();
    }
    let archive = sources::pack_directory(temp.path()).unwrap();
    let repeated = sources::pack_directory(temp.path()).unwrap();
    assert_eq!(archive.sha256, repeated.sha256);
    assert_eq!(archive.bytes, repeated.bytes);
    let decoder = flate2::read::GzDecoder::new(archive.file);
    let mut archive = tar::Archive::new(decoder);
    let mut entries = Vec::new();
    for entry in archive.entries().unwrap() {
        let mut entry = entry.unwrap();
        let path = entry.path().unwrap().to_string_lossy().into_owned();
        assert!(entry.header().entry_type().is_file());
        assert_eq!(entry.header().uid().unwrap(), 0);
        assert_eq!(entry.header().gid().unwrap(), 0);
        assert_eq!(entry.header().mtime().unwrap(), 0);
        assert_eq!(
            entry.header().mode().unwrap(),
            if path == "run.sh" { 0o500 } else { 0o400 }
        );
        let mut text = String::new();
        entry.read_to_string(&mut text).unwrap();
        entries.push((path, text));
    }
    assert_eq!(
        entries,
        vec![
            ("nested/train.py".into(), "print('not executed')\n".into()),
            ("run.sh".into(), "#!/bin/sh\nexit 99\n".into())
        ]
    );
    assert!(!temp.path().join("executed").exists());
}

#[cfg(unix)]
#[test]
fn source_filename_choices_are_allowed_but_aliases_and_size_limits_remain() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    for case in [
        "secret",
        "symlink",
        "hardlink",
        "fifo",
        "writable",
        "nested-secret",
        "empty",
        "expanded",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("private"), "DO_NOT_LEAK_VALUE").unwrap();
        match case {
            "secret" => {
                std::fs::write(temp.path().join(".env.testing"), "DO_NOT_LEAK_VALUE").unwrap()
            }
            "symlink" => symlink(
                outside.path().join("private"),
                temp.path().join("linked.py"),
            )
            .unwrap(),
            "hardlink" => std::fs::hard_link(
                outside.path().join("private"),
                temp.path().join("linked.py"),
            )
            .unwrap(),
            "fifo" => {
                assert!(
                    std::process::Command::new("mkfifo")
                        .arg(temp.path().join("fifo"))
                        .status()
                        .unwrap()
                        .success()
                );
            }
            "writable" => {
                std::fs::write(temp.path().join("train.py"), "data").unwrap();
                std::fs::set_permissions(
                    temp.path().join("train.py"),
                    std::fs::Permissions::from_mode(0o666),
                )
                .unwrap();
            }
            "nested-secret" => {
                std::fs::create_dir(temp.path().join("nested")).unwrap();
                std::fs::write(
                    temp.path().join("nested/credentials.json"),
                    "DO_NOT_LEAK_VALUE",
                )
                .unwrap();
            }
            "empty" => {}
            "expanded" => {
                File::create(temp.path().join("large.bin"))
                    .unwrap()
                    .set_len(ml_exp_core::sources::Policy::default().max_source_bytes + 1)
                    .unwrap();
            }
            _ => unreachable!(),
        }
        // A hard link or a group-writable file in the user's own tree packs as
        // ordinary hashed content, with its mode normalized in the archive.
        if matches!(case, "secret" | "nested-secret" | "hardlink" | "writable") {
            assert!(sources::pack_directory(temp.path()).is_ok(), "{case}");
            continue;
        }
        let error = sources::pack_directory(temp.path()).err().expect(case);
        assert!(!error.to_string().contains("DO_NOT_LEAK_VALUE"));
    }
    let temp = tempfile::tempdir().unwrap();
    let real = temp.path().join("real");
    std::fs::create_dir(&real).unwrap();
    std::fs::write(real.join("train.py"), "data").unwrap();
    symlink(&real, temp.path().join("alias")).unwrap();
    assert!(sources::pack_directory(&temp.path().join("alias")).is_err());
    let huge = tempfile::tempfile().unwrap();
    huge.set_len(ml_exp_core::sources::Policy::default().max_archive_bytes + 1)
        .unwrap();
    assert!(sources::inspect_archive(huge).is_err());
}

#[test]
fn prepared_archive_hashes_from_zero_and_restores_stream_position() {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(b"abcdef").unwrap();
    file.seek(SeekFrom::Start(3)).unwrap();
    let mut prepared = sources::inspect_archive(file).unwrap();
    assert_eq!(prepared.bytes, 6);
    assert_eq!(prepared.sha256, format!("{:x}", Sha256::digest(b"abcdef")));
    assert_eq!(prepared.file.stream_position().unwrap(), 0);
    assert!(sources::inspect_archive(tempfile::tempfile().unwrap()).is_err());
}

fn git(directory: &std::path::Path, arguments: &[&str]) {
    let status = std::process::Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(arguments)
        .status()
        .unwrap();
    assert!(status.success());
}
fn members(archive: sources::PreparedArchive) -> Vec<String> {
    let mut archive = tar::Archive::new(flate2::read::GzDecoder::new(archive.file));
    let mut names: Vec<String> = archive
        .entries()
        .unwrap()
        .map(|entry| entry.unwrap().path().unwrap().display().to_string())
        .collect();
    names.sort();
    names
}
#[test]
fn a_git_repository_packs_what_git_would_keep() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    git(root, &["init", "-q"]);
    std::fs::write(root.join(".gitignore"), "data/\n*.log\n").unwrap();
    std::fs::write(root.join("train.py"), "tracked").unwrap();
    std::fs::write(root.join("notes.txt"), "untracked").unwrap();
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("sub/keep.py"), "kept").unwrap();
    std::fs::write(root.join("sub/skip.log"), "ignored").unwrap();
    std::fs::create_dir_all(root.join("data")).unwrap();
    File::create(root.join("data/large.bin"))
        .unwrap()
        .set_len(ml_exp_core::sources::Policy::default().max_source_bytes + 1)
        .unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("nowhere", root.join("data/link")).unwrap();
    git(root, &["add", "train.py"]);
    let first = sources::pack_directory(root).unwrap();
    let sha256 = first.sha256.clone();
    assert_eq!(
        members(first),
        [".gitignore", "notes.txt", "sub/keep.py", "train.py"]
    );
    std::fs::write(root.join("sub/skip.log"), "ignored and changed").unwrap();
    assert_eq!(sources::pack_directory(root).unwrap().sha256, sha256);
    std::fs::write(root.join("notes.txt"), "changed").unwrap();
    assert_ne!(sources::pack_directory(root).unwrap().sha256, sha256);
}
#[test]
fn a_subdirectory_of_a_repository_packs_only_itself() {
    let temp = tempfile::tempdir().unwrap();
    git(temp.path(), &["init", "-q"]);
    std::fs::write(temp.path().join("outside.py"), "other").unwrap();
    let inner = temp.path().join("project");
    std::fs::create_dir(&inner).unwrap();
    std::fs::write(inner.join("train.py"), "inside").unwrap();
    assert_eq!(
        members(sources::pack_directory(&inner).unwrap()),
        ["train.py"]
    );
}
#[test]
fn a_refusal_names_the_file_and_how_to_leave_it_out() {
    let temp = tempfile::tempdir().unwrap();
    File::create(temp.path().join("large.bin"))
        .unwrap()
        .set_len(ml_exp_core::sources::Policy::default().max_source_bytes + 1)
        .unwrap();
    let text = sources::pack_directory(temp.path())
        .err()
        .unwrap()
        .to_string();
    assert!(
        text.contains("large.bin") && text.contains(".gitignore"),
        "{text}"
    );
}
#[cfg(unix)]
#[test]
fn a_kept_symlink_is_named_rather_than_failing_the_whole_directory_silently() {
    let temp = tempfile::tempdir().unwrap();
    git(temp.path(), &["init", "-q"]);
    std::fs::write(temp.path().join("train.py"), "code").unwrap();
    std::os::unix::fs::symlink("train.py", temp.path().join("latest")).unwrap();
    let text = sources::pack_directory(temp.path())
        .err()
        .unwrap()
        .to_string();
    assert!(
        text.contains("latest") && text.contains("symlink"),
        "{text}"
    );
}

#[test]
fn a_nested_repository_is_kept_by_its_own_rules() {
    let temp = tempfile::tempdir().unwrap();
    git(temp.path(), &["init", "-q"]);
    std::fs::write(temp.path().join("train.py"), "code").unwrap();
    let nested = temp.path().join("vendor");
    std::fs::create_dir(&nested).unwrap();
    git(&nested, &["init", "-q"]);
    std::fs::write(nested.join("lib.py"), "dependency").unwrap();
    std::fs::write(nested.join(".gitignore"), "secret.env\n").unwrap();
    std::fs::write(nested.join("secret.env"), "SECRET").unwrap();
    assert_eq!(
        members(sources::pack_directory(temp.path()).unwrap()),
        ["train.py", "vendor/.gitignore", "vendor/lib.py"]
    );
}
#[test]
fn a_repository_git_cannot_read_uploads_nothing() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join(".git"), "not a repository").unwrap();
    std::fs::write(temp.path().join(".env"), "SECRET").unwrap();
    std::fs::write(temp.path().join("train.py"), "code").unwrap();
    let text = sources::pack_directory(temp.path())
        .err()
        .unwrap()
        .to_string();
    assert!(text.contains("git could not list"), "{text}");
}
#[test]
fn the_content_digest_ignores_how_the_archive_is_compressed_but_not_what_is_in_it() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("train.py"), "code").unwrap();
    let first = sources::pack_directory(temp.path()).unwrap();
    assert_ne!(first.content_sha256, first.sha256);
    assert_eq!(
        sources::pack_directory(temp.path()).unwrap().content_sha256,
        first.content_sha256
    );
    std::fs::write(temp.path().join("train.py"), "changed").unwrap();
    assert_ne!(
        sources::pack_directory(temp.path()).unwrap().content_sha256,
        first.content_sha256
    );
}
#[test]
fn a_download_of_results_is_not_part_of_the_source() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("train.py"), "code").unwrap();
    let first = sources::pack_directory(temp.path()).unwrap();
    assert_eq!(first.paths, ["train.py"]);
    let sha256 = first.sha256.clone();
    std::fs::write(
        temp.path()
            .join(format!("run-1{}", sources::RESULTS_SUFFIX)),
        "results",
    )
    .unwrap();
    assert_eq!(sources::pack_directory(temp.path()).unwrap().sha256, sha256);
}
