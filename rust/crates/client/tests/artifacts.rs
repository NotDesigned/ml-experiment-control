//! Real HTTP transfer failures retain the local archive and never imply an ACK.
use ml_exp_client::{Client, artifacts};
use ml_exp_core::artifacts::{
    ArchiveReceipt,
    publication::{Bundle, CONTRACT, State},
};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    time::Duration,
};
const ACTION: &str = "action-0123456789abcdef";
fn health() -> Value {
    json!({"api_protocol_version":2,"min_client_protocol_version":2,"capabilities":["native-artifacts.v1"]})
}
fn read(socket: &mut TcpStream) -> (String, Value) {
    socket
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut h = Vec::new();
    while !h.ends_with(b"\r\n\r\n") {
        let mut b = [0];
        socket.read_exact(&mut b).unwrap();
        h.push(b[0]);
        assert!(h.len() < 16384);
    }
    let h = String::from_utf8(h).unwrap().to_lowercase();
    assert!(h.contains("authorization: bearer synthetic-token"));
    assert!(h.contains("x-ml-expd-client-protocol: 2"));
    let n = h
        .lines()
        .find_map(|l| l.strip_prefix("content-length: "))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let mut b = vec![0; n];
    socket.read_exact(&mut b).unwrap();
    (
        h,
        if n > 0 {
            serde_json::from_slice(&b).unwrap()
        } else {
            Value::Null
        },
    )
}
fn reply(socket: &mut TcpStream, state: &State) {
    let b=serde_json::to_vec(&json!({"submission_id":ACTION,"state":state,"download_available":state.available().unwrap(),"scheduler_mutation_performed":false})).unwrap();
    write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",b.len()).unwrap();
    socket.write_all(&b).unwrap();
}
fn fixture() -> (Vec<u8>, State) {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("weights.bin"), b"weights").unwrap();
    let directory = ml_exp_io::directory::Directory::open(root.path(), false).unwrap();
    let mut file = tempfile::tempfile().unwrap();
    let (sha256, bytes, files) =
        ml_exp_io::artifacts::pack(&directory, &mut file, &Default::default()).unwrap();
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut body = Vec::new();
    file.read_to_end(&mut body).unwrap();
    let archive = ArchiveReceipt {
        project: "demo".into(),
        run_id: "train".into(),
        attempt_id: "attempt-001".into(),
        object_key: format!("demo/train/attempt-001/{sha256}.tar"),
        sha256,
        bytes,
        files,
    };
    let state = State::new(Bundle {
        contract: CONTRACT.into(),
        intent_sha256: format!("sha256:{}", "1".repeat(64)),
        config_sha256: format!("sha256:{}", "2".repeat(64)),
        archive,
    })
    .unwrap();
    (body, state)
}
#[test]
fn selected_file_is_durable_only_after_whole_archive_verification_and_never_acks() {
    for corrupt in [false, true] {
        let (mut body, state) = fixture();
        if corrupt {
            *body.last_mut().unwrap() = 42;
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let expected = state.clone();
        let worker = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            assert!(
                read(&mut socket)
                    .0
                    .starts_with(&format!("get /api/submissions/{ACTION}/artifacts "))
            );
            reply(&mut socket, &expected);
            let (mut socket, _) = listener.accept().unwrap();
            assert!(read(&mut socket).0.starts_with(&format!(
                "get /api/submissions/{ACTION}/artifacts/{} ",
                expected.bundle.archive.sha256
            )));
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/x-tar\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
            socket.write_all(&body).unwrap();
            drop(socket);
            if !corrupt {
                let (mut socket, _) = listener.accept().unwrap();
                assert!(read(&mut socket).0.starts_with("get "));
                reply(&mut socket, &expected);
            }
            listener.set_nonblocking(true).unwrap();
            assert!(
                listener.accept().is_err(),
                "partial file download must never ACK or release"
            );
        });
        let client = Client::new(&format!("http://{addr}"), "synthetic-token".into()).unwrap();
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("model.bin");
        let result = artifacts::download_file(&client, ACTION, "weights.bin", &target, &health());
        if corrupt {
            assert!(result.is_err());
            assert!(!target.exists());
            assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        } else {
            assert_eq!(result.unwrap()["acknowledgement_sent"], false);
            assert_eq!(std::fs::read(&target).unwrap(), b"weights");
            assert_eq!(
                artifacts::download_file(&client, ACTION, "weights.bin", &target, &health())
                    .unwrap()["reused_existing_file"],
                true
            );
        }
        worker.join().unwrap();
    }
}
#[test]
fn streamed_download_verifies_then_acknowledges_once_and_reuses_saved_bytes_after_lost_ack() {
    let (body, state) = fixture();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let expected = state.clone();
    let worker = std::thread::spawn(move || {
        for pass in 0..2 {
            let (mut socket, _) = listener.accept().unwrap();
            let (h, _) = read(&mut socket);
            assert!(h.starts_with(&format!("get /api/submissions/{ACTION}/artifacts ")));
            reply(&mut socket, &expected);
            if pass == 0 {
                let (mut socket, _) = listener.accept().unwrap();
                let (h, _) = read(&mut socket);
                assert!(h.starts_with(&format!(
                    "get /api/submissions/{ACTION}/artifacts/{} ",
                    expected.bundle.archive.sha256
                )));
                write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/x-tar\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
                socket.write_all(&body).unwrap();
            }
            let (mut socket, _) = listener.accept().unwrap();
            let (h, b) = read(&mut socket);
            assert!(h.starts_with(&format!("post /api/submissions/{ACTION}/acknowledge ")));
            let proof = serde_json::from_value(b["proof"].clone()).unwrap();
            assert_eq!(b["release"], true);
            let after = expected
                .acknowledge(proof, "2026-10-29T00:00:00Z", true)
                .unwrap();
            if pass == 1 {
                reply(&mut socket, &after);
            }
        }
    });
    let client = Client::new(&format!("http://{addr}"), "synthetic-token".into()).unwrap();
    let output = tempfile::tempdir().unwrap();
    let target = output.path().join("results.tar");
    assert!(artifacts::download(&client, ACTION, &target, true, &health()).is_err());
    assert!(target.is_file());
    let result = artifacts::download(&client, ACTION, &target, true, &health()).unwrap();
    assert_eq!(result["reused_existing_archive"], true);
    assert_eq!(result["local_archive_verified"], true);
    worker.join().unwrap();
}
#[test]
fn corrupt_or_truncated_downloads_send_no_ack_and_leave_no_final_file() {
    for mode in ["corrupt", "truncated"] {
        let (mut body, state) = fixture();
        if mode == "corrupt" {
            body[600] ^= 42;
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            read(&mut socket);
            reply(&mut socket, &state);
            let (mut socket, _) = listener.accept().unwrap();
            read(&mut socket);
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/x-tar\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
            socket
                .write_all(if mode == "truncated" {
                    &body[..body.len() / 2]
                } else {
                    &body
                })
                .unwrap();
            drop(socket);
            listener.set_nonblocking(true).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            assert!(listener.accept().is_err());
        });
        let client = Client::new(&format!("http://{addr}"), "synthetic-token".into()).unwrap();
        let output = tempfile::tempdir().unwrap();
        let target = output.path().join("results.tar");
        assert!(artifacts::download(&client, ACTION, &target, true, &health()).is_err());
        assert!(!target.exists());
        worker.join().unwrap();
    }
}
