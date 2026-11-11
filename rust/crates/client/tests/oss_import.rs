use ml_exp_client::{Client, datasets};
use ml_exp_core::datasets::oss::Import;
use serde_json::json;
use std::{
    io::{Read, Write},
    net::TcpListener,
    time::Duration,
};

#[test]
fn queued_imports_are_successful_responses_and_wrong_identity_or_progress_is_refused() {
    for (status, copied, conflict, accepted) in [
        ("QUEUED", 0, false, true),
        ("COPYING_VERIFYING", 0, false, true),
        ("FAILED", 0, false, true),
        ("ARCHIVED", 1, false, true),
        ("ARCHIVED", 0, false, false),
        ("QUEUED", 2, false, false),
        ("QUEUED", 0, true, false),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let source = Import {
            object_key: "users/alice/datasets/train.tar".into(),
            sha256: "a".repeat(64),
            archive_bytes: 2048,
        };
        let expected = source.clone();
        let worker = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut header = Vec::new();
            while !header.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                socket.read_exact(&mut byte).unwrap();
                header.push(byte[0]);
                assert!(header.len() < 16384);
            }
            let header = String::from_utf8(header).unwrap().to_lowercase();
            assert!(header.starts_with("post /api/projects/demo/datasets/pai-cpu/import-oss "));
            let size = header
                .lines()
                .find_map(|l| l.strip_prefix("content-length: "))
                .unwrap()
                .parse::<usize>()
                .unwrap();
            let mut body = vec![0; size];
            socket.read_exact(&mut body).unwrap();
            assert_eq!(serde_json::from_slice::<Import>(&body).unwrap(), expected);
            let value = json!({"contract":"oss-import.v2","project":if conflict{"other"}else{"demo"},"executor":"pai-cpu","asset_id":expected.asset_id(),"import_source":expected,"status":status,"copied_parts":copied,"total_parts":1,"archive_verified":status=="ARCHIVED"});
            let body = serde_json::to_vec(&value).unwrap();
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
            socket.write_all(&body).unwrap();
        });
        let client = Client::new(&format!("http://{address}"), "synthetic-token".into()).unwrap();
        let health = json!({"capabilities":["native-datasets.v1","native-oss-dataset-import.v2"]});
        assert_eq!(
            datasets::import_oss(&client, "demo", "pai-cpu", source, &health).is_ok(),
            accepted
        );
        worker.join().unwrap();
    }
}
#[test]
fn old_synchronous_protocol_is_refused_without_sending_a_request() {
    let client = Client::new("http://127.0.0.1:1", "synthetic-token".into()).unwrap();
    let source = Import {
        object_key: "users/alice/datasets/train.tar".into(),
        sha256: "a".repeat(64),
        archive_bytes: 2048,
    };
    let health = json!({"capabilities":["native-datasets.v1","native-oss-dataset-import.v1"]});
    let error = datasets::import_oss(&client, "demo", "pai-cpu", source, &health).unwrap_err();
    assert!(error.to_string().contains("requires the upgraded client"));
}
