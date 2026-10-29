use ml_exp_client::{Client, datasets};
use ml_exp_core::{
    artifacts::FileReceipt,
    datasets::{Definition, State, oss::Import},
    multipart::PartRecord,
    slurm::helper::Target,
};
use serde_json::{Value, json};
use std::{
    io::{Read, Write},
    net::TcpListener,
    time::Duration,
};

#[test]
fn existing_object_import_preserves_content_identity_and_rejects_wrong_response() {
    for conflict in [false, true] {
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
                let mut b = [0];
                socket.read_exact(&mut b).unwrap();
                header.push(b[0]);
                assert!(header.len() < 16384);
            }
            let header = String::from_utf8(header).unwrap().to_lowercase();
            assert!(header.starts_with("post /api/projects/demo/datasets/pai-cpu/import-oss "));
            assert!(header.contains("authorization: bearer synthetic-token"));
            let size = header
                .lines()
                .find_map(|l| l.strip_prefix("content-length: "))
                .unwrap()
                .parse::<usize>()
                .unwrap();
            let mut body = vec![0; size];
            socket.read_exact(&mut body).unwrap();
            assert_eq!(serde_json::from_slice::<Import>(&body).unwrap(), expected);
            let mut state = State::new(
                "demo".into(),
                Target {
                    backend_id: "pai-cpu".into(),
                    profile_sha256: format!("sha256:{}", "c".repeat(64)),
                    storage_scope: format!("sha256:{}", "d".repeat(64)),
                },
                Definition {
                    sha256: if conflict {
                        "b".repeat(64)
                    } else {
                        expected.sha256
                    },
                    archive_bytes: 2048,
                    files: vec![FileReceipt {
                        path: "train.bin".into(),
                        bytes: 4,
                        sha256: "e".repeat(64),
                    }],
                },
            )
            .unwrap();
            state = state
                .record_part(
                    0,
                    PartRecord {
                        sha256: "f".repeat(64),
                        bytes: 2048,
                    },
                )
                .unwrap();
            let mut value = json!(state);
            value["archive_verified"] = json!(true);
            value["import_source"] = json!(null);
            let body = serde_json::to_vec(&value).unwrap();
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
            socket.write_all(&body).unwrap();
        });
        let client = Client::new(&format!("http://{address}"), "synthetic-token".into()).unwrap();
        let health: Value =
            json!({"capabilities":["native-datasets.v1","native-oss-dataset-import.v1"]});
        let result = datasets::import_oss(&client, "demo", "pai-cpu", source.clone(), &health);
        if conflict {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap()["asset_id"], source.asset_id());
        }
        worker.join().unwrap();
    }
}
