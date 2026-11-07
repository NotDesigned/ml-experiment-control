//! Code delivered apart from the image, through the real store, Action and server
//! executable. Credentials and images are synthetic; no provider call is made.
use ml_exp_core::{
    cloud::{action::Execution, pai, run::Runtime},
    first_submission::native::{Approval, attempt_slot, creation::Request},
    sources::{CodeManifest, MAX_CODE_BYTES, Policy, tree_source_id},
};
use ml_exp_server::{
    cloud_runs::{Service, Settings},
    read_config, sources,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, io::Cursor, os::unix::fs::PermissionsExt, sync::Arc};

const NOW: &str = "2026-10-29T00:00:00Z";

fn private(path: &std::path::Path, value: &Value) {
    ml_exp_io::write_atomic(path, &serde_json::to_vec(value).unwrap(), true).unwrap();
}
fn archive_of(files: &[(&str, &[u8])]) -> Vec<u8> {
    let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    let mut tar = tar::Builder::new(encoder);
    for (path, body) in files {
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o400);
        header.set_cksum();
        tar.append_data(&mut header, path, *body).unwrap();
    }
    tar.into_inner().unwrap().finish().unwrap()
}
struct World {
    temp: tempfile::TempDir,
    settings: read_config::Settings,
    sources: Arc<sources::Service>,
}
fn world() -> World {
    let temp = tempfile::tempdir().unwrap();
    fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let registry = temp.path().join("registry");
    read_config::initialize(&registry).unwrap();
    let project = temp.path().join("project.json");
    let runs = temp.path().join("runs");
    fs::create_dir(&runs).unwrap();
    private(
        &project,
        &json!({"schema_version":1,"project":"demo","title":"Fixture","run_roots":["runs"]}),
    );
    let settings = read_config::Settings {
        schema_version: 1,
        registry_root: registry,
        projects: vec![read_config::Project {
            project: "demo".into(),
            project_file: project,
            run_roots: vec![runs],
            runs: Vec::new(),
            discover_runs: None,
        }],
        source_policy: Some(Policy::default()),
    };
    sources::Service::initialize(&settings).unwrap();
    let sources = Arc::new(sources::Service::open(&settings, None).unwrap());
    World {
        temp,
        settings,
        sources,
    }
}
impl World {
    fn publish(&self, files: &[(&str, &[u8])]) -> String {
        let bytes = archive_of(files);
        self.sources
            .publish(
                "demo",
                &mut Cursor::new(&bytes),
                &ml_exp_core::slurm::helper::sha256_bytes(&bytes)[7..],
                NOW,
            )
            .unwrap()["source_id"]
            .as_str()
            .unwrap()
            .to_owned()
    }
    fn runs(&self) -> std::path::PathBuf {
        self.temp.path().join("runs")
    }
}
struct Server(std::process::Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
/// The client's local identity is what the real store computes for its archive, or
/// the question about reuse would be wrong without anyone noticing.
#[test]
fn the_environment_identity_the_client_computes_is_the_one_the_server_publishes() {
    let world = world();
    let directory = world.temp.path().join("repo");
    fs::create_dir_all(directory.join("deps/nested")).unwrap();
    fs::write(directory.join("Dockerfile"), "FROM x\n").unwrap();
    fs::write(directory.join("deps/nested/lock"), "a\n").unwrap();
    fs::write(directory.join("deps/run.sh"), vec![7; 3 * 1024 * 1024]).unwrap();
    fs::set_permissions(
        directory.join("deps/run.sh"),
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    fs::write(directory.join("train.py"), "code\n").unwrap();
    let files = vec![
        "Dockerfile".to_owned(),
        "deps/nested/lock".to_owned(),
        "deps/run.sh".to_owned(),
    ];
    let whole = ml_exp_client::sources::pack_directory(&directory).unwrap();
    let packed =
        ml_exp_client::sources::pack_subset(&whole, &files, ("Dockerfile", "FROM x\n")).unwrap();
    assert_eq!(packed.paths, files);
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut &packed.file, &mut bytes).unwrap();
    let published = world
        .sources
        .publish("demo", &mut Cursor::new(&bytes), &packed.sha256, NOW)
        .unwrap();
    assert_eq!(
        published["source_id"],
        ml_exp_client::sources::local_source_id(&directory, &files).unwrap()
    );
}
#[test]
fn an_attempt_reads_only_its_own_frozen_code_and_the_run_can_say_what_it_was_made_from() {
    let world = world();
    let environment = world.publish(&[("Dockerfile", &b"FROM x\n"[..])]);
    let code_id = world.publish(&[
        ("src/train.py", &b"print('code')\n"[..]),
        ("uv.lock", &b"x\n"[..]),
    ]);
    // In the same Project, and no request can name it.
    let other = world.publish(&[("other.py", &b"never served\n"[..])]);
    let runtime = Runtime {
        runtime_id: format!("runtime.{}", "1".repeat(64)),
        source_id: environment.clone(),
        image: format!("registry.example/team/code@sha256:{}", "2".repeat(64)),
        source_dir: "/workspace".into(),
        worker_path: "/opt/ml-expd/worker".into(),
        worker_sha256: format!("sha256:{}", "3".repeat(64)),
    };
    let credentials = world.temp.path().join("credentials.json");
    private(
        &credentials,
        &json!({"provider":"pai","access_key_id":"synthetic","access_key_secret":"never-sent"}),
    );
    let storage = pai::Storage {
        cpfs_uri: "bmcpfs://cpfs-example.us-southeast-1.cpfs.aliyuncs.com/".into(),
        workspace_user: "researcher".into(),
        role_arn: "acs:ram::12345:role/pai-data-researcher".into(),
    };
    let signing_key = world.temp.path().join("worker-signing-key");
    ml_exp_io::write_atomic(
        &signing_key,
        b"synthetic-exchange-key-0000000000000000000000",
        true,
    )
    .unwrap();
    let worker = runtime.worker_sha256.clone();
    let socket = world.temp.path().join("builder.sock");
    let peer = ml_exp_io::directory::effective_uid();
    let config = json!({"schema_version":1,"enabled":false,"allow_run_creation":true,
        "builds":{"socket":socket,"peer_uid":peer,"worker_sha256":worker},
        "exchange":{"public_url":"https://api.example/ml-expd","signing_key_file":signing_key},
        "backends":[{"executor":"pai-h200","profile":{"provider":"pai","region":"us-southeast-1",
            "workspace_id":"253","quota_id":"quota1","max_gpus":8,"max_cpus":200,"max_memory_gb":1600,
            "storage":storage},"run_root":"/mnt/workspace/ml-expd/runs","credentials_file":credentials,
            "registry_auth_file":null,"runtimes":{runtime.runtime_id.clone():runtime}}]});
    Service::initialize(&world.settings).unwrap();
    let service = Service::open(
        world.settings.clone(),
        Settings::from_slice(&serde_json::to_vec(&config).unwrap()).unwrap(),
        world.sources.clone(),
        None,
        Default::default(),
    )
    .unwrap();
    let request = |run: &str, code: Option<&str>| -> Request {
        let mut value = json!({"run_id":run,"runtime_id":runtime.runtime_id,"executor":"pai-h200",
            "command":["python","train.py"],"resources":{"gpus":1,"cpus":8,"memory_gb":64},
            "walltime_seconds":300,"seed":42});
        if let Some(code) = code {
            value["code_source_id"] = json!(code);
        }
        serde_json::from_value(value).unwrap()
    };
    let package = |run: &str, transition: &str| {
        service.create_run("demo", request(run, None), NOW).ok();
        service
            .prepare(
                "demo",
                run,
                Approval {
                    max_gpu_hours: 1.,
                    reason: String::new(),
                },
                "2026-10-29T00:01:00Z",
                "2026-10-29T01:01:00Z",
                &transition.repeat(32),
            )
            .unwrap();
        let slot = world
            .settings
            .registry_root
            .join("cloud-submissions/slots")
            .join(format!(
                "{}.json",
                attempt_slot("demo", run, "attempt-001").unwrap()
            ));
        let saved: Value = serde_json::from_slice(&fs::read(slot).unwrap()).unwrap();
        let execution: Execution = serde_json::from_value(saved["execution"].clone()).unwrap();
        execution.package().unwrap()
    };
    service
        .create_run("demo", request("code-run", Some(&code_id)), NOW)
        .unwrap();
    let code_package = package("code-run", "a");
    let plain_package = package("plain-run", "b");
    assert_eq!(
        code_package.code_source_id.as_deref(),
        Some(code_id.as_str())
    );
    assert!(
        code_package
            .manifest
            .workdir
            .ends_with("/code-run/attempt-001/code")
    );
    assert_eq!(plain_package.manifest.workdir, "/workspace");
    let frozen: Value =
        serde_json::from_slice(&fs::read(world.runs().join("code-run/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(frozen["code_source_id"], code_id.as_str());
    assert_eq!(frozen["source_id"], environment.as_str());

    // A code source that is absent, or above the limit, stops the Run being created.
    let absent = format!("source.{}", "f".repeat(64));
    assert!(
        service
            .create_run("demo", request("code-absent", Some(&absent)), NOW)
            .is_err()
    );
    assert!(!world.runs().join("code-absent").exists());
    let weights = vec![0; MAX_CODE_BYTES as usize + 1];
    let large = world.publish(&[("weights.bin", weights.as_slice())]);
    drop(weights);
    let refused = service
        .create_run("demo", request("code-large", Some(&large)), NOW)
        .unwrap_err();
    assert_eq!(refused.code, "CODE_SOURCE_TOO_LARGE");
    assert_eq!(refused.http_status, Some(422));
    assert!(!world.runs().join("code-large").exists());
    // A Runtime with another worker cannot be paired with code, and a Run already created
    // stays valid when the configured worker moves on.
    let mut upgraded = config.clone();
    upgraded["builds"]["worker_sha256"] = json!(format!("sha256:{}", "9".repeat(64)));
    let later = Service::open(
        world.settings.clone(),
        Settings::from_slice(&serde_json::to_vec(&upgraded).unwrap()).unwrap(),
        world.sources.clone(),
        None,
        Default::default(),
    )
    .unwrap();
    let refused = later
        .create_run("demo", request("code-old-worker", Some(&code_id)), NOW)
        .unwrap_err();
    assert_eq!(refused.code, "CODE_DELIVERY_WORKER_OUTDATED");
    assert!(!world.runs().join("code-old-worker").exists());
    assert_eq!(
        later
            .create_run("demo", request("code-run", Some(&code_id)), NOW)
            .unwrap()["reused"],
        true
    );
    drop(later);
    // An explicit Runtime saved in a preparation, and then the server is upgraded. One
    // whose Run exists goes on; one that would create a Run is refused, and says why.
    let saved = |id: &str, run: &str| -> ml_exp_core::preparation::Request {
        serde_json::from_value(
            json!({"request_id":id.repeat(32),"run":request(run, Some(&code_id)),
            "approval":{"max_gpu_hours":1.0}}),
        )
        .unwrap()
    };
    let continuing = service
        .experiment_prepare("demo", saved("c", "code-run"), None)
        .unwrap()["preparation_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let creating = service
        .experiment_prepare("demo", saved("d", "code-never"), None)
        .unwrap()["preparation_id"]
        .as_str()
        .unwrap()
        .to_owned();
    upgraded["enabled"] = json!(true);
    let later = Arc::new(
        Service::open(
            world.settings.clone(),
            Settings::from_slice(&serde_json::to_vec(&upgraded).unwrap()).unwrap(),
            world.sources.clone(),
            None,
            Default::default(),
        )
        .unwrap(),
    );
    let go_on = |id: &str| {
        later
            .experiment_continue(
                "demo",
                id,
                &format!("PREPARE {id}"),
                None,
                ml_exp_server::runs::Context {
                    now: "2026-10-29T00:02:00Z",
                    expires: "2026-10-29T01:02:00Z",
                    transition_id: &"f".repeat(32),
                },
            )
            .err()
            .map(|error| error.code)
    };
    assert_eq!(go_on(&continuing), None);
    assert_eq!(
        go_on(&creating).as_deref(),
        Some("CODE_DELIVERY_WORKER_OUTDATED")
    );
    assert!(!world.runs().join("code-never").exists());
    drop(later);
    // Refused before any image is built for a Run a Slurm executor could never create.
    let on_slurm: ml_exp_core::preparation::Request = serde_json::from_value(json!({
        "request_id":"e".repeat(32),"run":{"run_id":"code-on-slurm",
        "runtime_id":format!("runtime.{}","1".repeat(64)),"executor":"slurm-test",
        "command":["train"],"resources":{"gpus":1,"cpus":8,"memory_gb":64},
        "walltime_seconds":300,"seed":42,"code_source_id":code_id},
        "approval":{"max_gpu_hours":1.0}}))
    .unwrap();
    let refused = service
        .experiment_prepare(
            "demo",
            on_slurm,
            Some(&BTreeMap::from([("slurm-test".to_owned(), true)])),
        )
        .unwrap_err();
    assert_eq!(refused.code, "CODE_DELIVERY_UNSUPPORTED");

    let temp = world.temp.path();
    let read_file = temp.join("reads.json");
    private(&read_file, &serde_json::to_value(&world.settings).unwrap());
    let cloud_file = temp.join("cloud.json");
    private(&cloud_file, &config);
    let token = "native-cloud-http-synthetic-token-000000000000";
    let token_file = temp.join("token");
    ml_exp_io::write_atomic(&token_file, token.as_bytes(), true).unwrap();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_ml-expd-rs"))
        .args(["--bind", "127.0.0.1:0", "--token-file"])
        .arg(&token_file)
        .arg("--read-config")
        .arg(&read_file)
        .arg("--source-archives")
        .arg("--cloud-config")
        .arg(&cloud_file)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let mut server = Server(child);
    let stdout = server.0.stdout.take().unwrap();
    let (sent, received) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let line = std::io::BufReader::new(stdout)
            .lines()
            .next()
            .unwrap()
            .unwrap();
        let _ = sent.send(line);
    });
    let address = received
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("server startup");

    // The real server tells a client what it needs, and the Run says what it is.
    let client = ml_exp_client::Client::new(&format!("http://{address}"), token.into()).unwrap();
    let health = client.health().unwrap();
    let status = |run: &str| {
        ml_exp_client::status::observe(
            &client,
            &health,
            None,
            ml_exp_client::observation::Selection {
                project: Some("demo"),
                run: Some(run),
                attempt: Some("attempt-001"),
            },
            false,
        )
        .unwrap()
    };
    let shown = status("code-run");
    assert_eq!(shown["identity"]["code_source_id"], code_id.as_str());
    assert_eq!(
        shown["identity"]["environment_source_id"],
        environment.as_str()
    );
    assert_eq!(
        shown["identity"]["runtime_id"],
        format!("runtime.{}", "1".repeat(64))
    );
    let text = ml_exp_client::presentation::status_summary(&shown).unwrap();
    assert!(text.contains(&code_id), "{text}");
    let plain = status("plain-run");
    assert!(plain["identity"].get("code_source_id").is_none());
    assert!(
        ml_exp_client::presentation::status_summary(&plain)
            .unwrap()
            .contains("code in the image")
    );

    // The token of this Attempt reaches exactly the code frozen in this Attempt.
    let http = reqwest::blocking::Client::new();
    let base = "/api/worker-exchange/demo/code-run/attempt-001";
    let code_token = code_package.channel.token.as_str();
    let plain_token = plain_package.channel.token.as_str();
    let read = |credential: Option<&str>, path: &str| {
        let request = http.get(format!("http://{address}{path}"));
        match credential {
            Some(credential) => request.bearer_auth(credential),
            None => request,
        }
        .send()
        .unwrap()
    };
    let listing: CodeManifest = serde_json::from_slice(
        &read(Some(code_token), &format!("{base}/code"))
            .bytes()
            .unwrap(),
    )
    .unwrap();
    let plan = listing.validate(&code_id).unwrap();
    assert_eq!(listing.directories, ["src"]);
    assert_eq!(
        listing
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>(),
        ["src/train.py", "uv.lock"]
    );
    assert_ne!(listing.source_id, other);
    let content = read(Some(code_token), &format!("{base}/code-content"));
    assert_eq!(content.status().as_u16(), 200);
    assert_eq!(content.content_length(), Some(listing.total_bytes()));
    let content = content.bytes().unwrap();
    assert_eq!(content.as_ref(), b"print('code')\nx\n");
    // What a worker computes from these bytes is what the store published.
    let mut offsets = BTreeMap::new();
    let mut at = 0;
    for file in &listing.files {
        offsets.insert(file.path.clone(), at);
        at += file.bytes as usize;
    }
    let found = tree_source_id(plan, |path, offset, len| {
        let start = offsets[path] + offset as usize;
        Ok(content[start..start + len].to_vec())
    })
    .unwrap();
    assert_eq!(found, code_id);
    for suffix in ["code", "code-content"] {
        let path = format!("{base}/{suffix}");
        for credential in [None, Some("wrong"), Some(plain_token), Some(token)] {
            assert_eq!(
                read(credential, &path).status().as_u16(),
                401,
                "{path} {credential:?}"
            );
        }
        // A real capability reaches no other Attempt.
        assert_eq!(
            read(
                Some(code_token),
                &format!("/api/worker-exchange/demo/code-run/attempt-002/{suffix}")
            )
            .status()
            .as_u16(),
            401
        );
        // A Run with no delivered code has nothing to serve, even to its own token.
        assert_eq!(
            read(
                Some(plain_token),
                &format!("/api/worker-exchange/demo/plain-run/attempt-001/{suffix}")
            )
            .status()
            .as_u16(),
            404
        );
    }
    // And no ordinary route.
    assert_eq!(read(Some(code_token), "/api/health").status().as_u16(), 401);
}

/// A builder that publishes whatever it is asked to, as the real one's reply says, and
/// counts the builds. What a real publisher does with the image only a real build proves.
fn fake_builder(socket: &std::path::Path) -> Arc<std::sync::Mutex<Vec<Value>>> {
    use std::io::{BufRead, Read, Write};
    let listener = std::os::unix::net::UnixListener::bind(socket).unwrap();
    let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = requests.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut reader = std::io::BufReader::new(stream.unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.strip_prefix("Content-Length: ") {
                    length = value.trim().parse::<usize>().unwrap();
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let request: Value = serde_json::from_slice(&body).unwrap();
            let reply = json!({"project":request["project"],"runtime_id":request["runtime_id"],
                "source_id":request["source_id"],"recipe":ml_exp_core::cloud::build::CONTRACT,
                "runtime":{"runtime_id":request["runtime_id"],"source_id":request["source_id"],
                    "image":request["base_image"],"source_dir":"/workspace",
                    "worker_path":"/usr/local/bin/ml-exp-worker-rs",
                    "worker_sha256":request["worker_sha256"]}})
            .to_string();
            seen.lock().unwrap().push(request);
            write!(
                reader.get_mut(),
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                reply.len(),
                reply
            )
            .unwrap();
        }
    });
    requests
}
fn digest_of(bytes: &[u8]) -> String {
    ml_exp_core::slurm::helper::sha256_bytes(bytes)
}
/// An upgrade through the real store, identities, Runs and server executable: no rebuild,
/// the Run freezes the new worker, and only its own Attempt can fetch it.
#[test]
fn an_upgrade_needs_no_rebuild_and_an_attempt_fetches_only_the_worker_it_froze() {
    use ml_exp_core::cloud::build::{BOOTSTRAP, Definition};
    let world = world();
    let temp = world.temp.path().to_path_buf();
    let environment = world.publish(&[("Dockerfile", &b"FROM x\n"[..])]);
    let code_id = world.publish(&[("train.py", &b"print('code')\n"[..])]);
    let store = temp.join("workers");
    fs::create_dir(&store).unwrap();
    let keep = |bytes: &[u8]| {
        let digest = digest_of(bytes);
        fs::write(store.join(digest.strip_prefix("sha256:").unwrap()), bytes).unwrap();
        digest
    };
    let (first, second) = (keep(b"worker one"), keep(b"worker two"));
    let absent = format!("sha256:{}", "9".repeat(64));
    let old_worker = format!("sha256:{}", "3".repeat(64));
    let old_runtime = Runtime {
        runtime_id: format!("runtime.{}", "1".repeat(64)),
        source_id: environment.clone(),
        image: format!("registry.example/team/code@sha256:{}", "2".repeat(64)),
        source_dir: "/workspace".into(),
        worker_path: "/usr/local/bin/ml-exp-worker-rs".into(),
        worker_sha256: old_worker.clone(),
    };
    let credentials = temp.join("credentials.json");
    private(
        &credentials,
        &json!({"provider":"pai","access_key_id":"synthetic","access_key_secret":"never-sent"}),
    );
    let signing_key = temp.join("worker-signing-key");
    ml_exp_io::write_atomic(
        &signing_key,
        b"synthetic-exchange-key-0000000000000000000000",
        true,
    )
    .unwrap();
    let storage = pai::Storage {
        cpfs_uri: "bmcpfs://cpfs-example.us-southeast-1.cpfs.aliyuncs.com/".into(),
        workspace_user: "researcher".into(),
        role_arn: "acs:ram::12345:role/pai-data-researcher".into(),
    };
    let socket = temp.join("builder.sock");
    let requests = fake_builder(&socket);
    let config = |worker: &str, with_store: bool| {
        let mut builds = json!({"socket":socket,"peer_uid":ml_exp_io::directory::effective_uid(),
            "worker_sha256":worker});
        if with_store {
            builds["worker_store"] = json!(store);
        }
        json!({"schema_version":1,"enabled":true,"allow_run_creation":true,"builds":builds,
            "exchange":{"public_url":"https://api.example/ml-expd","signing_key_file":signing_key},
            "backends":[{"executor":"pai-h200","profile":{"provider":"pai","region":"us-southeast-1",
                "workspace_id":"253","quota_id":"quota1","max_gpus":8,"max_cpus":200,"max_memory_gb":1600,
                "storage":storage},"run_root":"/mnt/workspace/ml-expd/runs","credentials_file":credentials,
                "registry_auth_file":null,"runtimes":{old_runtime.runtime_id.clone():old_runtime}}]})
    };
    Service::initialize(&world.settings).unwrap();
    let open = |config: &Value| {
        Arc::new(
            Service::open(
                world.settings.clone(),
                Settings::from_slice(&serde_json::to_vec(config).unwrap()).unwrap(),
                world.sources.clone(),
                None,
                Default::default(),
            )
            .unwrap(),
        )
    };
    let definition = Definition {
        source_id: environment.clone(),
        executor: "pai-h200".into(),
        base_image: format!("registry.example/base@sha256:{}", "4".repeat(64)),
        dockerfile: "Dockerfile".into(),
    };
    let built = |service: &Arc<Service>, id: &str| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let view = service.runtime_status("demo", id).unwrap();
            if view["status"] == "READY" {
                return view;
            }
            assert!(std::time::Instant::now() < deadline, "{view}");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    };

    // Without a store of workers nothing has changed: the identity names the worker.
    let legacy = open(&config(&first, false));
    let legacy_view = legacy.prepare_runtime("demo", definition.clone()).unwrap();
    let legacy_id = legacy_view["runtime_id"].as_str().unwrap().to_owned();
    assert_eq!(legacy_id, definition.id("demo", &first).unwrap());
    drop(legacy);

    // Prepared with one worker, and built with the one current when the build begins.
    let service = open(&config(&first, true));
    let prepared = service.prepare_runtime("demo", definition.clone()).unwrap();
    let id = prepared["runtime_id"].as_str().unwrap().to_owned();
    assert_eq!(id, definition.id("demo", BOOTSTRAP).unwrap());
    assert_ne!(id, legacy_id);
    let upgraded = open(&config(&second, true));
    upgraded
        .execute_runtime("demo", &id, &format!("BUILD {id}"))
        .unwrap();
    let view = built(&upgraded, &id);
    assert_eq!(view["runtime"]["worker_sha256"], second.as_str());
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert_eq!(
        requests.lock().unwrap()[0]["worker_sha256"],
        second.as_str()
    );
    // The upgrade itself: the same recipe finds the same Runtime and asks for no build.
    let later = open(&config(&first, true));
    let again = later.prepare_runtime("demo", definition.clone()).unwrap();
    assert_eq!(again["runtime_id"], id.as_str());
    assert_eq!(again["status"], "READY");
    assert_eq!(requests.lock().unwrap().len(), 1);
    // A Runtime stored before this is still read, under the identity it always had.
    assert_eq!(
        later.runtime_status("demo", &legacy_id).unwrap()["runtime_id"],
        legacy_id.as_str()
    );

    // A Run freezes the worker of the server that creates it, and may carry code at once.
    let request = |run: &str, runtime: &str, code: bool| -> Request {
        let mut value = json!({"run_id":run,"runtime_id":runtime,"executor":"pai-h200",
            "command":["python","train.py"],"resources":{"gpus":1,"cpus":8,"memory_gb":64},
            "walltime_seconds":300,"seed":42});
        if code {
            value["code_source_id"] = json!(code_id);
        }
        serde_json::from_value(value).unwrap()
    };
    let saved_package = |run: &str| {
        let slot = world
            .settings
            .registry_root
            .join("cloud-submissions/slots")
            .join(format!(
                "{}.json",
                attempt_slot("demo", run, "attempt-001").unwrap()
            ));
        let saved: Value = serde_json::from_slice(&fs::read(slot).unwrap()).unwrap();
        let execution: Execution = serde_json::from_value(saved["execution"].clone()).unwrap();
        execution.package().unwrap()
    };
    let prepare = |service: &Service, run: &str, transition: &str| {
        service
            .prepare(
                "demo",
                run,
                Approval {
                    max_gpu_hours: 1.,
                    reason: String::new(),
                },
                "2026-10-29T00:01:00Z",
                "2026-10-29T01:01:00Z",
                &transition.repeat(32),
            )
            .unwrap();
    };
    let manifest = |run: &str| -> Value {
        serde_json::from_slice(&fs::read(world.runs().join(run).join("manifest.json")).unwrap())
            .unwrap()
    };
    upgraded
        .create_run("demo", request("fresh", &id, true), NOW)
        .unwrap();
    prepare(&upgraded, "fresh", "a");
    assert_eq!(
        manifest("fresh")["native_cloud"]["worker_sha256"],
        second.as_str()
    );
    let package = saved_package("fresh");
    assert_eq!(package.worker_sha256, second);
    assert_eq!(package.code_source_id.as_deref(), Some(code_id.as_str()));

    // A Runtime that cannot bootstrap freezes nothing, on the same server.
    upgraded
        .create_run(
            "demo",
            request("old-image", &old_runtime.runtime_id, false),
            NOW,
        )
        .unwrap();
    prepare(&upgraded, "old-image", "b");
    assert!(
        manifest("old-image")["native_cloud"]
            .get("worker_sha256")
            .is_none()
    );
    assert_eq!(saved_package("old-image").worker_sha256, old_worker);
    // And cannot be paired with code it may not know how to fetch.
    assert_eq!(
        upgraded
            .create_run(
                "demo",
                request("old-code", &old_runtime.runtime_id, true),
                NOW
            )
            .unwrap_err()
            .code,
        "CODE_DELIVERY_WORKER_OUTDATED"
    );

    // A later server keeps both Runs as they were, and they can still be spent.
    let newest = open(&config(&absent, true));
    assert_eq!(
        newest
            .create_run("demo", request("fresh", &id, true), NOW)
            .unwrap()["reused"],
        true
    );
    prepare(&newest, "fresh", "a");
    assert_eq!(
        manifest("fresh")["native_cloud"]["worker_sha256"],
        second.as_str()
    );
    // A worker no job could fetch stops a new Run before anything is spent.
    assert_eq!(
        newest
            .create_run("demo", request("no-worker", &id, false), NOW)
            .unwrap_err()
            .code,
        "WORKER_BINARY_UNAVAILABLE"
    );
    assert!(!world.runs().join("no-worker").exists());

    // An unbuilt preparation with an old identity moves to the new one, and no further.
    let recipe = Definition {
        source_id: world.publish(&[("Dockerfile.other", &b"FROM y\n"[..])]),
        dockerfile: "Dockerfile.other".into(),
        ..definition.clone()
    };
    let on_old_scheme = open(&config(&first, false));
    let saved: ml_exp_core::preparation::Request = serde_json::from_value(json!({
        "request_id":"a".repeat(32),"run":{"run_id":"waited","executor":"pai-h200",
        "command":["train"],"resources":{"gpus":1,"cpus":8,"memory_gb":64},
        "walltime_seconds":300,"seed":42},"runtime":recipe,"approval":{"max_gpu_hours":1.0}}))
    .unwrap();
    let waiting = on_old_scheme
        .experiment_prepare("demo", saved, None)
        .unwrap();
    assert_eq!(
        waiting["runtime"]["runtime_id"],
        recipe.id("demo", &first).unwrap().as_str()
    );
    let preparation = waiting["preparation_id"].as_str().unwrap().to_owned();
    drop(on_old_scheme);
    let on_new_scheme = open(&config(&second, true));
    let context = || ml_exp_server::runs::Context {
        now: "2026-10-29T00:02:00Z",
        expires: "2026-10-29T01:02:00Z",
        transition_id: &"f".repeat(32),
    };
    assert!(
        on_new_scheme
            .experiment_continue(
                "demo",
                &preparation,
                &format!("PREPARE {preparation}"),
                None,
                context()
            )
            .is_ok()
    );
    let moved = recipe.id("demo", BOOTSTRAP).unwrap();
    built(&on_new_scheme, &moved);
    assert_eq!(
        on_new_scheme
            .experiment_status("demo", &preparation)
            .unwrap()["runtime"]["runtime_id"],
        moved.as_str()
    );
    drop((service, upgraded, later, newest, on_new_scheme));

    // Through the real server executable, as a job would.
    let read_file = temp.join("reads.json");
    private(&read_file, &serde_json::to_value(&world.settings).unwrap());
    let cloud_file = temp.join("cloud.json");
    private(&cloud_file, &config(&second, true));
    let token = "native-cloud-http-synthetic-token-000000000000";
    let token_file = temp.join("token");
    ml_exp_io::write_atomic(&token_file, token.as_bytes(), true).unwrap();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_ml-expd-rs"))
        .args(["--bind", "127.0.0.1:0", "--token-file"])
        .arg(&token_file)
        .arg("--read-config")
        .arg(&read_file)
        .arg("--source-archives")
        .arg("--cloud-config")
        .arg(&cloud_file)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let mut server = Server(child);
    let stdout = server.0.stdout.take().unwrap();
    let (sent, received) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let line = std::io::BufReader::new(stdout)
            .lines()
            .next()
            .unwrap()
            .unwrap();
        let _ = sent.send(line);
    });
    let address = received
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("server startup");
    let http = reqwest::blocking::Client::new();
    let get = |credential: Option<&str>, path: &str| {
        let request = http.get(format!("http://{address}{path}"));
        match credential {
            Some(credential) => request.bearer_auth(credential),
            None => request,
        }
        .send()
        .unwrap()
    };
    let executors: Value = get(Some(token), "/api/executors").json().unwrap();
    assert_eq!(
        executors["executors"][0]["capabilities"]["code_delivery"]["runtime_identity"],
        BOOTSTRAP
    );
    let path = "/api/worker-exchange/demo/fresh/attempt-001/worker";
    let own = package.channel.token.as_str();
    let other = saved_package("old-image").channel.token;
    let fetched = get(Some(own), path);
    assert_eq!(fetched.status().as_u16(), 200);
    assert_eq!(fetched.content_length(), Some(10));
    assert_eq!(fetched.bytes().unwrap().as_ref(), b"worker two");
    // The worker is reached by no other credential, and no request can name another.
    for (credential, path, status) in [
        (None, path, 401),
        (Some("0".repeat(64)), path, 401),
        (Some(other.clone()), path, 401),
        (Some(token.to_owned()), path, 401),
        (
            Some(own.to_owned()),
            "/api/worker-exchange/demo/fresh/attempt-002/worker",
            401,
        ),
        (
            Some(own.to_owned()),
            "/api/worker-exchange/demo/old-image/attempt-001/worker",
            401,
        ),
    ] {
        assert_eq!(
            get(credential.as_deref(), path).status().as_u16(),
            status,
            "{path}"
        );
    }
    assert_eq!(
        get(
            Some(own),
            &format!("{path}?worker=sha256:{}", "1".repeat(64))
        )
        .status()
        .as_u16(),
        422
    );
    // An Attempt on an old image names a worker this store never held.
    let refused = get(
        Some(&other),
        "/api/worker-exchange/demo/old-image/attempt-001/worker",
    );
    assert_eq!(refused.status().as_u16(), 409);
    // A stored file that is not what its name says is never served, nor is a link to it.
    let held = store.join(second.strip_prefix("sha256:").unwrap());
    fs::write(&held, b"worker 2wo").unwrap();
    let tampered = get(Some(own), path);
    assert_eq!(tampered.status().as_u16(), 409);
    assert!(
        tampered
            .text()
            .unwrap()
            .contains("WORKER_BINARY_UNAVAILABLE")
    );
    fs::remove_file(&held).unwrap();
    std::os::unix::fs::symlink(store.join(first.strip_prefix("sha256:").unwrap()), &held).unwrap();
    assert_eq!(get(Some(own), path).status().as_u16(), 409);
}
