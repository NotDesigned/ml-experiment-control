#!/usr/bin/env bash
set -euo pipefail

# No provider configuration, production workspace, Python or real user token.
preview_bins="$(cd "${CARGO_TARGET_DIR:-rust/target}/debug" && pwd)"
preview_root="$(mktemp -d)"
preview_pid=
cleanup() {
  local result=$?
  trap - EXIT
  if [[ -n "$preview_pid" ]]; then
    kill "$preview_pid" 2>/dev/null || true
    wait "$preview_pid" 2>/dev/null || true
  fi
  rm -rf -- "$preview_root"
  exit "$result"
}
trap cleanup EXIT
umask 077
printf '%s' 'native-preview-test-token-not-a-real-credential' > "$preview_root/token"
mkdir "$preview_root/bin"
for preview_python in python python3 python3.12; do
  printf '#!/bin/sh\necho "unexpected Python platform subprocess" >&2\nexit 99\n' > "$preview_root/bin/$preview_python"
  chmod 700 "$preview_root/bin/$preview_python"
done
export PATH="$preview_root/bin:$PATH"

# Explicit initialization is the only operation that creates preview locks.
# The configured evidence is a synthetic private fixture, never production.
"$preview_bins/ml-expd-rs" --initialize-registry "$preview_root/registry" > "$preview_root/workspace-id"
if "$preview_bins/ml-expd-rs" --initialize-registry "$preview_root/registry" > "$preview_root/reinitialize.log" 2>&1; then
  echo 'existing registry was unexpectedly reinitialized' >&2; exit 1
fi
mkdir -p "$preview_root/project/runs/train/attempts/attempt-001"
cat > "$preview_root/project/project.json" <<'JSON'
{"schema_version":1,"project":"demo","title":"Native smoke fixture"}
JSON
cat > "$preview_root/project/runs/train/manifest.yaml" <<'JSON'
{"schema_version":1,"identity_version":2,"project":"demo","run_id":"train","source_id":"source.fixture","image_id":"sha256:fixture","config_path":"config.json","seed":42,"campaign":"fixture","evaluation":{"protocol_id":"fixture-protocol"},"parameters":{},"resources":{"gpus":0},"container":{"runtime_id":"runtime.fixture"},"backend":{"kind":"slurm"},"inputs":[]}
JSON
cat > "$preview_root/project/runs/train/attempts/attempt-001/attempt.yaml" <<'JSON'
{"schema_version":1,"identity_version":2,"project":"demo","run_id":"train","attempt_id":"attempt-001","source_id":"source.fixture","image_id":"sha256:fixture","config_path":"config.json","seed":42,"campaign":"fixture","evaluation":{"protocol_id":"fixture-protocol"},"parameters":{},"resources":{"gpus":0},"container":{"runtime_id":"runtime.fixture"},"backend":{"kind":"slurm"},"inputs":[]}
JSON
cat > "$preview_root/project/runs/train/status.json" <<'JSON'
{"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"RUNNING"}
JSON
cp "$preview_root/project/runs/train/status.json" "$preview_root/project/runs/train/attempts/attempt-001/status.json"
printf 'native exact log\n' > "$preview_root/project/runs/train/attempts/attempt-001/stdout.log"
cat > "$preview_root/project/runs/train/attempts/attempt-001/metrics.jsonl" <<'JSON'
{"name":"loss","value":2.5,"unit":"nats/token","step":1,"protocol_id":"fixture-protocol"}
JSON
cat > "$preview_root/reads.json" <<JSON
{"schema_version":1,"registry_root":"$preview_root/registry","projects":[{"project":"demo","project_file":"$preview_root/project/project.json","run_roots":["$preview_root/project/runs"],"runs":[{"run_id":"train","root_index":0,"relative_components":["train"]}]}]}
JSON

# port 0 is assigned by the OS, avoiding collisions with another checkout.
"$preview_bins/ml-expd-rs" --bind 127.0.0.1:0 --token-file "$preview_root/token" --read-config "$preview_root/reads.json" > "$preview_root/server.log" 2>&1 &
preview_pid=$!
for _ in {1..100}; do
  if [[ -s "$preview_root/server.log" ]]; then break; fi
  kill -0 "$preview_pid"
  sleep .05
done
IFS= read -r preview_address < "$preview_root/server.log"
[[ "$preview_address" == 127.0.0.1:* ]]
export ML_EXPD_API_TOKEN_FILE="$preview_root/token"
unset ML_EXPD_API_TOKEN ML_EXPD_API_URL ML_EXPD_CLIENT_CONFIG
"$preview_bins/ml-exp-rs" --config "$preview_root/client.json" --url "http://$preview_address" config set --token-file "$preview_root/token" > "$preview_root/config.json"
unset ML_EXPD_API_TOKEN_FILE
"$preview_bins/ml-exp-rs" --config "$preview_root/client.json" check > "$preview_root/health.json"
"$preview_bins/ml-exp-rs" --config "$preview_root/client.json" get /api/readiness > "$preview_root/readiness.json"
if "$preview_bins/ml-exp-rs" --config "$preview_root/client.json" get /api/actions/execute > "$preview_root/not-implemented.json" 2>&1; then
  echo 'unported mutation unexpectedly succeeded' >&2; exit 1
fi
grep -q 'HTTP 501' "$preview_root/not-implemented.json"
grep -q '"scheduler_mutations": false' "$preview_root/health.json"
grep -q '"training": "INCOMPLETE"' "$preview_root/readiness.json"
if grep -q 'native-preview-test-token' "$preview_root/config.json" "$preview_root/health.json" "$preview_root/server.log"; then
  echo 'credential leaked into output' >&2; exit 1
fi

# Real binaries compose the same pure reducers and bounded read API.
"$preview_bins/ml-exp-rs" --config "$preview_root/client.json" run status --project demo --run train --json > "$preview_root/status.json"
"$preview_bins/ml-exp-rs" --config "$preview_root/client.json" run logs --project demo --run train > "$preview_root/logs.txt"
"$preview_bins/ml-exp-rs" --config "$preview_root/client.json" run metrics --project demo --run train > "$preview_root/metrics.json"
grep -q '"state": "RUNNING"' "$preview_root/status.json"
grep -q '"download_available": false' "$preview_root/status.json"
grep -q 'native exact log' "$preview_root/logs.txt"
grep -q '2.5' "$preview_root/metrics.json"
grep -q 'nats/token' "$preview_root/metrics.json"
[[ ! -e "$preview_root/registry/tracking" ]]

# A payload command can use any language; the platform here is three Rust binaries.
"$preview_bins/ml-exp-worker-rs" --workdir "$preview_root" --output "$preview_root/success" --timeout-seconds 5 -- sh -c 'printf result > "$OUTPUT_DIR/result.txt"; printf native-worker' > "$preview_root/worker.json"
[[ "$(cat "$preview_root/success/result.txt")" == result ]]
grep -q '"exit_code":0' "$preview_root/worker.json"
if "$preview_bins/ml-exp-worker-rs" --workdir "$preview_root" --output "$preview_root/failure" --timeout-seconds 5 -- sh -c 'exit 7' > "$preview_root/failure.json"; then
  echo 'failed payload reported success' >&2; exit 1
fi
grep -q '"exit_code":7' "$preview_root/failure.json"
if "$preview_bins/ml-exp-worker-rs" --workdir "$preview_root" --output "$preview_root/timeout" --timeout-seconds 1 -- sh -c 'sleep 60 & wait' > "$preview_root/timeout.json"; then
  echo 'timed-out payload reported success' >&2; exit 1
fi
grep -q '"timed_out":true' "$preview_root/timeout.json"

echo 'PASS: native server/client exact reads, isolated registry, private config, no unsupported mutation or Python subprocess, worker success/failure/timeout'
