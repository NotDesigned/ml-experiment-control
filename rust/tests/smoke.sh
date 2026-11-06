#!/usr/bin/env bash
set -euo pipefail

# No provider configuration, production workspace, Python or real user token.
smoke_bins="$(cd "${CARGO_TARGET_DIR:-rust/target}/${NATIVE_PROFILE:-debug}" && pwd)"
smoke_root="$(mktemp -d)"
smoke_pid=
cleanup() {
  local result=$?
  trap - EXIT
  if [[ -n "$smoke_pid" ]]; then
    kill "$smoke_pid" 2>/dev/null || true
    wait "$smoke_pid" 2>/dev/null || true
  fi
  # Source trees are deliberately sealed read-only. This is disposable test
  # state, so make it removable before deleting the fixture as an unprivileged
  # CI runner.
  chmod -R u+rwX "$smoke_root" 2>/dev/null || true
  rm -rf -- "$smoke_root"
  exit "$result"
}
trap cleanup EXIT
umask 077
printf '%s' 'native-smoke-test-token-not-a-real-credential' > "$smoke_root/token"
mkdir "$smoke_root/bin"
for fake_python in python python3 python3.12; do
  printf '#!/bin/sh\necho "unexpected Python platform subprocess" >&2\nexit 99\n' > "$smoke_root/bin/$fake_python"
  chmod 700 "$smoke_root/bin/$fake_python"
done
export PATH="$smoke_root/bin:$PATH"

# Explicit initialization is the only operation that creates locks.
# The configured evidence is a synthetic private fixture, never production.
"$smoke_bins/ml-expd-rs" --initialize-registry "$smoke_root/registry" > "$smoke_root/workspace-id"
if "$smoke_bins/ml-expd-rs" --initialize-registry "$smoke_root/registry" > "$smoke_root/reinitialize.log" 2>&1; then
  echo 'existing registry was unexpectedly reinitialized' >&2; exit 1
fi
mkdir -p "$smoke_root/project/runs/train/attempts/attempt-001"
cat > "$smoke_root/project/project.json" <<'JSON'
{"schema_version":1,"project":"demo","title":"Native smoke fixture","run_roots":["runs"]}
JSON
cat > "$smoke_root/project/runs/train/manifest.json" <<'JSON'
{"schema_version":1,"project":"demo","run_id":"train","source_id":"source.fixture","image_id":"sha256:fixture","config_path":"config.json","seed":42,"campaign":"fixture","evaluation":{"protocol_id":"fixture-protocol"},"parameters":{},"resources":{"gpus":0},"container":{"runtime_id":"runtime.fixture"},"backend":{"kind":"slurm"},"inputs":[],"native_creation":{"format":"native-run-creation.v1"}}
JSON
cat > "$smoke_root/project/runs/train/attempts/attempt-001/attempt.json" <<'JSON'
{"schema_version":1,"project":"demo","run_id":"train","attempt_id":"attempt-001","source_id":"source.fixture","image_id":"sha256:fixture","config_path":"config.json","seed":42,"campaign":"fixture","evaluation":{"protocol_id":"fixture-protocol"},"parameters":{},"resources":{"gpus":0},"container":{"runtime_id":"runtime.fixture"},"backend":{"kind":"slurm"},"inputs":[],"native_creation":{"format":"native-run-creation.v1"}}
JSON
cat > "$smoke_root/project/runs/train/status.json" <<'JSON'
{"project":"demo","run_id":"train","attempt_id":"attempt-001","state":"RUNNING"}
JSON
cp "$smoke_root/project/runs/train/status.json" "$smoke_root/project/runs/train/attempts/attempt-001/status.json"
printf 'native exact log\n' > "$smoke_root/project/runs/train/attempts/attempt-001/stdout.log"
cat > "$smoke_root/project/runs/train/attempts/attempt-001/metrics.jsonl" <<'JSON'
{"name":"loss","value":2.5,"unit":"nats/token","step":1,"protocol_id":"fixture-protocol"}
JSON
mkdir -p "$smoke_root/source"
printf 'print("synthetic source")\n' > "$smoke_root/source/train.py"
printf '{"seed":42}\n' > "$smoke_root/source/config.json"
cat > "$smoke_root/reads.json" <<JSON
{"schema_version":1,"registry_root":"$smoke_root/registry","projects":[{"project":"demo","project_file":"$smoke_root/project/project.json","run_roots":["$smoke_root/project/runs"]}]}
JSON

# port 0 is assigned by the OS, avoiding collisions with another checkout.
cat > "$smoke_root/server.json" <<JSON
{"schema_version":1,"bind":"127.0.0.1:0","token_file":"$smoke_root/token","read_config":"$smoke_root/reads.json"}
JSON
"$smoke_bins/ml-expd-rs" --config "$smoke_root/server.json" > "$smoke_root/server.log" 2> "$smoke_root/server.stderr" &
smoke_pid=$!
for _ in {1..100}; do
  if [[ -s "$smoke_root/server.log" ]]; then break; fi
  kill -0 "$smoke_pid"
  sleep .05
done
IFS= read -r server_address < "$smoke_root/server.log"
[[ "$server_address" == 127.0.0.1:* ]]
export ML_EXPD_API_TOKEN_FILE="$smoke_root/token"
unset ML_EXPD_API_TOKEN ML_EXPD_API_URL ML_EXPD_CLIENT_CONFIG
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" --url "http://$server_address" config set --token-file "$smoke_root/token" > "$smoke_root/config.json"
unset ML_EXPD_API_TOKEN_FILE
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" check > "$smoke_root/health.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" get /api/readiness > "$smoke_root/readiness.json"
if grep -q 'source-archive.v1' "$smoke_root/health.json"; then
  echo 'source imports unexpectedly enabled' >&2; exit 1
fi
if "$smoke_bins/ml-exp" --config "$smoke_root/client.json" get /api/actions/execute > "$smoke_root/not-implemented.json" 2>&1; then
  echo 'unported mutation unexpectedly succeeded' >&2; exit 1
fi
grep -q 'HTTP 501' "$smoke_root/not-implemented.json"
grep -q '"scheduler_mutations": false' "$smoke_root/health.json"
grep -q '"training": "INCOMPLETE"' "$smoke_root/readiness.json"
if grep -q 'native-smoke-test-token' "$smoke_root/config.json" "$smoke_root/health.json" "$smoke_root/server.log" "$smoke_root/server.stderr"; then
  echo 'credential leaked into output' >&2; exit 1
fi

# Real binaries compose the same pure reducers and bounded read API.
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" run status --project demo --run train --json > "$smoke_root/status.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" run logs --project demo --run train > "$smoke_root/logs.txt"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" run metrics --project demo --run train > "$smoke_root/metrics.json"
grep -q '"state": "RUNNING"' "$smoke_root/status.json"
grep -q '"download_available": false' "$smoke_root/status.json"
grep -q 'native exact log' "$smoke_root/logs.txt"
grep -q '2.5' "$smoke_root/metrics.json"
grep -q 'nats/token' "$smoke_root/metrics.json"
[[ ! -e "$smoke_root/registry/tracking" ]]

# Source archives are a separate explicit namespace and never execute source code.
kill "$smoke_pid"
wait "$smoke_pid"
smoke_pid=
"$smoke_bins/ml-expd-rs" --read-config "$smoke_root/reads.json" --initialize-source-archives
: > "$smoke_root/server.log"
: > "$smoke_root/server.stderr"
"$smoke_bins/ml-expd-rs" --bind 127.0.0.1:0 --token-file "$smoke_root/token" --read-config "$smoke_root/reads.json" --source-archives > "$smoke_root/server.log" 2> "$smoke_root/server.stderr" &
smoke_pid=$!
for _ in {1..100}; do
  if [[ -s "$smoke_root/server.log" ]]; then break; fi
  kill -0 "$smoke_pid"
  sleep .05
done
IFS= read -r server_address < "$smoke_root/server.log"
[[ "$server_address" == 127.0.0.1:* ]]
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" --url "http://$server_address" config set --token-file "$smoke_root/token" > "$smoke_root/config.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" check > "$smoke_root/source-health.json"
grep -q '"scheduler_mutations": false' "$smoke_root/source-health.json"
grep -q 'source-archive.v1' "$smoke_root/source-health.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" source import --project demo --directory "$smoke_root/source" > "$smoke_root/source-import.json"
source_id="$(jq -r '.source_id' "$smoke_root/source-import.json")"
archive_sha="$(jq -r '.observation.sha256' "$smoke_root/source-import.json")"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" source show --project demo --source "$source_id" > "$smoke_root/source-show.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" source lookup --project demo --sha256 "$archive_sha" > "$smoke_root/source-lookup.json"
grep -q '"source_id"' "$smoke_root/source-show.json"
grep -q '"source_id"' "$smoke_root/source-lookup.json"
grep -q 'synthetic source' "$smoke_root/source/train.py"

# Metadata writes are an explicit separate mode; approved files are read only.
sha256sum "$smoke_root/project/project.json" "$smoke_root/project/runs/train/manifest.json" > "$smoke_root/research.before"
if "$smoke_bins/ml-exp" --config "$smoke_root/client.json" project list > "$smoke_root/disabled.log" 2>&1; then
  echo 'unconfigured project lifecycle unexpectedly available' >&2; exit 1
fi
kill "$smoke_pid"
wait "$smoke_pid"
smoke_pid=
"$smoke_bins/ml-expd-rs" --read-config "$smoke_root/reads.json" --initialize-project-lifecycle
start_lifecycle() {
  # Truncate synchronously: background redirection may run after the first poll.
  : > "$smoke_root/server.log"
  : > "$smoke_root/server.stderr"
  "$smoke_bins/ml-expd-rs" --bind 127.0.0.1:0 --token-file "$smoke_root/token" --read-config "$smoke_root/reads.json" --project-lifecycle > "$smoke_root/server.log" 2> "$smoke_root/server.stderr" &
  smoke_pid=$!
  for _ in {1..100}; do
    if [[ -s "$smoke_root/server.log" ]]; then break; fi
    kill -0 "$smoke_pid"
    sleep .05
  done
  IFS= read -r server_address < "$smoke_root/server.log"
  [[ "$server_address" == 127.0.0.1:* ]]
  "$smoke_bins/ml-exp" --config "$smoke_root/client.json" --url "http://$server_address" config set --token-file "$smoke_root/token" > "$smoke_root/config.json"
}
start_lifecycle
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" project pause demo --expected-revision 1 > "$smoke_root/paused.json"
grep -q '"state": "PAUSED"' "$smoke_root/paused.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" project resume demo --expected-revision 2 > "$smoke_root/resumed.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" project archive demo --reason completed --expected-revision 3 > "$smoke_root/archived.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" project restore demo --expected-revision 4 > "$smoke_root/restored.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" project unregister demo --expected-revision 5 > "$smoke_root/unregistered.json"
kill "$smoke_pid"
wait "$smoke_pid"
smoke_pid=
start_lifecycle
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" project list > "$smoke_root/empty-ledger.json"
grep -q '"projects": \[\]' "$smoke_root/empty-ledger.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" get /api/projects > "$smoke_root/empty-projects.json"
[[ "$(cat "$smoke_root/empty-projects.json")" == '[]' ]]
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" project register "$smoke_root/project/project.json" --expected-revision 6 > "$smoke_root/registered.json"
"$smoke_bins/ml-exp" --config "$smoke_root/client.json" run status --project demo --run train --json > "$smoke_root/registered-status.json"
sha256sum "$smoke_root/project/project.json" "$smoke_root/project/runs/train/manifest.json" > "$smoke_root/research.after"
cmp "$smoke_root/research.before" "$smoke_root/research.after"

# A payload command can use any language; the platform here is three Rust binaries.
"$smoke_bins/ml-exp-worker-rs" --workdir "$smoke_root" --output "$smoke_root/success" --timeout-seconds 5 -- sh -c 'printf result > "$OUTPUT_DIR/result.txt"; printf native-worker' > "$smoke_root/worker.json"
[[ "$(cat "$smoke_root/success/result.txt")" == result ]]
grep -q '"exit_code":0' "$smoke_root/worker.json"
if "$smoke_bins/ml-exp-worker-rs" --workdir "$smoke_root" --output "$smoke_root/failure" --timeout-seconds 5 -- sh -c 'exit 7' > "$smoke_root/failure.json"; then
  echo 'failed payload reported success' >&2; exit 1
fi
grep -q '"exit_code":7' "$smoke_root/failure.json"
if "$smoke_bins/ml-exp-worker-rs" --workdir "$smoke_root" --output "$smoke_root/timeout" --timeout-seconds 1 -- sh -c 'sleep 60 & wait' > "$smoke_root/timeout.json"; then
  echo 'timed-out payload reported success' >&2; exit 1
fi
grep -q '"timed_out":true' "$smoke_root/timeout.json"

echo 'PASS: native discovery/reads, explicit project lifecycle and restart, unchanged research files, no unsupported mutation or Python subprocess, worker success/failure/timeout'
