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

# port 0 is assigned by the OS, avoiding collisions with another checkout.
"$preview_bins/ml-expd-rs" --bind 127.0.0.1:0 --token-file "$preview_root/token" > "$preview_root/server.log" 2>&1 &
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

echo 'PASS: native server/client handshake, private config, no unsupported mutation, worker success/failure/timeout'
