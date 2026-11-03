#!/usr/bin/env bash
set -euo pipefail
umask 077

# Client-only installation. Never starts a server or reads an API credential.
native_version=
native_prefix="${XDG_BIN_HOME:-$HOME/.local/bin}"
while (($#)); do
  case "$1" in
    --version) native_version="${2:?missing version}"; shift 2 ;;
    --prefix) native_prefix="${2:?missing install directory}"; shift 2 ;;
    -h|--help) echo 'Usage: install-native.sh --version X.Y.Z [--prefix /absolute/bin]'; exit 0 ;;
    *) echo 'Unknown native installer option' >&2; exit 2 ;;
  esac
done
[[ "$native_version" =~ ^(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})\.(0|[1-9][0-9]{0,5})$ ]] || { echo 'An exact stable version is required' >&2; exit 2; }
[[ "$native_prefix" == /* && ! -L "$native_prefix" ]] || { echo 'Use an absolute install directory without a directory symlink' >&2; exit 2; }
case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) native_target=x86_64-unknown-linux-gnu ;;
  *) echo 'This release installer supports Linux x86_64, including WSL2' >&2; exit 2 ;;
esac
command -v curl >/dev/null
command -v sha256sum >/dev/null
mkdir -p -- "$native_prefix"
native_destination="$native_prefix/ml-exp-rs"
[[ ! -L "$native_destination" ]] || { echo 'Refusing to replace a symlink' >&2; exit 2; }
native_tmp="$(mktemp -d "$native_prefix/.ml-exp-install.XXXXXX")"
trap 'rm -rf -- "$native_tmp"' EXIT
native_base="https://github.com/NotDesigned/ml-experiment-control/releases/download/native-v$native_version"
native_asset="ml-exp-rs-$native_target"
curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location --max-time 60 --max-filesize 65536 "$native_base/SHA256SUMS" -o "$native_tmp/SHA256SUMS"
native_hash="$(awk -v name="$native_asset" '$2 == name { hash=$1; count++ } END { if(count!=1) exit 1; print hash }' "$native_tmp/SHA256SUMS")"
[[ "$native_hash" =~ ^[0-9a-f]{64}$ ]]
curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location --max-time 180 --max-filesize 134217728 "$native_base/$native_asset" -o "$native_tmp/ml-exp-rs"
printf '%s  %s\n' "$native_hash" "$native_tmp/ml-exp-rs" | sha256sum --check --status
chmod 755 "$native_tmp/ml-exp-rs"
native_actual="$(timeout 5 "$native_tmp/ml-exp-rs" --version)"
[[ "$native_actual" == "ml-exp-rs $native_version" ]] || { echo 'Downloaded executable version differs' >&2; exit 1; }
mv -f -- "$native_tmp/ml-exp-rs" "$native_destination"
echo "Installed ml-exp-rs $native_version to $native_destination"
