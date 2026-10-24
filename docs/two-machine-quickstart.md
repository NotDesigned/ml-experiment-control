# Deploy a basic API and a separate client

This example installs a basic API on one Linux machine and the standalone client
on another. Both can be CPU DSW instances in an approved VPC. It imports source
without Docker, a builder, S3, backend credentials or a training executor. It does
not run a remote experiment. Add those components using the operator guide after
the basic connection works.

You need Python with `venv`/`ensurepip`, `openssl`, `sha256sum`, and a reachable
private IP or DNS name for the server. The commands below use root on the server;
the client does not need root. Use new directories and machines for this example,
not an existing installation. A DSW container need not have systemd.

## 1. Obtain the installation packages

Use reviewed release artifacts from the project's GitHub Releases, or matching
artifacts supplied by your operator. Do not mix wheels from different releases.
Verify the archive or wheel hashes against the trusted release checksum file.
The offline bundle contains all runtime dependency wheels, exact versions and
hashes; installation never contacts a package index or compiles Rust.

Use the server bundle matching your Python minor version, Linux architecture and
compatible libc; check `manifest.json`. The standard build targets Linux x86-64
and Python 3.12. The client bundle needs only Python 3.10+.

On the server, extract the **verified** server archive and enter `ml-expd-server`:

```bash
tar -xzf /path/to/server-bundle.tar.gz
cd ml-expd-server
install -d -m 0755 /opt/ml-expd
./install.sh /opt/ml-expd/.venv
```

`install.sh` verifies bundled file hashes, refuses an existing destination, creates
the virtual environment and installs locally with `--no-index --require-hashes`.
It does not start services or configure cloud accounts.
The installer sets public program permissions independently of the calling
shell's private umask; tokens, keys and state are created separately below.
If your operator supplies
individual wheels instead, follow the wheel installation in the operator guide;
their runtime dependencies require a reachable package index.

## 2. Create the server's user, directories and token

Run in the extracted server bundle directory:

```bash
set -eu
if [ -e /etc/ml-expd/ml-expd.yaml ] || [ -e /etc/ml-expd/http.token ]; then
  echo 'Existing ML-Expd configuration: use the upgrade procedure instead.' >&2
  exit 1
fi
id ml-expd >/dev/null 2>&1 || useradd --system --user-group \
  --home-dir /var/lib/ml-expd --shell /usr/sbin/nologin ml-expd
install -d -m 0750 -o root -g ml-expd /etc/ml-expd
install -d -m 0700 -o ml-expd -g ml-expd /var/lib/ml-expd
install -m 0640 -o root -g ml-expd minimal-api.yaml /etc/ml-expd/ml-expd.yaml
umask 077
python3 - <<'PY'
from pathlib import Path
import secrets
with Path('/etc/ml-expd/http.token').open('x') as token:
    token.write(secrets.token_urlsafe(32) + '\n')
PY
chown ml-expd:ml-expd /etc/ml-expd/http.token
chmod 0600 /etc/ml-expd/http.token
```

The token is never printed. These are fresh-install commands: do not overwrite an
existing config/token/state directory. The template enables project creation and
source import, with scheduler writes, collector and telemetry disabled.

## 3. Configure HTTPS

For a public service, use a certificate from your operator and skip this private
CA example. For two test machines, use a distinct private CA and server certificate.
Set `SERVER_IP` to this server's actual address reachable from the client. Only
open the selected API port within your approved network; this example does not
change security groups, gateways or account permissions.

```bash
export SERVER_IP='10.0.0.12'  # replace with the reachable private IPv4 address
openssl req -x509 -newkey rsa:2048 -nodes -days 7 \
  -keyout /etc/ml-expd/ca.key -out /etc/ml-expd/ca.crt \
  -subj '/CN=ML-Expd example CA' \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign'
openssl req -new -newkey rsa:2048 -nodes \
  -keyout /etc/ml-expd/server.key -out /etc/ml-expd/server.csr \
  -subj '/CN=ML-Expd example server'
cat > /etc/ml-expd/server.ext <<EOF
subjectAltName=IP:${SERVER_IP}
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
EOF
openssl x509 -req -in /etc/ml-expd/server.csr -days 7 \
  -CA /etc/ml-expd/ca.crt -CAkey /etc/ml-expd/ca.key -CAcreateserial \
  -extfile /etc/ml-expd/server.ext -out /etc/ml-expd/server.crt
chown ml-expd:ml-expd /etc/ml-expd/server.key /etc/ml-expd/server.crt
chmod 0600 /etc/ml-expd/server.key /etc/ml-expd/ca.key
```

The SAN must match the address the client uses. For DNS, use a DNS SAN and that
hostname in the API URL. Do not disable certificate or hostname verification.
The CA private key stays with the operator and is not needed by the daemon/client.

## 4. Start the server

```bash
runuser -u ml-expd -- env PATH="/opt/ml-expd/.venv/bin:$PATH" \
  /opt/ml-expd/.venv/bin/ml-expd --config /etc/ml-expd/ml-expd.yaml doctor
runuser -u ml-expd -- env PATH="/opt/ml-expd/.venv/bin:$PATH" \
  /opt/ml-expd/.venv/bin/ml-expd --config /etc/ml-expd/ml-expd.yaml \
  --host 0.0.0.0 --port 8765 \
  --ssl-certfile /etc/ml-expd/server.crt --ssl-keyfile /etc/ml-expd/server.key
```

Keep the second command running. For a persistent Linux service, adapt the
operator guide's systemd unit to the same paths, UID, PATH and TLS options.
The basic `doctor` check does not test cloud access or reserve resources.

## 5. Privately provide the client connection files

In another server terminal, set the same `SERVER_IP` and prepare only the API URL,
token and public CA certificate:

```bash
export SERVER_IP='10.0.0.12'  # same address as step 3
umask 077
install -d -m 0700 /root/ml-expd-connection
install -m 0600 /etc/ml-expd/http.token /root/ml-expd-connection/client.token
install -m 0644 /etc/ml-expd/ca.crt /root/ml-expd-connection/ca.crt
printf 'https://%s:8765\n' "$SERVER_IP" > /root/ml-expd-connection/api-url
tar -C /root/ml-expd-connection -czf /root/ml-expd-connection.tar.gz \
  api-url client.token ca.crt
```

Transfer this private archive and the verified client installation bundle through
an authenticated channel, such as your operator's file transfer or DSW's approved
file-transfer channel. Never put the connection archive in Git, uploaded source,
public storage or logs. Do not send either TLS private key or cloud credentials.
Remove the temporary connection archive/directory on both ends after provisioning.

## 6. Install and connect the separate client

On the client machine, extract the verified client bundle, then:

```bash
cd ml-expd-client
./install.sh "$HOME/.local/share/ml-expd/client-venv"
export PATH="$HOME/.local/share/ml-expd/client-venv/bin:$PATH"
install -d -m 0700 "$HOME/.config/ml-expd"
tar --no-same-owner -xzf /path/to/ml-expd-connection.tar.gz -C "$HOME/.config/ml-expd"
chmod 0600 "$HOME/.config/ml-expd/client.token"
unset ML_EXPD_API_TOKEN
export ML_EXPD_API_URL="$(cat "$HOME/.config/ml-expd/api-url")"
export ML_EXPD_API_TOKEN_FILE="$HOME/.config/ml-expd/client.token"
export SSL_CERT_FILE="$HOME/.config/ml-expd/ca.crt"
ml-exp check
ml-exp check --json --schema "$HOME/.config/ml-expd/openapi.json"
```

The default output summarizes API, source import and training readiness. `--json`
retains complete machine-readable evidence. This basic deployment should report
`API: READY`, `Source import: AVAILABLE`, `Training: INCOMPLETE`.
`ml-exp check --training` intentionally exits 2 and names the unconfigured
executor/builder/result store and disabled scheduler writes. No cloud job starts.
For a publicly trusted TLS certificate, use normal system trust instead of the
example private CA.

## 7. Upload source without building or submitting

Keep connection files outside the source directory:

```bash
mkdir source
printf 'print("hello from my experiment")\n' > source/main.py
ml-exp source-import --project demo --directory ./source
```

The response contains `source_id`, `tree_digest` and import metadata. Repeat the
command to verify the same source tree retains its immutable identity. No
Dockerfile, image build, Run, Attempt or GPU is required. The existing source
limits and rejection of credential-looking filenames still apply; this is not a
content scan for secrets. For training, configure a backend, builder and result
store first, then use the ordinary one-config `ml-exp experiment` workflow.

Stop the example server with Ctrl-C when done. Stop/delete owned temporary DSWs
in your cloud console/control channel and verify they are no longer allocated.
