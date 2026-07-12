"""WYD Slurm/Apptainer side-effect adapter."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .services import BackendServices
from ..preflight import PreflightCheck, PreflightReport
from ..redaction import redact_line
from ..checkpoints import select_latest_checkpoint_name
from ..identity import IdentityReport


SLURM_STATES = {
    "PENDING": "QUEUED", "CONFIGURING": "QUEUED", "REQUEUED": "QUEUED",
    "REQUEUE_FED": "QUEUED", "RUNNING": "RUNNING", "COMPLETING": "RUNNING",
    "COMPLETED": "SUCCEEDED", "PREEMPTED": "PREEMPTED",
    "FAILED": "FAILED", "NODE_FAIL": "FAILED", "OUT_OF_MEMORY": "FAILED",
    "TIMEOUT": "FAILED", "CANCELLED": "CANCELLED",
}


def normalize_state(raw_state: str, exit_code: str | None = None) -> str:
    raw = raw_state.split()[0].rstrip("+").upper()
    state = SLURM_STATES.get(raw, "UNKNOWN")
    return "FAILED" if raw == "COMPLETED" and exit_code not in {None, "", "0:0"} else state


def scheduler_job_name(run_id: str, attempt_id: str) -> str:
    raw = f"{run_id}--{attempt_id}"
    if len(raw) <= 128:
        return raw
    return f"{raw[:113]}--{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def checkpoint_probe_command(run_dir: str) -> str:
    """Build a remote read-only probe that validates marker step and payload size."""
    code = (
        "import glob,json,os,re,sys; root=sys.argv[1]; "
        "markers=glob.glob(os.path.join(root,'checkpoint_*.complete')); "
        "\nfor marker in markers:\n"
        " payload=marker[:-9]; match=re.fullmatch(r'checkpoint_(\\d+)',os.path.basename(payload));\n"
        " if not match or not os.path.isfile(payload): continue\n"
        " try: metadata=json.load(open(marker,encoding='utf-8'))\n"
        " except (OSError,ValueError): continue\n"
        " if metadata.get('step')==int(match.group(1)) and metadata.get('bytes')==os.path.getsize(payload): print(os.path.basename(payload))"
    )
    return shlex.join(["python3", "-c", code, run_dir])


def log_probe_command(paths: list[str], *, tail: int) -> str:
    """Build a bounded probe that reads the first existing exact log path."""
    if not 1 <= tail <= 10000:
        raise ValueError("tail must be between 1 and 10000")
    if not paths:
        raise ValueError("at least one log path is required")
    candidates = " ".join(shlex.quote(path) for path in paths)
    return (
        f"for path in {candidates}; do "
        "if test -f \"$path\"; then "
        "printf '%s\\n' \"$path\"; "
        f"tail -n {tail} -- \"$path\"; exit 0; "
        "fi; done; exit 1"
    )


def render_job(manifest: dict[str, Any]) -> str:
    backend, resources = manifest["backend"], manifest.get("resources", {})
    run_dir, source_dir, sif_path = (
        manifest["storage"]["run_dir"], backend["source_dir"], backend["sif_path"]
    )
    mount_root = str(backend["mount_root"])
    project_root = str(manifest["storage"]["project_data_root"])
    cache = str(backend.get("apptainer_cache_dir", f"{project_root}/apptainer/cache"))
    temp = str(backend.get("apptainer_tmp_dir", f"{project_root}/apptainer/tmp"))
    command = shlex.join(manifest["command"])
    execution = manifest["execution"]
    container_path = str(execution["source_mount"])
    workdir = str(execution["workdir"])
    comment = shlex.quote(f"{manifest.get('campaign', 'campaign')}/{manifest['run_id']}/{manifest['attempt_id']}")
    job_name = scheduler_job_name(str(manifest["run_id"]), str(manifest["attempt_id"]))
    return f"""#!/usr/bin/env bash
#SBATCH --partition={backend['partition']}
#SBATCH --account={backend['account']}
#SBATCH --qos={backend['qos']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={int(resources.get('cpus', 8))}
#SBATCH --gres={backend['gres']}
#SBATCH --time={backend['time']}
#SBATCH --job-name={job_name}
#SBATCH --comment={comment}
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export APPTAINER_CACHEDIR={shlex.quote(cache)}
export APPTAINER_TMPDIR={shlex.quote(temp)}
export BACKEND_JOB_ID="$SLURM_JOB_ID"
mkdir -p {shlex.quote(run_dir)}
attempt_log_dir={shlex.quote(f"{run_dir}/attempts/{manifest['attempt_id']}")}
mkdir -p "$attempt_log_dir"
exec > >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.out") \\
     2> >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.err" >&2)
test -d {shlex.quote(source_dir)}
test -s {shlex.quote(sif_path)}
srun apptainer exec --nv \\
  --bind {shlex.quote(mount_root)}:{shlex.quote(mount_root)} \\
  --bind {shlex.quote(source_dir)}:{shlex.quote(container_path)} \\
  --pwd {shlex.quote(workdir)} \\
  {shlex.quote(sif_path)} \\
  {command}
"""


def parse_accounting(output: str, *, job_id: str, run_id: str, partition: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return {
            "run_id": run_id, "backend": "slurm", "backend_job_id": job_id,
            "state": "UNKNOWN", "raw_state": "UNKNOWN", "partition": partition,
            "elapsed": None, "exit_code": None,
        }
    fields = lines[0].split("|") + [""] * 6
    raw = fields[3].split()[0].rstrip("+")
    return {
        "run_id": run_id, "backend": "slurm", "backend_job_id": job_id,
        "state": normalize_state(raw, fields[5]), "raw_state": raw,
        "partition": fields[2] or partition, "elapsed": fields[4] or None,
        "exit_code": fields[5] or None,
    }


class WydSlurmBackend:
    kind = "slurm"
    # Keep multiplexing within one controller process, but never share the
    # socket startup/ownership lifecycle with a concurrent controller.
    ssh_control_path = f"/tmp/experimentctl-{os.getuid()}-{os.getpid()}-%C"

    def __init__(self, services: BackendServices):
        self.s = services

    @property
    def ssh_bin(self) -> str:
        return os.environ.get("EXPERIMENTCTL_SSH_BIN", "ssh")

    @property
    def rsync_bin(self) -> str:
        return os.environ.get("EXPERIMENTCTL_RSYNC_BIN", "rsync")

    def ssh_transport(self) -> str:
        return shlex.join([
            self.ssh_bin, "-o", "ControlMaster=auto", "-o", "ControlPersist=900",
            "-o", f"ControlPath={self.ssh_control_path}",
        ])

    def remote_exec(self, alias: str, command: str, *, check: bool = True):
        return self.s.run_command(
            [
                self.ssh_bin, "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
                "-o", "ControlPersist=900", "-o", f"ControlPath={self.ssh_control_path}",
                alias, command,
            ],
            check=check,
        )

    def _remote_query(self, alias: str, command: str, *, operation: str):
        """Run a read-only query and reject unavailable remote evidence."""
        result = self.remote_exec(alias, command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"{operation} failed; remote scheduler/storage evidence is unavailable"
            )
        return result

    def _remote_predicate(
        self, alias: str, command: str, *, operation: str
    ) -> bool:
        """Return a remote predicate, distinguishing false from transport failure."""
        result = self.remote_exec(alias, command, check=False)
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise RuntimeError(
            f"{operation} failed; remote scheduler/storage evidence is unavailable"
        )

    def validate(self, run: dict[str, Any]) -> None:
        backend, storage = run["backend"], run["storage"]
        required = {
            "ssh_alias", "partition", "account", "qos", "gres", "time",
            "source_dir", "sif_path", "mount_root",
        }
        missing = sorted(key for key in required if not backend.get(key))
        if missing:
            raise ValueError(f"run {run['run_id']} backend is missing: {missing}")
        gres_match = re.fullmatch(
            r"gpu:[A-Za-z0-9_-]+:([1-9][0-9]*)", str(backend["gres"])
        )
        if not gres_match:
            raise ValueError(f"run {run['run_id']} has invalid Slurm gres: {backend['gres']!r}")
        resources = run.get("resources", {})
        declared_gpus = resources.get("gpus", 1)
        declared_nodes = resources.get("nodes", 1)
        if (
            isinstance(declared_gpus, bool)
            or not isinstance(declared_gpus, int)
            or declared_gpus < 1
        ):
            raise ValueError(f"run {run['run_id']} resources.gpus must be a positive integer")
        if declared_gpus != int(gres_match.group(1)):
            raise ValueError(
                f"run {run['run_id']} resources.gpus={declared_gpus} does not match "
                f"backend.gres={backend['gres']!r}"
            )
        if (
            isinstance(declared_nodes, bool)
            or not isinstance(declared_nodes, int)
            or declared_nodes != 1
        ):
            raise ValueError(
                f"run {run['run_id']} Slurm backend currently requires resources.nodes=1"
            )
        if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", str(run["image_id"])):
            raise ValueError(f"run {run['run_id']} Slurm image_id must be a SIF sha256 digest")
        for field in ("partition", "account", "qos"):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(backend[field])):
                raise ValueError(f"run {run['run_id']} has invalid Slurm {field}: {backend[field]!r}")
        mount_root = Path(str(backend["mount_root"]))
        if not mount_root.is_absolute():
            raise ValueError(f"run {run['run_id']} backend.mount_root must be an absolute path")
        for field in ("apptainer_cache_dir", "apptainer_tmp_dir"):
            value = backend.get(field)
            if value is not None and not Path(str(value)).is_absolute():
                raise ValueError(f"run {run['run_id']} backend.{field} must be an absolute path")
        for field, value in (
            ("storage.run_dir", storage["run_dir"]),
            ("backend.source_dir", backend["source_dir"]),
            ("backend.sif_path", backend["sif_path"]),
            *((f"storage.{key}", value) for key, value in storage.items() if key.endswith(("_root", "_home", "_cache"))),
        ):
            path = Path(str(value))
            if not path.is_absolute() or not path.is_relative_to(mount_root):
                raise ValueError(f"run {run['run_id']} {field} must be under declared mount_root {mount_root}")

    def environment(self, campaign, run, source_id, attempt_id) -> dict[str, str]:
        return {"QUOTA_TYPE": "normal"}

    def preflight(self, run: dict[str, Any], *, scope: str) -> PreflightReport:
        """Check local transport tools plus live Slurm/storage compatibility."""
        if scope not in {"stage", "submit", "observe"}:
            raise ValueError(f"unsupported preflight scope: {scope}")
        checks: list[PreflightCheck] = []
        tool_commands = [("ssh-cli", [self.ssh_bin, "-V"])]
        if scope == "stage":
            tool_commands.append(("rsync-cli", [self.rsync_bin, "--version"]))
        for name, command in tool_commands:
            result = self.s.run_command(command, check=False)
            checks.append(PreflightCheck(
                name, "tool", "PASS" if result.returncode == 0 else "FAIL",
                f"{name} is executable" if result.returncode == 0 else f"{name} is unavailable",
            ))
        if any(check.status == "FAIL" for check in checks):
            return PreflightReport(self.kind, scope, tuple(checks))
        try:
            live = (
                self.validate_live(run) if scope == "submit"
                else self.validate_control(run)
            )
        except subprocess.CalledProcessError:
            checks.append(PreflightCheck(
                "slurm-access", "transport", "FAIL",
                "SSH transport or remote Slurm query failed",
            ))
        except RuntimeError as error:
            resource_failure = "partition" in str(error).lower() or "gres" in str(error).lower()
            checks.append(PreflightCheck(
                "slurm-access", "resource" if resource_failure else "authorization", "FAIL",
                "requested partition/GPU is unavailable" if resource_failure
                else "Slurm account/QOS association is unavailable",
            ))
        else:
            checks.append(PreflightCheck(
                "slurm-access", "resource" if scope == "submit" else "authorization", "PASS",
                f"partition {live['partition']} exposes the requested GPU type"
                if scope == "submit" else "Slurm control query is permitted",
            ))
            if scope in {"stage", "submit"}:
                backend = run["backend"]
                storage = self.remote_exec(
                    backend["ssh_alias"],
                    (
                        "command -v apptainer >/dev/null && " if scope == "submit" else ""
                    ) + f"test -d {shlex.quote(str(backend['mount_root']))}",
                    check=False,
                )
                checks.append(PreflightCheck(
                    "runtime-storage", "storage",
                    "PASS" if storage.returncode == 0 else "FAIL",
                    ("Apptainer and mount root are available" if scope == "submit"
                     else "mount root is available") if storage.returncode == 0
                    else ("Apptainer or mount root is unavailable" if scope == "submit"
                          else "mount root is unavailable"),
                ))
        return PreflightReport(self.kind, scope, tuple(checks))

    def submission_request(self, campaign, run, attempt_id) -> dict[str, Any]:
        return {"scheduler_name": scheduler_job_name(str(run["run_id"]), attempt_id)}

    def _matching_jobs(self, run, intent, attempt_id) -> list[str]:
        backend = run["backend"]
        token = str(intent["submission_token"])
        expected_name = scheduler_job_name(str(run["run_id"]), attempt_id)
        queue = self._remote_query(
            backend["ssh_alias"], "squeue -u $(id -un) -h -o '%i|%j|%k'",
            operation="Slurm queue identity query",
        )
        matches: set[str] = set()
        for line in queue.stdout.splitlines():
            if not line.strip():
                continue
            fields = line.split("|", 2)
            if len(fields) != 3:
                raise RuntimeError("Slurm queue identity query returned malformed evidence")
            if fields[1] == expected_name or fields[2] == token:
                if not fields[0].isdigit():
                    raise RuntimeError("Slurm queue identity query returned an invalid job ID")
                matches.add(fields[0])
        accounting = self._remote_query(
            backend["ssh_alias"],
            "sacct -S now-30days -u $(id -un) -X -n -P -o JobIDRaw,JobName",
            operation="Slurm accounting identity query",
        )
        for line in accounting.stdout.splitlines():
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) != 2:
                raise RuntimeError("Slurm accounting identity query returned malformed evidence")
            if fields[1] == expected_name:
                if not fields[0].isdigit():
                    raise RuntimeError("Slurm accounting identity query returned an invalid job ID")
                matches.add(fields[0])
        return sorted(matches, key=lambda value: (len(value), value))

    def recover_submission(self, run, intent, attempt_id) -> str | None:
        matches = self._matching_jobs(run, intent, attempt_id)
        if len(matches) > 1:
            raise RuntimeError(
                f"ambiguous scheduler identity: {len(matches)} jobs match this attempt"
            )
        return matches[0] if matches else None

    def identity(self, campaign, run, attempt_id) -> IdentityReport:
        token = f"{campaign['campaign']}/{run['run_id']}/{attempt_id}"
        matches = self._matching_jobs(
            run, {"submission_token": token}, attempt_id
        )
        manifest_exists = self._remote_predicate(
            run["backend"]["ssh_alias"],
            f"test -e {shlex.quote(str(run['storage']['run_dir']))}/manifest.yaml",
            operation="remote manifest identity probe",
        )
        manifest_matches: bool | None = None
        if manifest_exists:
            local_manifest = self.s.local_run_dir(campaign, run) / "manifest.yaml"
            if local_manifest.is_file():
                expected_sha = hashlib.sha256(local_manifest.read_bytes()).hexdigest()
                remote_manifest = f"{str(run['storage']['run_dir'])}/manifest.yaml"
                digest = self._remote_query(
                    run["backend"]["ssh_alias"],
                    f"{shlex.join(['sha256sum', '--', remote_manifest])} | "
                    "awk '{print $1}'",
                    operation="remote manifest digest probe",
                )
                fields = digest.stdout.strip().splitlines()
                if len(fields) != 1 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
                    raise RuntimeError(
                        "remote manifest digest probe returned malformed evidence"
                    )
                manifest_matches = fields[0].lower() == expected_sha
        return IdentityReport(
            available=not matches and not manifest_exists,
            ambiguous=len(matches) > 1,
            scheduler_job_ids=tuple(matches),
            remote_manifest_exists=manifest_exists,
            remote_manifest_matches=manifest_matches,
        )

    def verify_assets(self, run, probes) -> dict[str, Any]:
        missing = []
        alias = str(run["backend"]["ssh_alias"])
        for probe in probes:
            predicate = "-s" if probe.file else "-d"
            exists = self._remote_predicate(
                alias, shlex.join(["test", predicate, probe.path]),
                operation=f"required asset probe for {probe.path}",
            )
            if not exists:
                missing.append({**probe.requirement.__dict__, "path": probe.path})
        return {"missing": missing, "verification": "remote-ssh", "verified_on": alias}

    def stage(self, campaign, run, source_id, source_bundle) -> bool:
        backend = run["backend"]
        expected_suffix = f"/sources/{source_id}"
        if not str(backend["source_dir"]).endswith(expected_suffix):
            raise ValueError(f"source_dir must end with {expected_suffix}")
        source_marker = f"{backend['source_dir']}/.source-complete"
        self.remote_exec(
            backend["ssh_alias"],
            shlex.join(["mkdir", "-p", backend["source_dir"], str(Path(run["storage"]["run_dir"]).parent)]),
        )
        staged = self._remote_predicate(
            backend["ssh_alias"], f"test -f {shlex.quote(source_marker)}",
            operation="staged source marker probe",
        )
        if not staged:
            transport = self.ssh_transport()
            command = [self.rsync_bin, "-a", "--delete", "-e", transport]
            command.extend(
                argument for pattern in source_bundle.excludes
                for argument in ("--exclude", pattern)
            )
            command.extend([
                f"{source_bundle.root}/",
                f"{backend['ssh_alias']}:{backend['source_dir']}/",
            ])
            self.s.run_command(command)
            self.remote_exec(backend["ssh_alias"], shlex.join(["touch", source_marker]))
        expected_image = str(run["image_id"])
        expected_sha = expected_image.removeprefix("sha256:")
        marker = f"{backend['sif_path']}.sha256-{expected_sha}.verified"
        valid = self._remote_predicate(
            backend["ssh_alias"],
            f"test -s {shlex.quote(backend['sif_path'])} -a -f {shlex.quote(marker)}",
            operation="verified SIF marker probe",
        )
        if not valid:
            verify = self.remote_exec(
                backend["ssh_alias"],
                f"test -s {shlex.quote(backend['sif_path'])} && sha256sum {shlex.quote(backend['sif_path'])}",
            )
            actual_sha = verify.stdout.split()[0]
            if expected_image.startswith("sha256:") and actual_sha != expected_sha:
                raise ValueError(f"SIF checksum mismatch: expected {expected_image}, got sha256:{actual_sha}")
            self.remote_exec(backend["ssh_alias"], shlex.join(["touch", marker]))
        for required_path in source_bundle.required_paths:
            relative = Path(required_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"required source path must be relative: {required_path}")
            staged_path = str(Path(backend["source_dir"]) / relative)
            exists = self._remote_predicate(
                backend["ssh_alias"],
                shlex.join(["test", "-s", staged_path]),
                operation=f"required staged source probe for {required_path}",
            )
            if not exists:
                raise RuntimeError(
                    f"staged source is missing required project path: {required_path}"
                )
        return True

    def render(self, manifest: dict[str, Any]) -> str:
        return render_job(manifest)

    def validate_live(self, run: dict[str, Any]) -> dict[str, str]:
        backend = run["backend"]
        partition = backend["partition"]
        expected_gpu = backend["gres"].split(":", 2)[1]
        query = (
            f"sinfo -h -p {shlex.quote(partition)} -o '%P|%a|%l|%G'; "
            "sacctmgr -n -P show assoc where user=$(id -un) format=User,Account,Partition,QOS,DefaultQOS"
        )
        result = self.remote_exec(backend["ssh_alias"], query)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        partition_lines = [line for line in lines if line.split("|", 1)[0].rstrip("*") == partition]
        if not partition_lines:
            raise RuntimeError(f"Slurm partition is not currently visible: {partition}")
        fields = partition_lines[0].split("|")
        if len(fields) < 4 or fields[1] != "up" or f"gpu:{expected_gpu}:" not in fields[3]:
            raise RuntimeError(f"Slurm partition/GRES is not currently usable: {partition}/{backend['gres']}")
        associations = [line for line in lines if line not in partition_lines and "|" in line]
        if not any(backend["account"] in line.split("|") and backend["qos"] in line.split("|") for line in associations):
            raise RuntimeError(f"Slurm association does not expose account={backend['account']} qos={backend['qos']}")
        return {"partition": partition, "availability": fields[1], "gres": fields[3]}

    def validate_control(self, run: dict[str, Any]) -> dict[str, str]:
        """Verify SSH and a minimal user-scoped Slurm control query."""
        result = self.remote_exec(
            run["backend"]["ssh_alias"], "squeue -u $(id -un) -h -o '%i'"
        )
        return {"query": "squeue", "output": "nonempty" if result.stdout.strip() else "empty"}

    def submit(self, campaign, run, manifest, *, dry_run: bool) -> str:
        local_dir = self.s.local_run_dir(campaign, run)
        script_path = local_dir / "attempts" / manifest["attempt_id"] / "job.sbatch"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(self.render(manifest), encoding="utf-8")
        if dry_run:
            return "DRY_RUN"
        backend = run["backend"]
        self.validate_live(run)
        remote_script = f"{run['storage']['run_dir']}/controller-{manifest['attempt_id']}.sbatch"
        claim_dir = f"{run['storage']['run_dir']}/.submission-{manifest['attempt_id']}"
        claim = self.remote_exec(
            backend["ssh_alias"],
            f"{shlex.join(['mkdir', '-p', run['storage']['run_dir']])} && "
            f"{shlex.join(['mkdir', claim_dir])}",
            check=False,
        )
        if claim.returncode != 0:
            if claim.returncode != 1:
                raise RuntimeError(
                    "remote submission claim failed; scheduler/storage state is unknown"
                )
            raise FileExistsError(
                "remote submission claim already exists; reconcile before retrying"
            )
        transport = self.ssh_transport()
        self.s.run_command([
            self.rsync_bin, "-a", "-e", transport,
            str(local_dir / "manifest.yaml"),
            f"{backend['ssh_alias']}:{run['storage']['run_dir']}/manifest.yaml",
        ])
        self.s.run_command([self.rsync_bin, "-a", "-e", transport, str(script_path), f"{backend['ssh_alias']}:{remote_script}"])
        result = self.remote_exec(backend["ssh_alias"], f"sbatch --parsable {shlex.quote(remote_script)}")
        job_id = result.stdout.strip().split(";", 1)[0]
        if not re.fullmatch(r"\d+", job_id):
            raise ValueError(f"unexpected sbatch response: {result.stdout!r}")
        return job_id

    def status(self, campaign, run) -> dict[str, Any]:
        record = self.s.backend_record(campaign, run)
        backend, job_id = run["backend"], str(record["backend_job_id"])
        result = self._remote_query(
            backend["ssh_alias"],
            f"sacct -j {shlex.quote(job_id)} -X -n -P -o JobID,JobName,Partition,State,Elapsed,ExitCode",
            operation=f"Slurm accounting status query for job {job_id}",
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            queue = self._remote_query(
                backend["ssh_alias"],
                f"squeue -j {shlex.quote(job_id)} -h -o '%i|%j|%P|%T|%M|0:0'",
                operation=f"Slurm queue status query for job {job_id}",
            )
            lines = [line for line in queue.stdout.splitlines() if line.strip()]
        status = parse_accounting("\n".join(lines), job_id=job_id, run_id=run["run_id"], partition=backend["partition"])
        raw = status["raw_state"]
        status["failure_class"] = (
            "preemption" if raw == "PREEMPTED" else
            "scheduler" if raw in {"NODE_FAIL", "BOOT_FAIL"} else
            "resource" if raw in {"OUT_OF_MEMORY", "TIMEOUT"} else None
        )
        return status

    def cancel(self, campaign, run) -> dict[str, Any]:
        current = self.status(campaign, run)
        if current["state"] in {"SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"}:
            return current
        self.remote_exec(run["backend"]["ssh_alias"], f"scancel {shlex.quote(str(current['backend_job_id']))}")
        return self.status(campaign, run)

    def collect(self, campaign, run) -> dict[str, Any]:
        backend = run["backend"]
        mirror = self.s.local_run_dir(campaign, run) / "collected_run"
        mirror.mkdir(parents=True, exist_ok=True)
        transport = self.ssh_transport()
        self.s.run_command(
            [self.rsync_bin, "-a", "--delete", "-e", transport,
             "--include=*/", "--include=manifest.yaml", "--include=status.json",
             "--include=backend.json", "--include=events.jsonl",
             "--include=train_metrics.jsonl", "--include=metrics.jsonl",
             "--include=all_generated_*.jsonl",
             "--include=all_token_reconstructed_*.jsonl", "--exclude=*",
             f"{backend['ssh_alias']}:{run['storage']['run_dir']}/", f"{mirror}/"]
        )
        summary = self.s.summarize_run(campaign, mirror)
        run_dir = str(run["storage"]["run_dir"])
        checkpoint_probe = self._remote_query(
            backend["ssh_alias"],
            checkpoint_probe_command(run_dir),
            operation="completed checkpoint probe",
        )
        selected = select_latest_checkpoint_name(checkpoint_probe.stdout.splitlines())
        if selected:
            name, step = selected
            summary["latest_completed_checkpoint"] = f"{run_dir}/{name}"
            summary["latest_completed_checkpoint_step"] = step
        summary["collected_from"] = run["storage"]["run_dir"]
        summary["run_dir"] = run["storage"]["run_dir"]
        diagnostics = self.logs(campaign, run, tail=80)
        summary["process_evidence"] = {
            "observed": bool(diagnostics["stdout"] or diagnostics["stderr"]),
            "sources": diagnostics["sources"],
            "stdout_tail": diagnostics["stdout"],
            "stderr_tail": diagnostics["stderr"],
        }
        return summary

    def logs(self, campaign, run, *, tail: int) -> dict[str, Any]:
        if not 1 <= tail <= 10000:
            raise ValueError("tail must be between 1 and 10000")
        record = self.s.backend_record(campaign, run)
        attempt_id = str(record["attempt_id"])
        job_id = str(record["backend_job_id"])
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError("Slurm backend job ID must be numeric")
        run_dir = str(run["storage"]["run_dir"])
        attempt_dir = f"{run['storage']['run_dir']}/attempts/{attempt_id}"
        streams: dict[str, list[str]] = {}
        sources: dict[str, str | None] = {}
        for stream, suffix in (("stdout", "out"), ("stderr", "err")):
            paths = [
                f"{attempt_dir}/{stream}.log",
                f"{run_dir}/{stream}.log",
                f"{attempt_dir}/slurm-{job_id}.{suffix}",
                f"{run_dir}/slurm-{job_id}.{suffix}",
            ]
            result = self.remote_exec(
                run["backend"]["ssh_alias"],
                log_probe_command(paths, tail=tail),
                check=False,
            )
            if result.returncode not in {0, 1}:
                raise RuntimeError(
                    f"{stream} log probe failed; remote storage evidence is unavailable"
                )
            normalized = [
                redact_line(line) for line in result.stdout.replace("\r", "\n").splitlines()
                if line.strip()
            ]
            if result.returncode == 0 and normalized:
                sources[stream] = normalized[0]
                streams[stream] = normalized[1:][-tail:]
            else:
                sources[stream] = None
                streams[stream] = []
        return {
            "run_id": run["run_id"], "backend": "slurm",
            "backend_job_id": record["backend_job_id"], "attempt_id": attempt_id,
            "tail": tail, "sources": sources, **streams,
        }
