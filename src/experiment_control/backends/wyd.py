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
from ..safe_sco import redact_line


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
    execution = manifest.get("execution", {})
    container_path = str(execution.get("source_mount", "/workspace"))
    workdir = str(execution.get("workdir", container_path))
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
    ssh_control_path = "/tmp/experimentctl-%C"

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

    def validate(self, run: dict[str, Any]) -> None:
        backend, storage = run["backend"], run["storage"]
        required = {
            "ssh_alias", "partition", "account", "qos", "gres", "time",
            "source_dir", "sif_path", "mount_root",
        }
        missing = sorted(key for key in required if not backend.get(key))
        if missing:
            raise ValueError(f"run {run['run_id']} backend is missing: {missing}")
        if not re.fullmatch(r"gpu:[A-Za-z0-9_-]+:[1-9][0-9]*", str(backend["gres"])):
            raise ValueError(f"run {run['run_id']} has invalid Slurm gres: {backend['gres']!r}")
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
        checks: list[PreflightCheck] = []
        for name, command in (
            ("ssh-cli", [self.ssh_bin, "-V"]),
            ("rsync-cli", [self.rsync_bin, "--version"]),
        ):
            result = self.s.run_command(command, check=False)
            checks.append(PreflightCheck(
                name, "tool", "PASS" if result.returncode == 0 else "FAIL",
                f"{name} is executable" if result.returncode == 0 else f"{name} is unavailable",
            ))
        if any(check.status == "FAIL" for check in checks):
            return PreflightReport(self.kind, scope, tuple(checks))
        try:
            live = self.validate_live(run)
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
                "slurm-access", "resource", "PASS",
                f"partition {live['partition']} exposes the requested GPU type",
            ))
            backend = run["backend"]
            storage = self.remote_exec(
                backend["ssh_alias"],
                f"command -v apptainer >/dev/null && test -d {shlex.quote(str(backend['mount_root']))}",
                check=False,
            )
            checks.append(PreflightCheck(
                "runtime-storage", "storage",
                "PASS" if storage.returncode == 0 else "FAIL",
                "Apptainer and mount root are available" if storage.returncode == 0
                else "Apptainer or mount root is unavailable",
            ))
        return PreflightReport(self.kind, scope, tuple(checks))

    def submission_request(self, campaign, run, attempt_id) -> dict[str, Any]:
        return {"scheduler_name": scheduler_job_name(str(run["run_id"]), attempt_id)}

    def recover_submission(self, run, intent, attempt_id) -> str | None:
        backend = run["backend"]
        token = str(intent["submission_token"])
        expected_name = scheduler_job_name(str(run["run_id"]), attempt_id)
        result = self.remote_exec(
            backend["ssh_alias"], "squeue -u $(id -un) -h -o '%i|%j|%k'", check=False
        )
        matches = []
        for line in result.stdout.splitlines():
            fields = line.split("|", 2)
            if len(fields) == 3 and (fields[1] == expected_name or fields[2] == token):
                matches.append(fields[0])
        if not matches:
            accounting = self.remote_exec(
                backend["ssh_alias"],
                "sacct -S now-7days -u $(id -un) -X -n -P -o JobIDRaw,JobName",
                check=False,
            )
            matches = [
                line.split("|", 1)[0] for line in accounting.stdout.splitlines()
                if line.endswith(f"|{expected_name}") and line.split("|", 1)[0].isdigit()
            ]
        return matches[0] if len(matches) == 1 else None

    def verify_assets(self, run, probes) -> dict[str, Any]:
        missing = []
        alias = str(run["backend"]["ssh_alias"])
        for probe in probes:
            predicate = "-s" if probe.file else "-d"
            if self.remote_exec(alias, shlex.join(["test", predicate, probe.path]), check=False).returncode:
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
        staged = self.remote_exec(
            backend["ssh_alias"], f"test -f {shlex.quote(source_marker)}", check=False
        ).returncode == 0
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
        valid = self.remote_exec(
            backend["ssh_alias"],
            f"test -s {shlex.quote(backend['sif_path'])} -a -f {shlex.quote(marker)}",
            check=False,
        ).returncode == 0
        if not valid:
            verify = self.remote_exec(
                backend["ssh_alias"],
                f"test -s {shlex.quote(backend['sif_path'])} && sha256sum {shlex.quote(backend['sif_path'])}",
            )
            actual_sha = verify.stdout.split()[0]
            if expected_image.startswith("sha256:") and actual_sha != expected_sha:
                raise ValueError(f"SIF checksum mismatch: expected {expected_image}, got sha256:{actual_sha}")
            self.remote_exec(backend["ssh_alias"], shlex.join(["touch", marker]))
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
        self.remote_exec(backend["ssh_alias"], shlex.join(["mkdir", "-p", run["storage"]["run_dir"]]))
        transport = self.ssh_transport()
        self.s.run_command([self.rsync_bin, "-a", "-e", transport, str(script_path), f"{backend['ssh_alias']}:{remote_script}"])
        result = self.remote_exec(backend["ssh_alias"], f"sbatch --parsable {shlex.quote(remote_script)}")
        job_id = result.stdout.strip().split(";", 1)[0]
        if not re.fullmatch(r"\d+", job_id):
            raise ValueError(f"unexpected sbatch response: {result.stdout!r}")
        return job_id

    def status(self, campaign, run) -> dict[str, Any]:
        record = self.s.backend_record(campaign, run)
        backend, job_id = run["backend"], str(record["backend_job_id"])
        result = self.remote_exec(
            backend["ssh_alias"],
            f"sacct -j {shlex.quote(job_id)} -X -n -P -o JobID,JobName,Partition,State,Elapsed,ExitCode",
            check=False,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            queue = self.remote_exec(
                backend["ssh_alias"], f"squeue -j {shlex.quote(job_id)} -h -o '%i|%j|%P|%T|%M|0:0'", check=False
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
             "--include=backend.json", "--include=train_metrics.jsonl",
             "--include=metrics.jsonl", "--exclude=*",
             f"{backend['ssh_alias']}:{run['storage']['run_dir']}/", f"{mirror}/"]
        )
        summary = self.s.summarize_run(campaign, mirror)
        summary["collected_from"] = run["storage"]["run_dir"]
        summary["run_dir"] = run["storage"]["run_dir"]
        return summary

    def logs(self, campaign, run, *, tail: int) -> dict[str, Any]:
        record = self.s.backend_record(campaign, run)
        attempt_id = str(record["attempt_id"])
        attempt_dir = f"{run['storage']['run_dir']}/attempts/{attempt_id}"
        streams: dict[str, list[str]] = {}
        for stream in ("stdout", "stderr"):
            path = f"{attempt_dir}/{stream}.log"
            result = self.remote_exec(
                run["backend"]["ssh_alias"],
                f"test -f {shlex.quote(path)} && tail -n {tail} {shlex.quote(path)}",
                check=False,
            )
            normalized = [
                redact_line(line) for line in result.stdout.replace("\r", "\n").splitlines()
                if line.strip()
            ]
            streams[stream] = normalized[-tail:]
        return {
            "run_id": run["run_id"], "backend": "slurm",
            "backend_job_id": record["backend_job_id"], "attempt_id": attempt_id,
            "tail": tail, **streams,
        }
