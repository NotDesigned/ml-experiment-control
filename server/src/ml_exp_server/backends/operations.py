"""Explicit server preparation/collection boundary beside core scheduling.

Core BackendRegistry remains responsible for scheduler capabilities, staging,
submit/status/cancel. These adapters own only the additional server data and
result operations, and never add a second operation journal.
"""
from __future__ import annotations

from pathlib import Path
import yaml

from experiment_control.backends.sensecore_rest import SenseCoreREST

from ..application_errors import ApplicationError
from .sensecore_preparation import SenseCoreScriptPreparation
from .states import TERMINAL
from .wyd import WydDataStager


class WydOperations:
    def prepare_asset(self, owner, project, preparation, value, selected, asset, observe):
        stager = WydDataStager(owner.runtime)
        marker = "wyd:" + asset
        verified = stager.check(project, asset, selected["profile"])
        if verified["status"] != "READY":
            if marker in value["delivery_requested"]:
                raise ApplicationError("saved WYD transfer is not verified; inspect shared storage, never replay", code="WYD_DATA_NOT_READY")
            value["delivery_requested"].append(marker)
            owner.update(project, preparation, delivery_requested=value["delivery_requested"])
            receipt = stager.publish(project, asset, selected["profile"])
        else:
            receipt = stager.receipt(project, asset, selected["profile"])
        value.setdefault("data_receipts", {})[asset] = receipt
        owner.update(project, preparation, data_receipts=value["data_receipts"])

    def prepare_script(self, owner, project, preparation, value, run):
        # The existing core Slurm stage executes the script in its SIF on the
        # connected gateway, preserving the original staging identity/receipt.
        return None

    def script_progress(self, owner, project, reference):
        return reference

    def confirm_terminal(self, owner, backend, name):
        owner.verify_wyd_terminal(backend, name)

    def collect(self, owner, identity, root, definition, transfer, operation, local):
        if local == root / "attempts" / identity[2] / "collected_run" / "attempts" / identity[2] / "outputs":
            # Generic mirrors may contain a filtered or partial copy. Read the
            # original shared outputs before publishing an independent recovery.
            local = None
        if local is None:
            owner.update(identity, status="COLLECTING", phase="READING_SHARED_STORAGE")
            local = root / "attempts" / identity[2] / "recovered_outputs"
            size = owner.wyd_output_bytes(definition["backend"], definition, identity[2])
            with owner.capacity().reserve(4 * size + 128 * 1024 ** 2) as reservation:
                owner.read_wyd_outputs(definition["backend"], definition, identity[2], local)
                owner.update(identity, status="COLLECTING", phase="VERIFYING_AND_PUBLISHING")
                owner.publish_local(identity, local, transfer, reservation=reservation)
        else:
            publish_existing(owner, identity, local, transfer)
        return True


class SenseCoreOperations:
    def prepare_asset(self, owner, project, preparation, value, selected, asset, observe):
        from ..data.data_delivery import DataDeliveryService
        delivery_service = DataDeliveryService(owner.runtime)
        delivery = (delivery_service.read(project, value["deliveries"][asset]) if asset in value["deliveries"]
                    else delivery_service.prepare(project, asset, selected["executor"]))
        value["deliveries"][asset] = delivery["delivery_id"]
        owner.update(project, preparation, deliveries=value["deliveries"])
        if delivery["status"] == "PREPARED" and delivery["delivery_id"] not in value["delivery_requested"]:
            value["delivery_requested"].append(delivery["delivery_id"])
            owner.update(project, preparation, delivery_requested=value["delivery_requested"])
            pending = delivery_service.begin(project, delivery["delivery_id"], delivery["confirmation"])
            if pending is not None:
                delivery_service.finish(pending)
        active = {"EXECUTING", "BUILDING_IMAGE", "IMAGE_READY", "SUBMITTING", "QUEUED", "COPYING"}
        delivery = delivery_service.read(project, delivery["delivery_id"])
        if delivery["status"] in active:
            delivery = observe(lambda: delivery_service.read(project, delivery["delivery_id"]), active)
        if delivery["status"] != "READY":
            raise ApplicationError("inspect the saved data delivery; uncertain copy jobs are never replayed", code="PREPARATION_DATA_NOT_READY")

    @staticmethod
    def scripts(owner):
        return SenseCoreScriptPreparation(owner.runtime.config.container_execution.artifact_store_file,
                                          owner.runtime.config.project_registry_root_path())

    def prepare_script(self, owner, project, preparation, value, run):
        if run["data_preparation_execution"]["phase"] != "before_job":
            return None
        if not owner.runtime.config.action_runtime.allow_scheduler_mutations:
            raise ApplicationError("CPU data preparation is disabled by scheduler policy", code="DATA_SCRIPT_DISABLED")
        owner.update(project, preparation, phase="PREPARING_DATA_SCRIPT", run=run)
        campaign = Path(owner.runtime.project(project).base_dir) / "experiments" / "campaigns" / (run["campaign"] + ".yaml")
        frozen = yaml.safe_load(campaign.read_text())["runs"][0]
        scripts = self.scripts(owner)
        reference = scripts.reference(frozen)
        previous = value.get("data_script")
        if previous is not None and previous["operation_id"] != reference["operation_id"]:
            raise ApplicationError("original CPU data preparation identity changed", code="DATA_SCRIPT_IDENTITY_CHANGED")
        owner.update(project, preparation, data_script=reference)
        owner.update(project, preparation, data_script=scripts.ensure(project, frozen))

    def script_progress(self, owner, project, reference):
        return self.scripts(owner).read(project, reference["operation_id"])

    def confirm_terminal(self, owner, backend, name):
        if SenseCoreREST.from_environment().describe(backend, name)["state"] not in TERMINAL:
            raise ApplicationError("provider Attempt is still running", code="ORIGINAL_JOB_NOT_TERMINAL")

    def collect(self, owner, identity, root, definition, transfer, operation, local):
        if local is not None:
            publish_existing(owner, identity, local, transfer)
            return True
        return owner.sensecore(identity, definition, transfer, operation, SenseCoreREST.from_environment())


def publish_existing(owner, identity, local, transfer):
    owner.update(identity, status="COLLECTING", phase="VERIFYING_AND_PUBLISHING")
    owner.publish_local(identity, local, transfer)


def backend_operations(kind, adapters=None):
    """A small injectable mapping, not plugin discovery or a second registry DB."""
    implementations = adapters if adapters is not None else {"slurm": WydOperations(), "sensecore": SenseCoreOperations()}
    try:
        return implementations[kind]
    except KeyError:
        raise ValueError("backend does not support server preparation or collection") from None
