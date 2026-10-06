"""Prepare data on NAS through the API, without replaying uncertain ACP requests."""
import json
import time

from .api import ClientError, save, segment


def deliver_data(client, project, asset_id, executor, state_path, seconds):
    binding = {"project": project, "asset_id": asset_id, "executor": executor}
    state = json.loads(state_path.read_text()) if state_path.exists() else binding
    if any(state.get(key) != value for key, value in binding.items()):
        raise ClientError("saved data delivery belongs to another asset or executor")
    if "delivery" not in state:
        state["delivery"] = client.call(f"/api/projects/{segment(project)}/assets/{segment(asset_id)}/deliveries/prepare",
                                       data={"executor": executor})
        save(state_path, state)
    endpoint = f"/api/projects/{segment(project)}/data-deliveries/{segment(state['delivery']['delivery_id'])}"
    value = client.call(endpoint)
    if value["status"] == "PREPARED":
        if state.get("requested"):
            raise ClientError("data-copy request outcome is uncertain; inspect the saved delivery, never resubmit")
        state["requested"] = True
        save(state_path, state)
        client.call(endpoint + "/execute", data={"confirmation": value["confirmation"]})
    deadline = time.monotonic() + seconds
    while True:
        value = client.call(endpoint)
        state["delivery"] = value
        save(state_path, state)
        if value["status"] == "READY":
            return value
        if value["status"] in {"FAILED", "RECONCILE_REQUIRED"}:
            raise ClientError("data copy requires inspection/reconciliation; no GPU Run was created")
        if time.monotonic() >= deadline:
            raise ClientError("data copy is still preparing; resume with the saved state, never resubmit")
        time.sleep(5)
