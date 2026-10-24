"""Targeted uploads preserve backend identity and require durable receipts."""
import hashlib
import io
import json
import tarfile

import pytest

from ml_exp_client.api import ClientError
from ml_exp_client.archives import data_archive, data_transport_identity
from ml_exp_client.assets import upload_asset_parts


def packed(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "values.bin").write_bytes(b"".join(hashlib.sha256(str(i).encode()).digest() for i in range(100)))
    stream = io.BytesIO()
    sha, size = data_archive(directory, stream)
    return stream, sha, size


class Destination:
    def __init__(self, sha, size, *, receipt="valid", executor="slurm-gpu"):
        self.sha, self.size, self.receipt, self.executor = sha, size, receipt, executor
        self.parts, self.calls = {}, []
        self.upload = "upload." + "a" * 64

    def call(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if path.endswith("asset-uploads") or path.endswith(self.upload):
            if "data" in kwargs:
                self.request = kwargs["data"]
            return {"upload_id": self.upload, "sha256": self.sha, "bytes": self.size, "part_bytes": 1024,
                    "part_count": (self.size + 1023) // 1024, "parts": self.parts.copy(), "status": "UPLOADING",
                    "binding": {"target": {"executor": self.executor, "backend": "slurm",
                        "storage_scope": "storage." + "1" * 64, "profile_sha256": "2" * 64}}}
        if "/parts/" in path:
            number = int(path.split("/parts/")[1].split("?")[0])
            body = kwargs["raw"]
            item = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
            self.parts[str(number)] = item
            result = {"contract": "upload-part.v1", "upload_id": self.upload, "number": number, **item, "durable": True}
            if self.receipt == "not-durable":
                result["durable"] = False
            elif self.receipt == "wrong-hash":
                result["sha256"] = "f" * 64
            return result
        assert path.endswith("/complete")
        return {"project": "demo", "asset_id": "asset." + self.sha, "sha256": self.sha,
                "archive_bytes": self.size, "status": "READY"}


def test_layer_bytes_and_file_identities_match_archive(tmp_path):
    stream, sha, size = packed(tmp_path)
    files, layer_sha = data_transport_identity(stream, size)
    payload = stream.read()
    header = tarfile.TarInfo("payload/dataset.tar")
    header.size, header.mode = size, 0o444
    layer = header.tobuf(format=tarfile.GNU_FORMAT) + payload + b"\0" * ((-size) % 512 + 1024)
    assert hashlib.sha256(layer).hexdigest() == layer_sha
    with tarfile.open(fileobj=io.BytesIO(layer)) as archive:
        assert archive.extractfile("payload/dataset.tar").read() == payload
    assert files == [{"path": "values.bin", "sha256": hashlib.sha256((tmp_path / "data/values.bin").read_bytes()).hexdigest(), "bytes": 3200}]
    assert hashlib.sha256(payload).hexdigest() == sha


def test_targeted_upload_confirms_and_resumes_only_missing_parts(tmp_path):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size)
    state = tmp_path / "state.json"
    result = upload_asset_parts(api, "demo", stream, sha, size, state, target={"executor": "slurm-gpu"})
    assert result["asset_id"] == "asset." + sha
    assert api.request["target"] == {"executor": "slurm-gpu"}
    assert api.request["files"][0]["bytes"] == 3200
    assert len(api.request["transport_sha256"]) == 64
    sent = sum("/parts/" in path for path, _ in api.calls)
    upload_asset_parts(api, "demo", stream, sha, size, state, target={"executor": "slurm-gpu"})
    assert sum("/parts/" in path for path, _ in api.calls) == sent
    assert json.loads(state.read_text())["target"] == {"executor": "slurm-gpu"}


@pytest.mark.parametrize("receipt", ["not-durable", "wrong-hash"])
def test_unconfirmed_part_never_completes(tmp_path, receipt):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size, receipt=receipt)
    with pytest.raises(ClientError, match="not confirmed"):
        upload_asset_parts(api, "demo", stream, sha, size, tmp_path / "state.json", target={"executor": "slurm-gpu"})
    assert not any(path.endswith("/complete") for path, _ in api.calls)


def test_wrong_destination_rejected_before_sending_bytes(tmp_path):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size, executor="other")
    with pytest.raises(ClientError, match="another executor"):
        upload_asset_parts(api, "demo", stream, sha, size, tmp_path / "state.json", target={"executor": "slurm-gpu"})
    assert not api.parts


def test_resume_cannot_change_destination(tmp_path):
    stream, sha, size = packed(tmp_path)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"target": {"executor": "other"}}))
    api = Destination(sha, size)
    with pytest.raises(ClientError, match="another executor"):
        upload_asset_parts(api, "demo", stream, sha, size, state, target={"executor": "slurm-gpu"})
    assert not api.calls


def test_resume_uses_original_session_and_freezes_full_public_target(tmp_path, monkeypatch):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size)
    state = tmp_path / 'state.json'
    upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor': 'slurm-gpu'})
    saved = json.loads(state.read_text())
    assert saved['target_binding'] == {'executor': 'slurm-gpu', 'backend': 'slurm',
        'storage_scope': 'storage.' + '1'*64, 'profile_sha256': '2'*64}
    api.calls.clear()
    monkeypatch.setattr('ml_exp_client.assets.data_transport_identity',
        lambda *args: pytest.fail('GET resume must not construct a new upload manifest'))
    upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor': 'slurm-gpu'})
    assert api.calls[0] == ('/api/projects/demo/asset-uploads/' + api.upload, {})
    assert not any(path.endswith('/asset-uploads') for path, _ in api.calls)


@pytest.mark.parametrize('field', ['upload_id', 'storage_scope', 'profile_sha256', 'backend'])
def test_resume_rejects_swapped_session_or_target_without_mutating_state_or_sending_bytes(tmp_path, field):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size)
    state = tmp_path / 'state.json'
    upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor': 'slurm-gpu'})
    before = state.read_bytes()
    original = api.call
    def substituted(path, **kwargs):
        value = original(path, **kwargs)
        if path.endswith(api.upload):
            if field == 'upload_id': value[field] = 'upload.' + 'b'*64
            elif field == 'storage_scope': value['binding']['target'][field] = 'storage.' + 'f'*64
            elif field == 'profile_sha256': value['binding']['target'][field] = 'f'*64
            else: value['binding']['target'][field] = 'sensecore'
        return value
    api.call = substituted
    api.calls.clear()
    with pytest.raises(ClientError, match='receipt|frozen target'):
        upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor': 'slurm-gpu'})
    assert len(api.calls) == 1 and state.read_bytes() == before


@pytest.mark.parametrize('binding', [None, {}, {'target': None},
    {'target': {'executor': 'slurm-gpu'}}, {'target': {'executor': 'slurm-gpu', 'backend': 'slurm',
        'storage_scope': 'unscoped', 'profile_sha256': '2'*64}}])
def test_malformed_public_target_is_rejected_before_any_part_upload(tmp_path, binding):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size)
    original = api.call
    def malformed(path, **kwargs):
        value = original(path, **kwargs)
        if path.endswith('/asset-uploads'): value['binding'] = binding
        return value
    api.call = malformed
    with pytest.raises(ClientError, match='executor storage'):
        upload_asset_parts(api, 'demo', stream, sha, size, tmp_path/'state.json', target={'executor': 'slurm-gpu'})
    assert not api.parts


def test_old_state_without_target_binding_learns_binding_via_exact_get(tmp_path):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size)
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'project':'demo','asset_id':'asset.'+sha,'archive_bytes':size,
        'upload_id':api.upload,'target':{'executor':'slurm-gpu'},'status':'UPLOADING'}))
    upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor':'slurm-gpu'})
    assert api.calls[0][0].endswith('/' + api.upload) and api.calls[0][1] == {}
    assert json.loads(state.read_text())['target_binding']['profile_sha256'] == '2'*64


def test_lost_local_state_posts_same_frozen_archive_identity_and_reuses_committed_parts(tmp_path):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size)
    state = tmp_path / 'state.json'
    upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor':'slurm-gpu'})
    state.unlink()
    api.calls.clear()
    upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor':'slurm-gpu'})
    assert api.calls[0][0].endswith('/asset-uploads') and 'data' in api.calls[0][1]
    assert not any('/parts/' in path for path, _ in api.calls)


@pytest.mark.parametrize('field,value', [('project','other'),('asset_id','asset.'+'f'*64),
    ('sha256','f'*64),('archive_bytes',17),('status','FAILED')])
@pytest.mark.parametrize('completed_get', [False, True])
def test_completion_receipt_must_bind_the_exact_uploaded_asset(tmp_path, field, value, completed_get):
    stream, sha, size = packed(tmp_path)
    api = Destination(sha, size)
    state = tmp_path / 'state.json'
    if completed_get:
        upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor':'slurm-gpu'})
    original = api.call
    def wrong_result(path, **kwargs):
        result = original(path, **kwargs)
        if completed_get and path.endswith(api.upload):
            asset = original(path+'/complete'); asset[field] = value
            result.update(status='COMPLETED',result=asset)
        elif not completed_get and path.endswith('/complete'):
            result[field] = value
        return result
    api.call = wrong_result
    with pytest.raises(ClientError, match='completed asset identity'):
        upload_asset_parts(api, 'demo', stream, sha, size, state, target={'executor':'slurm-gpu'})
