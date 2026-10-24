"""Personal OSS dataset originals; only bounded streams and metadata on the API host."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import io
from functools import wraps
from pathlib import PurePosixPath
import re
import tarfile

from ..application_errors import ApplicationError
from ..archive_limits import exceeds
from ..projects.source_imports import source_lock
from ..results.oss_archive import client, DigestReader
from ..storage import utc_now


def safe_oss(call):
    @wraps(call)
    def invoke(*args,**kwargs):
        import alibabacloud_oss_v2 as oss
        try:
            return call(*args,**kwargs)
        except oss.exceptions.OperationError as error:
            from ..results.oss_archive import failure
            code=failure(error).get("provider_code","OSS_REQUEST_FAILED")
            raise ApplicationError("OSS dataset request failed: "+code,status_code=503,code="OSS_DATASET_REQUEST_FAILED") from None
        except (tarfile.TarError,EOFError):
            raise ApplicationError("OSS dataset is not a valid complete tar archive",status_code=422,code="OSS_DATASET_ARCHIVE_INVALID") from None
    return invoke


def settings(config):
    value = config.get('oss_datasets')
    if (not isinstance(value, dict) or set(value) != {'bucket','region','user','credentials_file'}
            or not isinstance(value['bucket'], str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}[a-z0-9]',value['bucket'])
            or not isinstance(value['region'], str) or not re.fullmatch(r'[a-z][a-z0-9-]{1,63}',value['region'])
            or not isinstance(value['user'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',value['user'])
            or not isinstance(value['credentials_file'], str) or not value['credentials_file'].startswith('/')):
        raise ApplicationError('personal OSS datasets are not configured', status_code=503, code='OSS_DATASETS_DISABLED')
    return value


def object_scope(config, key):
    prefixes = ('users/' + config['user'] + '/datasets/','public/datasets/')
    if not isinstance(key,str):
        raise ValueError("invalid OSS dataset object key")
    path = PurePosixPath(key)
    if (not key.startswith(prefixes) or key.endswith('/') or path.as_posix()!=key
            or '..' in path.parts or '\\' in key or any(part.startswith('.') for part in path.parts)
            or any(ord(c)<32 for c in key)):
        raise ValueError('dataset object must be in the approved dataset prefixes')
    return {'bucket':config['bucket'],'region':config['region'],'user':config['user'],'object_key':key}


class ObjectBody(io.RawIOBase):
    """Adapt the SDK iterator without ever calling its unbounded read()."""
    def __init__(self, body):
        super().__init__()
        self.body, self.chunks, self.pending = body, iter(body.iter_bytes(block_size=1024**2)), b''

    def readable(self):
        return True

    def readinto(self, buffer):
        if not self.pending:
            self.pending = next(self.chunks, b'')
        count = min(len(buffer),len(self.pending))
        buffer[:count] = self.pending[:count]
        self.pending = self.pending[count:]
        return count

    def iter_chunks(self, chunk_size):
        yield from self.body.iter_bytes(block_size=chunk_size)

    def close(self):
        if not self.closed:
            self.body.close()
        super().close()


def inspect_object(destination, location, sha256, size, assets,progress=None):
    import alibabacloud_oss_v2 as oss
    from ml_exp_worker.data_input import relative_path
    response=destination.get_object(oss.GetObjectRequest(bucket=location['bucket'],key=location['object_key']))
    files=[]; names=set();expanded=0
    with closing(ObjectBody(response.body)) as raw, io.BufferedReader(raw,buffer_size=1024**2) as buffered:
        stream=DigestReader(buffered,progress or (lambda _:None),size,sha256)
        with tarfile.open(fileobj=stream,mode='r|*') as archive:
            for member in archive:
                if len(archive.members)>assets.file_limit*2:
                    raise ValueError('dataset archive contains too many members')
                path=relative_path(member.name).as_posix()
                if member.isdir():
                    continue
                if not member.isfile() or path in names or len(files)>=assets.file_limit or exceeds(member.size,assets.expanded_limit):
                    raise ValueError('dataset archive contains an unsupported or duplicate member')
                names.add(path); digest=hashlib.sha256(); count=0
                with archive.extractfile(member) as body:
                    while chunk:=body.read(1024**2):
                        digest.update(chunk);count+=len(chunk)
                files.append({'path':path,'bytes':count,'sha256':digest.hexdigest()});expanded+=count
                if exceeds(expanded,assets.expanded_limit):
                    raise ValueError('dataset expanded bytes exceed policy')
        while stream.read(1024**2):
            pass
    return sorted(assets.validate_inventory(files),key=lambda item:item['path'])


def locator(asset):
    if asset.get('remote_storage')=='oss':
        return asset['remote_locator']
    for value in asset.get('remote_locations',{}).values():
        if value.get('remote_storage')=='oss':
            return value['remote_locator']
    return None


class OSSDatasets:
    def __init__(self, assets):
        self.assets=assets
        self.config=settings(assets.objects.config)

    @safe_oss
    def register(self,project,target,key,sha256,size,*,files=None,progress=None):
        if target['backend']!='pai-dlc' or not re.fullmatch(r'[0-9a-f]{64}',sha256) or type(size) is not int or size<=0 or exceeds(size,self.assets.limit):
            raise ValueError('invalid OSS dataset identity or executor')
        location=object_scope(self.config,key)
        with source_lock(self.assets.root.parent/'oss-dataset-verification',project+'-'+sha256):
            try:
                previous=self.assets.read(project,'asset.'+sha256)
            except ApplicationError as error:
                if error.status_code!=404:
                    raise
            else:
                copy=previous.get('remote_locations',{}).get(target['storage_scope'])
                if copy is not None and copy.get('remote_storage')=='oss':
                    if copy['remote_locator']!=location or previous['archive_bytes']!=size or files is not None and previous['files']!=files:
                        raise ValueError('OSS dataset identity is already bound')
                    return {**previous,**copy}
            verified=inspect_object(client(self.config),location,sha256,size,self.assets,progress)
            if files is not None and verified!=files:
                raise ValueError('OSS dataset files differ from the upload inventory')
            value={'project':project,'asset_id':'asset.'+sha256,'sha256':sha256,'archive_bytes':size,
                   'bytes':sum(item['bytes'] for item in verified),'files':verified,'status':'READY','created_at':utc_now(),
                   'remote_storage':'oss','storage_scope':target['storage_scope'],'remote_locator':location,
                   'executor':target['executor'],'verification':'archive-and-files-sha256'}
            return self.assets.publish_remote(project,value)

    def location(self,asset):
        value=locator(asset)
        if value is None:
            raise ApplicationError('this asset has no OSS dataset copy',status_code=404,code='OSS_DATASET_UNAVAILABLE')
        expected=object_scope(self.config,value['object_key'])
        if value!=expected:
            raise ValueError('dataset OSS location differs from the configured personal scope')
        return value

    @safe_oss
    def ticket(self,asset):
        import alibabacloud_oss_v2 as oss
        value=self.location(asset)
        signed=client(self.config).presign(oss.GetObjectRequest(bucket=value['bucket'],key=value['object_key']),expires=timedelta(minutes=5))
        return {'transport':'oss-presigned-get','url':signed.url,'sha256':asset['sha256'],'bytes':asset['archive_bytes'],
                'asset_id':asset['asset_id'],'files':asset['files'],'acknowledgement_required':False,
                'expires_at':(datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat()}

    @safe_oss
    def archive(self,asset):
        import alibabacloud_oss_v2 as oss
        value=self.location(asset)
        return ObjectBody(client(self.config).get_object(oss.GetObjectRequest(bucket=value['bucket'],key=value['object_key'])).body)
