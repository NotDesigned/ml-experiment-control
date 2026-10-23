"""Resumable OSS multipart transport behind the existing bounded upload API."""
from __future__ import annotations

from contextlib import contextmanager,nullcontext
from datetime import datetime,timedelta,timezone
import hashlib
import json
import time
from pathlib import Path

from ..application_errors import ApplicationError
from ..projects.source_imports import source_lock
from ..storage import atomic_json,utc_now
from .multipart_upload import UploadStore,UPLOAD_ID
from .oss_datasets import OSSDatasets,client,object_scope,settings,safe_oss


class OSSDataUploads:
    remote=True
    part_size=staticmethod(UploadStore.part_size)

    def __init__(self,assets,target,files=None,transport_sha256=None):
        self.assets,self.target,self.files=assets,target,files
        self.config=settings(assets.objects.config)
        if target.get('oss_datasets')!=self.config:
            raise ApplicationError('OSS upload configuration changed',code='UPLOAD_TARGET_CHANGED')
        self.part_bytes=assets.objects.config.get('oss_dataset_part_bytes',16*1024**2)
        self.max_parts=10000
        if type(self.part_bytes) is not int or not 100*1024<=self.part_bytes<=64*1024**2:
            raise ApplicationError('invalid OSS dataset part size',status_code=503,code='OSS_DATASETS_DISABLED')
        self.root=assets.root.parent/'oss-data-uploads'

    @contextmanager
    def state(self,upload_id,binding,*,locked=True):
        if not UPLOAD_ID.fullmatch(upload_id):
            raise ValueError('invalid OSS upload identity')
        with source_lock(self.root,upload_id) if locked else nullcontext():
            path=self.root/(upload_id+'.json')
            value=json.loads(path.read_text()) if path.exists() else None
            if value is not None and value['binding']!=binding:
                raise ApplicationError('unknown OSS upload',status_code=404,code='UNKNOWN_UPLOAD')
            yield path,value

    @staticmethod
    def public(value):
        result={key:value[key] for key in ('upload_id','binding','sha256','bytes','part_bytes','part_count','status','expires_at','result','phase','diagnostic','verification_bytes_read','last_progress_at','object') if key in value}
        result['parts']={number:{key:item[key] for key in ('sha256','bytes')} for number,item in value['parts'].items()}
        return result

    @staticmethod
    def active(value):
        if value is None:
            raise ApplicationError('unknown OSS upload',status_code=404,code='UNKNOWN_UPLOAD')
        if value['status'] not in {'UPLOADING','FINALIZING'} or datetime.fromisoformat(value['expires_at'])<=datetime.now(timezone.utc):
            raise ApplicationError('OSS upload is not accepting data; inspect or abort the original session',code='UPLOAD_NOT_ACTIVE')

    @safe_oss
    def create(self,binding,sha,size,limit):
        import alibabacloud_oss_v2 as oss
        self.assets.validate_inventory(self.files or [])
        upload_id=UploadStore.identity(binding,sha,size)
        part_bytes=self.part_bytes
        if (size+part_bytes-1)//part_bytes>10000 or part_bytes<100*1024:
            raise ApplicationError('OSS requires at most 10000 parts, each non-final part at least 100 KiB',status_code=413,code='UPLOAD_LIMIT')
        with self.state(upload_id,binding) as (path,value):
            generation=value.get('generation',0)+1 if value else 1
            if value is not None and value['status']!='ABORTED':
                return self.public(value)
            key='/'.join(['users',self.config['user'],'datasets','ml-expd',binding['project'],upload_id+'-'+str(generation)+'.tar'])
            value={'upload_id':upload_id,'binding':binding,'sha256':sha,'bytes':size,'files':self.files,
                   'part_bytes':part_bytes,'part_count':(size+part_bytes-1)//part_bytes,'parts':{},'pending':{},
                   'status':'CREATING','phase':'INITIALIZING_OSS_UPLOAD','generation':generation,'created_at':utc_now(),'object':object_scope(self.config,key),
                   'expires_at':(datetime.now(timezone.utc)+timedelta(seconds=self.assets.objects.config.get('upload_session_seconds',86400))).isoformat()}
            atomic_json(path,value)
            # An unknown initiation is retained as CREATING; never create another
            # multipart session implicitly. Only explicit abort starts a generation.
            reply=client(self.config).initiate_multipart_upload(oss.InitiateMultipartUploadRequest(
                bucket=self.config['bucket'],key=key,forbid_overwrite=True,content_type='application/x-tar'))
            value.update(native_upload_id=reply.upload_id,status='UPLOADING',phase='UPLOADING_PARTS')
            atomic_json(path,value)
            return self.public(value)

    def read(self,upload_id,binding):
        # Atomic JSON snapshots stay readable during a long verification.
        with self.state(upload_id,binding,locked=False) as (_,value):
            if value is None:
                raise ApplicationError('unknown OSS upload',status_code=404,code='UNKNOWN_UPLOAD')
            return self.public(value)

    @safe_oss
    def part(self,upload_id,binding,number,temporary,sha,size):
        import alibabacloud_oss_v2 as oss
        with self.state(upload_id,binding) as (path,value):
            if value is None or value['status']!='COMPLETED':
                self.active(value)
            if self.part_size(value,number)!=size:
                raise ValueError('OSS part length differs')
            with Path(temporary).open('rb') as stream:
                body=stream.read(size+1)
            if len(body)!=size or hashlib.sha256(body).hexdigest()!=sha:
                raise ValueError('OSS upload part checksum differs')
            item={'sha256':sha,'bytes':size};key=str(number)
            previous=value['parts'].get(key)
            pending=value['pending'].get(key)
            if previous is not None:
                if any(previous[k]!=v for k,v in item.items()):
                    raise ValueError('OSS part is already sealed')
            elif value['status']=='COMPLETED':
                raise ValueError('completed OSS upload has no matching part')
            else:
                if pending is not None and pending!=item:
                    raise ValueError('pending OSS part differs')
                value['pending'][key]=item;atomic_json(path,value)
                reply=client(self.config).upload_part(oss.UploadPartRequest(bucket=self.config['bucket'],
                    key=value['object']['object_key'],upload_id=value['native_upload_id'],part_number=number+1,
                    content_length=size,body=body))
                value['parts'][key]={**item,'etag':reply.etag};value['pending'].pop(key);atomic_json(path,value)
            return {'contract':'upload-part.v1','upload_id':upload_id,'number':number,**item,'durable':True}

    @safe_oss
    def complete(self,upload_id,binding):
        import alibabacloud_oss_v2 as oss
        with self.state(upload_id,binding) as (path,value):
            if value is not None and value['status']=='COMPLETED':
                return self.assets.publish_remote(binding['project'],value['result'])
            self.active(value)
            if len(value['parts'])!=value['part_count']:
                raise ValueError('OSS upload has missing parts')
            destination=client(self.config)
            # Only a previously recorded FINALIZING operation may reconcile a
            # lost completion response against this exact object key.
            exists=False
            if value['status']=='FINALIZING':
                try:
                    destination.head_object(oss.HeadObjectRequest(bucket=self.config['bucket'],key=value['object']['object_key']))
                    exists=True
                except oss.exceptions.OperationError as error:
                    inner=error.unwrap()
                    if not isinstance(inner,oss.exceptions.ServiceError) or inner.code not in {'NoSuchKey','NotFound'}:
                        raise
            if not exists:
                value['status']='FINALIZING';atomic_json(path,value)
                destination.complete_multipart_upload(oss.CompleteMultipartUploadRequest(bucket=self.config['bucket'],
                    key=value['object']['object_key'],upload_id=value['native_upload_id'],forbid_overwrite=True,
                    complete_multipart_upload=oss.CompleteMultipartUpload(parts=[oss.UploadPart(part_number=i+1,
                        etag=value['parts'][str(i)]['etag']) for i in range(value['part_count'])])))
            value.update(phase='VERIFYING_OBJECT',diagnostic=None,verification_bytes_read=0,last_progress_at=utc_now());atomic_json(path,value)
            last_write=0.0
            def progress(size):
                nonlocal last_write
                now=time.monotonic()
                if now-last_write>=5 or size==value['bytes']:
                    value.update(verification_bytes_read=size,last_progress_at=utc_now());atomic_json(path,value);last_write=now
            try:
                result=OSSDatasets(self.assets).register(binding['project'],self.target,value['object']['object_key'],
                    value['sha256'],value['bytes'],files=value['files'],progress=progress)
            except (ValueError,ApplicationError) as error:
                value.update(phase='VERIFICATION_FAILED',diagnostic=error.code if isinstance(error,ApplicationError) else 'OSS_DATASET_VERIFICATION_FAILED')
                atomic_json(path,value)
                raise
            value.update(status='COMPLETED',phase='READY',diagnostic=None,result=result);atomic_json(path,value)
            return result

    @safe_oss
    def abort(self,upload_id,binding):
        import alibabacloud_oss_v2 as oss
        with self.state(upload_id,binding) as (path,value):
            if value is None:
                raise ApplicationError('unknown OSS upload',status_code=404,code='UNKNOWN_UPLOAD')
            if value['status']=='COMPLETED' or value['status']=='FINALIZING':
                raise ApplicationError('published or uncertain OSS data cannot be deleted by upload abort',code='UPLOAD_NOT_ACTIVE')
            if value['status']!='ABORTED' and value.get('native_upload_id'):
                client(self.config).abort_multipart_upload(oss.AbortMultipartUploadRequest(bucket=self.config['bucket'],
                    key=value['object']['object_key'],upload_id=value['native_upload_id']))
            value['status']='ABORTED';atomic_json(path,value)
            return self.public(value)
