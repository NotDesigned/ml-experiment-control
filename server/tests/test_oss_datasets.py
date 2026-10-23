"""Personal OSS datasets: immutable identities, resumable bounded I/O and CPFS use."""
from datetime import timedelta
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import alibabacloud_oss_v2 as oss
import pytest
import yaml

from ml_exp_server.application_errors import ApplicationError
from ml_exp_server.container_execution import ContainerExecutionService
from ml_exp_server.data import oss_datasets as d,oss_uploads as u
from ml_exp_server.data.data_assets import AssetStore
from ml_exp_server.data.remote_data_uploads import freeze_target,ManagedDataUploads,public_target
from ml_exp_server.data.locations import for_backend
from ml_exp_server.workers.data_input import deliver
from tests.test_sensecore_data_workflow import client,stored,custom_runtime
from tests.test_container_api import archive
from tests.test_oss_archive import service_error

S={'bucket':'ai-oss-h200-1020','region':'us-southeast-1','user':'owner','credentials_file':'/etc/private/oss.json'}
B={'kind':'pai-dlc','region':S['region'],'workspace_id':'253','quota_id':'quota-test','gpu_type':'H200',
   'storage_id':'cpfs-approved','filesystem_type':'cpfs','cpfs_uri':'bmcpfs://cpfs-approved.us-southeast-1.cpfs.aliyuncs.com/',
   'workspace_user':'owner','data_role_arn':'acs:ram::123:role/pai-data-owner'}


@pytest.fixture
def datasets(client,stored,monkeypatch):
    config,_=stored;value=json.loads(config.read_text());value.update(oss_datasets=S,upload_part_bytes=128*1024,oss_dataset_part_bytes=128*1024,max_asset_archive_bytes=None,max_asset_bytes=None);config.write_text(json.dumps(value))
    runtime=client.app.state.runtime
    profiles=Path(runtime.config.container_execution.profiles_file);value=yaml.safe_load(profiles.read_text());value['executors']['pai']={'backend':B,'storage_root':'/mnt/workspace/ml-expd','capacity':{'gpus':8,'cpus':216,'memory_gb':1800}};profiles.write_text(yaml.safe_dump(value))
    objects={};sessions={};calls=[]
    class Body:
        def __init__(self,data):self.stream=io.BytesIO(data)
        def iter_bytes(self,*,block_size):
            assert block_size<=1024**2
            while chunk:=self.stream.read(block_size):yield chunk
        def close(self):self.stream.close()
    class Cloud:
        def get_object(self,r):
            calls.append(('GET',r.key))
            if r.key not in objects:raise service_error('NoSuchKey')
            return SimpleNamespace(body=Body(objects[r.key]))
        def head_object(self,r):
            if r.key not in objects:raise service_error('NoSuchKey')
            return SimpleNamespace()
        def presign(self,r,*,expires):
            assert expires==timedelta(minutes=5)
            return SimpleNamespace(url='https://private.example/dataset?signed=private')
        def initiate_multipart_upload(self,r):
            calls.append(('CREATE',r.key));assert r.forbid_overwrite is True
            uid=str(len(sessions)+1);sessions[uid]={'key':r.key,'parts':{}}
            return SimpleNamespace(upload_id=uid)
        def upload_part(self,r):
            calls.append(('PART',r.part_number));assert 1<=r.part_number<=10000
            sessions[r.upload_id]['parts'][r.part_number]=r.body
            return SimpleNamespace(etag=hashlib.md5(r.body).hexdigest())
        def complete_multipart_upload(self,r):
            calls.append(('COMPLETE',r.key));assert r.forbid_overwrite is True
            parts=sessions[r.upload_id]['parts'];assert [p.part_number for p in r.complete_multipart_upload.parts]==sorted(parts)
            assert r.key not in objects
            objects[r.key]=b''.join(parts[p] for p in sorted(parts))
        def abort_multipart_upload(self,r):calls.append(('ABORT',r.key));sessions.pop(r.upload_id)
    cloud=Cloud();monkeypatch.setattr(d,'client',lambda _:cloud);monkeypatch.setattr(u,'client',lambda _:cloud)
    assets=AssetStore(config,runtime.config.project_registry_root_path());target=freeze_target(runtime,'demo',{'executor':'pai'})
    return client,assets,target,cloud,objects,sessions,calls


def put(datasets,content=None):
    cli,assets,target,cloud,objects,sessions,calls=datasets
    content=content if content is not None else archive({'tokens.bin':b'fineweb sample'*200})
    key='users/owner/datasets/fineweb/v1.tar';sha=hashlib.sha256(content).hexdigest();objects[key]=content
    return content,key,sha


def test_old_preparation_accepts_only_dlc_input_capability_expansion(datasets):
    from ml_exp_server.runs.experiment_preparation import ExperimentPreparationService
    cli,assets,target,*_=datasets
    service=object.__new__(ExperimentPreparationService)
    service.containers=ContainerExecutionService(cli.app.state.runtime)
    current=service.containers.profile_capabilities()['pai']
    selected={'executor':'pai','profile':copy.deepcopy(target['profile']),
              'capabilities':copy.deepcopy(current)}
    selected['capabilities']['data']['assets']=False
    original=copy.deepcopy(selected)
    service.check_profile(selected)
    assert selected==original
    assert service.containers.profile_capabilities()['pai']==current
    # This is not a general capability/profile drift exemption.
    for section,key,value in [('network','compute_internet',True),('storage','storage_id','other')]:
        changed=copy.deepcopy(selected);changed['capabilities'][section][key]=value
        with pytest.raises(ApplicationError,match='executor configuration changed'):
            service.check_profile(changed)
    changed=copy.deepcopy(selected);changed['profile']['backend']['quota_id']='other'
    with pytest.raises(ApplicationError):service.check_profile(changed)
    # Disabling input support is never accepted for a preparation that had it.
    service.containers.profile_capabilities=lambda _: {'pai':original['capabilities']}
    changed=copy.deepcopy(selected);changed['capabilities']=current
    with pytest.raises(ApplicationError):service.check_profile(changed)
    # A missing executor still fails closed.
    service.containers.profile_capabilities=lambda _: {}
    with pytest.raises(ApplicationError):service.check_profile(selected)


def test_registered_dataset_is_byte_verified_downloadable_and_mountable(datasets,monkeypatch,tmp_path):
    cli,assets,target,cloud,objects,sessions,calls=datasets;data,key,sha=put(datasets)
    request={'executor':'pai','object_key':key,'sha256':sha,'bytes':len(data)}
    result=cli.post('/api/projects/demo/assets/import-oss',json=request)
    assert result.status_code==200,result.text
    asset=result.json();assert asset['asset_id']=='asset.'+sha and asset['verification']=='archive-and-files-sha256'
    assert cli.post('/api/projects/demo/assets/import-oss',json=request).status_code==200
    assert sum(name=='GET' for name,_ in calls)==1
    ticket=cli.get('/api/projects/demo/assets/'+asset['asset_id']+'/download')
    assert ticket.status_code==200 and ticket.headers['Cache-Control']=='no-store'
    assert ticket.json()['transport']=='oss-presigned-get' and ticket.json()['acknowledgement_required'] is False
    assert cli.get('/api/projects/demo/assets/'+asset['asset_id']+'/archive').content==data
    assert cli.get('/api/projects/demo/assets').json()['assets'][0]['asset_id']==asset['asset_id']
    profile=target['profile'];assert for_backend(asset,profile,'demo')['remote_storage']=='oss'
    cap=ContainerExecutionService(cli.app.state.runtime).profile_capabilities()['pai']
    assert cap['data']['assets'] and cap['data']['asset_preparation']=='inside_job'
    limits=cli.get('/api/storage-limits').json()['oss_datasets']
    assert limits['max_parts']==10000 and limits['upload_prefix']=='users/owner/datasets/ml-expd/'
    assert limits['download_deletes_dataset'] is False
    runtime=custom_runtime(cli,monkeypatch)
    result=cli.post('/api/projects/demo/runs',json={'run_id':'oss-input','runtime_id':runtime['runtime_id'],'executor':'pai',
                         'inputs':[{'asset_id':asset['asset_id'],'mount_path':'/inputs/fineweb'}]})
    assert result.status_code==200,result.text
    snapshot=cli.post('/api/projects/demo/runs',json={'run_id':'no-uploaded-checkpoint','runtime_id':runtime['runtime_id'],
                        'executor':'pai','checkpoint_upload':{}})
    assert snapshot.status_code==409 and 'checkpoint_upload' in snapshot.text
    # Same production extractor: cache is populated and reused, corrupt cache is rejected.
    item={**asset,'url':'https://api.example/input'}
    tree=deliver(item,'scoped',tmp_path/'cpfs-cache',archive_stream=io.BytesIO(data))
    assert (tree/'tokens.bin').read_bytes()==b'fineweb sample'*200
    assert deliver(item,'scoped',tmp_path/'cpfs-cache')==tree
    assert not list(assets.root.rglob('*.tar')) and not list(assets.root.rglob('tree'))
    with assets.objects.record('demo','unrelated','attempt-001') as (_,value):assert value is None


def test_upload_uses_standard_parts_and_lost_completion_is_not_replayed(datasets,tmp_path,monkeypatch):
    cli,assets,target,cloud,objects,sessions,calls=datasets
    data=archive({'tokens.bin':bytes(range(256))*1200});sha=hashlib.sha256(data).hexdigest()
    from ml_exp_client.archives import data_transport_identity
    files,layer=data_transport_identity(io.BytesIO(data),len(data))
    request={'sha256':sha,'bytes':len(data),'target':{'executor':'pai'},'files':files,'transport_sha256':layer}
    first=cli.post('/api/projects/demo/asset-uploads',json=request);assert first.status_code==200,first.text
    value=first.json();endpoint='/api/projects/demo/asset-uploads/'+value['upload_id']
    for i in range(value['part_count']):
        body=data[i*value['part_bytes']:(i+1)*value['part_bytes']];h=hashlib.sha256(body).hexdigest()
        reply=cli.put(endpoint+'/parts/'+str(i),params={'sha256':h},content=body)
        assert reply.status_code==200,reply.text
        assert reply.json()['durable'] is True
        assert cli.put(endpoint+'/parts/'+str(i),params={'sha256':h},content=body).status_code==200
    assert len([c for c in calls if c[0]=='PART'])==value['part_count']
    original=cloud.complete_multipart_upload
    def lost(r):original(r);raise service_error('InternalError')
    monkeypatch.setattr(cloud,'complete_multipart_upload',lost)
    assert cli.post(endpoint+'/complete',json={}).status_code==503
    assert cli.get(endpoint).json()['status']=='FINALIZING'
    monkeypatch.setattr(cloud,'complete_multipart_upload',original)
    complete=cli.post(endpoint+'/complete',json={});assert complete.status_code==200,complete.text
    assert complete.json()['remote_storage']=='oss'
    assert sum(name=='COMPLETE' for name,_ in calls)==1
    assert cli.post(endpoint+'/complete',json={}).status_code==200
    assert cli.post('/api/projects/demo/asset-uploads',json=request).json()['status']=='COMPLETED'
    assert cli.delete(endpoint).status_code==409
    assert not list((assets.root.parent/'transfer-buffer').rglob('*.part'))


def test_registration_cannot_read_or_sign_outside_personal_scope(datasets):
    cli,assets,*_=datasets
    for key in ['users/other/datasets/a.tar','users/owner/results/a.tar','users/owner/datasets/../secret','users/owner/datasets//a','users/owner/datasets/.key','users/owner/datasets/a\\b','users/owner/datasets/a\n','users/owner/datasets/']:
        response=cli.post('/api/projects/demo/assets/import-oss',json={'executor':'pai','object_key':key,'sha256':'a'*64,'bytes':1})
        assert response.status_code==409,(key,response.text)
        assert assets.list('demo')['assets']==[]


def test_registration_does_not_publish_corrupt_or_missing_objects(datasets):
    cli,assets,*_=datasets;data,key,sha=put(datasets)
    for change in [{'sha256':'0'*64},{'bytes':1},{'bytes':100000},{'object_key':'users/owner/datasets/missing'}]:
        response=cli.post('/api/projects/demo/assets/import-oss',json={'executor':'pai','object_key':key,'sha256':sha,'bytes':len(data),**change})
        assert response.status_code in {409,503},(change,response.text)
        assert assets.list('demo')['assets']==[]


def test_settings_disabled_and_object_scope_mismatches_are_closed(datasets,monkeypatch):
    cli,assets,target,*_=datasets;data,key,sha=put(datasets)
    for value in [None,{},dict(S,bucket='BAD'),dict(S,region='bad/region'),dict(S,user='../x'),dict(S,credentials_file='relative')]:
        with pytest.raises(ApplicationError):d.settings({'oss_datasets':value})
    with pytest.raises(ValueError):d.object_scope(S,None)
    with pytest.raises(ValueError):d.OSSDatasets(assets).register('demo',{**target,'backend':'slurm'},key,sha,len(data))
    with pytest.raises(ApplicationError):d.OSSDatasets(assets).location({})
    with pytest.raises(ValueError):d.OSSDatasets(assets).location({'remote_storage':'oss','remote_locator':{'object_key':key,'bucket':'other','region':S['region'],'user':S['user']}})
    assert d.locator({'remote_locations':{'a':{'remote_storage':'wyd'},'b':{'remote_storage':'oss','remote_locator':{'x':1}}}})=={'x':1}
    assert d.locator({'remote_locations':{'a':{'remote_storage':'wyd'}}}) is None


def session(datasets,tmp_path):
    cli,assets,target,*_=datasets;data,_,sha=put(datasets)
    from ml_exp_client.archives import data_transport_identity
    files,layer=data_transport_identity(io.BytesIO(data),len(data))
    driver=ManagedDataUploads(assets,target,files,layer);binding={'project':'demo','kind':'asset'}
    value=driver.create(binding,sha,len(data),None);body=tmp_path/'part';body.write_bytes(data)
    return driver,value,body,data,sha


def test_upload_abort_generations_and_unknown_initiation_are_explicit(datasets,tmp_path,monkeypatch):
    driver,value,body,data,sha=session(datasets,tmp_path);native=driver.driver;binding=driver.binding
    assert driver.abort(value['upload_id'],binding)['status']=='ABORTED'
    assert driver.abort(value['upload_id'],binding)['status']=='ABORTED'
    second=driver.create({'project':'demo','kind':'asset'},sha,len(data),None)
    assert second['upload_id']==value['upload_id'] and second['status']=='UPLOADING'
    cloud=datasets[3];original=cloud.initiate_multipart_upload
    driver.abort(value['upload_id'],binding)
    monkeypatch.setattr(cloud,'initiate_multipart_upload',lambda *_:(_ for _ in ()).throw(service_error('InternalError')))
    with pytest.raises(ApplicationError):driver.create({'project':'demo','kind':'asset'},sha,len(data),None)
    assert driver.read(value['upload_id'],binding)['status']=='CREATING'
    count=len(datasets[-1])
    assert driver.create({'project':'demo','kind':'asset'},sha,len(data),None)['status']=='CREATING'
    assert len(datasets[-1])==count
    assert driver.abort(value['upload_id'],binding)['status']=='ABORTED'
    with pytest.raises(ValueError):native.read('invalid',binding)
    for method in [lambda:native.read('upload.'+'f'*64,binding),lambda:native.abort('upload.'+'f'*64,binding),
                   lambda:native.complete('upload.'+'f'*64,binding),lambda:native.part('upload.'+'f'*64,binding,0,body,sha,len(data))]:
        with pytest.raises(ApplicationError) as error:method()
        assert error.value.status_code==404
    with pytest.raises(ApplicationError):native.read(value['upload_id'],{'project':'another','kind':'asset'})


def test_lost_part_is_bound_before_send_and_reconciles_same_bytes(datasets,tmp_path,monkeypatch):
    driver,value,body,data,sha=session(datasets,tmp_path);native=driver.driver;binding=driver.binding;uid=value['upload_id'];cloud=datasets[3]
    original=cloud.upload_part
    def lost(r):original(r);raise service_error('InternalError')
    monkeypatch.setattr(cloud,'upload_part',lost)
    with pytest.raises(ApplicationError):driver.part(uid,binding,0,body,sha,len(data))
    wrong=b'!'*len(data);body.write_bytes(wrong)
    with pytest.raises(ValueError,match='pending'):driver.part(uid,binding,0,body,hashlib.sha256(wrong).hexdigest(),len(wrong))
    body.write_bytes(data);monkeypatch.setattr(cloud,'upload_part',original)
    driver.part(uid,binding,0,body,sha,len(data))
    body.write_bytes(wrong)
    with pytest.raises(ValueError,match='sealed'):driver.part(uid,binding,0,body,hashlib.sha256(wrong).hexdigest(),len(wrong))
    body.write_bytes(data)
    with pytest.raises(ValueError,match='length'):native.part(uid,binding,0,body,sha,len(data)+1)
    with pytest.raises(ValueError,match='checksum'):native.part(uid,binding,0,body,'0'*64,len(data))
    result=driver.complete(uid,binding)
    assert native.part(uid,binding,0,body,sha,len(data))['durable']
    with native.state(uid,binding) as (path,state):
        state['parts']={};from ml_exp_server.storage import atomic_json;atomic_json(path,state)
    with pytest.raises(ValueError,match='matching part'):native.part(uid,binding,0,body,sha,len(data))


def test_incomplete_expired_and_changed_uploads_are_not_sent(datasets,tmp_path):
    driver,value,body,data,sha=session(datasets,tmp_path);native=driver.driver;uid=value['upload_id'];binding=driver.binding
    with pytest.raises(ValueError,match='missing parts'):driver.complete(uid,binding)
    from ml_exp_server.storage import atomic_json
    with native.state(uid,binding) as (path,state):state['expires_at']='2000-01-01T00:00:00+00:00';atomic_json(path,state)
    with pytest.raises(ApplicationError):driver.part(uid,binding,0,body,sha,len(data))
    driver.abort(uid,binding)
    with pytest.raises(ApplicationError):driver.part(uid,binding,0,body,sha,len(data))
    assets=datasets[1];target=datasets[2]
    with pytest.raises(ApplicationError):u.OSSDataUploads(assets,{**target,'oss_datasets':{}})
    with pytest.raises(ApplicationError):native.create(binding,sha,native.assets.objects.config['upload_part_bytes']*10001,None)
    native.assets.objects.config['oss_dataset_part_bytes']=1024
    with pytest.raises(ApplicationError):u.OSSDataUploads(assets,target)


def test_complete_retries_only_known_parts_when_previous_request_did_not_publish(datasets,tmp_path,monkeypatch):
    driver,value,body,data,sha=session(datasets,tmp_path);uid=value['upload_id'];binding=driver.binding;cloud=datasets[3]
    driver.part(uid,binding,0,body,sha,len(data));original=cloud.complete_multipart_upload
    monkeypatch.setattr(cloud,'complete_multipart_upload',lambda *_:(_ for _ in ()).throw(service_error('InternalError')))
    with pytest.raises(ApplicationError):driver.complete(uid,binding)
    with pytest.raises(ApplicationError):driver.abort(uid,binding)
    head=cloud.head_object
    monkeypatch.setattr(cloud,'head_object',lambda *_:(_ for _ in ()).throw(service_error('AccessDenied')))
    with pytest.raises(ApplicationError):driver.complete(uid,binding)
    monkeypatch.setattr(cloud,'head_object',head);monkeypatch.setattr(cloud,'complete_multipart_upload',original)
    assert driver.complete(uid,binding)['status']=='READY'


def test_registration_freezes_location_and_reuses_existing_content_copy(datasets,monkeypatch):
    cli,assets,target,cloud,objects,*_=datasets;data,key,sha=put(datasets);service=d.OSSDatasets(assets)
    legacy=assets.receive('demo',io.BytesIO(data),sha,len(data))
    asset=service.register('demo',target,key,sha,len(data));assert asset['asset_id']==legacy['asset_id']
    assert d.locator(assets.read('demo',asset['asset_id']))==asset['remote_locator']
    assert assets.download('demo',asset['asset_id'])['transport']=='oss-presigned-get'
    objects[key+'copy']=data
    for newkey,size,files in [(key+'copy',len(data),None),(key,len(data)+1,None),(key,len(data),[])]:
        with pytest.raises(ValueError,match='bound'):service.register('demo',target,newkey,sha,size,files=files)
    # A different content identity cannot use a caller-supplied false file inventory.
    data2=archive({'different':b'new'});key2=key+'2';objects[key2]=data2
    with pytest.raises(ValueError,match='inventory'):service.register('demo',target,key2,hashlib.sha256(data2).hexdigest(),len(data2),files=[])
    original=assets.read
    monkeypatch.setattr(assets,'read',lambda *args:(_ for _ in ()).throw(ApplicationError('blocked',status_code=410)))
    with pytest.raises(ApplicationError) as e:service.register('demo',target,key,sha,len(data))
    assert e.value.status_code==410


def test_stream_reader_uses_bounded_chunks_and_closes_idempotently():
    closed=[]
    source=SimpleNamespace(iter_bytes=lambda **kw:iter([b'abc',b'def']),close=lambda:closed.append(True))
    reader=d.ObjectBody(source);assert reader.readable()
    assert reader.read(1)==b'a' and reader.read(2)==b'bc' and reader.read(3)==b'def' and reader.read(1)==b''
    reader.close();reader.close();assert closed==[True]


@pytest.mark.parametrize('kind',['link','duplicate','files','members','member-size','total-size','directory','trailing'])
def test_archive_validation_and_bounds(datasets,kind):
    _,assets,target,cloud,objects,*_=datasets;raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w') as package:
        names=['a','b'] if kind in {'files','total-size'} else ['a','a'] if kind=='duplicate' else ['a']
        if kind=='members':names=['dir1','dir2','dir3']
        for name in names:
            member=tarfile.TarInfo(name)
            if kind=='link':member.type=tarfile.SYMTYPE;member.linkname='/etc/private';package.addfile(member)
            elif kind in {'members','directory'}:member.type=tarfile.DIRTYPE;package.addfile(member)
            else:member.size=3;package.addfile(member,io.BytesIO(b'abc'))
        if kind=='directory':
            member=tarfile.TarInfo('file');member.size=3;package.addfile(member,io.BytesIO(b'abc'))
    data=raw.getvalue()+(b'\0'*100000 if kind=='trailing' else b'')
    key='users/owner/datasets/'+kind;objects[key]=data
    if kind in {'files','members'}:assets.file_limit=1
    if kind=='member-size':assets.expanded_limit=2
    if kind=='total-size':assets.expanded_limit=4
    method=lambda:d.OSSDatasets(assets).register('demo',target,key,hashlib.sha256(data).hexdigest(),len(data))
    if kind in {'directory','trailing'}:assert method()['status']=='READY'
    else:
        with pytest.raises(ValueError):method()


def test_public_datasets_are_read_only_and_dlc_storage_is_personal(datasets):
    cli,assets,target,cloud,objects,*_=datasets;data,_,sha=put(datasets);key='public/datasets/fineweb/v1.tar';objects[key]=data
    assert d.OSSDatasets(assets).register('demo',target,key,sha,len(data))['status']=='READY'
    config=assets.objects.config;config['oss_datasets']={**S,'user':'other'}
    Path(cli.app.state.runtime.config.container_execution.artifact_store_file).write_text(json.dumps(config))
    with pytest.raises(ApplicationError,match='personal workspace'):freeze_target(cli.app.state.runtime,'demo',{'executor':'pai'})


def test_malformed_tar_and_large_declared_upload_have_explicit_limits(datasets,tmp_path):
    cli,assets,target,cloud,objects,*_=datasets
    data=b'not a tar archive';key='users/owner/datasets/broken';objects[key]=data
    response=cli.post('/api/projects/demo/assets/import-oss',json={'executor':'pai','object_key':key,
                      'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    assert response.status_code==422 and response.headers['X-ML-Expd-Error-Code']=='OSS_DATASET_ARCHIVE_INVALID'
    driver,value,body,content,sha=session(datasets,tmp_path)
    assets.objects.config['oss_dataset_part_bytes']=16*1024**2
    driver=ManagedDataUploads(assets,target,driver.files,driver.transport_sha256)
    large=driver.create({'project':'demo','kind':'asset'},sha,5*1024**3,None)
    assert large['bytes']==5*1024**3 and large['part_count']==320
    assert not large['parts']
    driver.abort(large['upload_id'],driver.binding)


def test_large_verification_has_readable_phase_and_bounded_progress(datasets,tmp_path,monkeypatch):
    cli,assets,target,cloud,objects,sessions,calls=datasets
    data=archive({'shard.bin':hashlib.shake_256(b'owned').digest(2*1024**2)})
    from ml_exp_client.archives import data_transport_identity
    files,layer=data_transport_identity(io.BytesIO(data),len(data));driver=ManagedDataUploads(assets,target,files,layer)
    state=driver.create({'project':'demo','kind':'asset'},hashlib.sha256(data).hexdigest(),len(data),None)
    uid=state['upload_id'];binding=driver.binding;body=tmp_path/'part'
    for number in range(state['part_count']):
        raw=data[number*state['part_bytes']:(number+1)*state['part_bytes']];body.write_bytes(raw)
        driver.part(uid,binding,number,body,hashlib.sha256(raw).hexdigest(),len(raw))
    original=cloud.get_object
    def observe(request):
        snapshot=driver.read(uid,binding)
        assert snapshot['phase']=='VERIFYING_OBJECT' and snapshot['verification_bytes_read']==0
        return original(request)
    monkeypatch.setattr(cloud,'get_object',observe)
    assert driver.complete(uid,binding)['status']=='READY'
    final=driver.read(uid,binding)
    assert final['phase']=='READY' and final['verification_bytes_read']==len(data)


@pytest.mark.parametrize('kind',['hash','transport'])
def test_failed_verification_has_durable_safe_diagnostic(datasets,tmp_path,monkeypatch,kind):
    driver,value,body,data,sha=session(datasets,tmp_path);uid=value['upload_id'];binding=driver.binding;cloud=datasets[3]
    driver.part(uid,binding,0,body,sha,len(data))
    if kind=='hash':
        original=cloud.complete_multipart_upload
        def corrupt(r):original(r);datasets[4][r.key]=b'!'*len(data)
        monkeypatch.setattr(cloud,'complete_multipart_upload',corrupt)
        expected='OSS_DATASET_VERIFICATION_FAILED'
    else:
        monkeypatch.setattr(cloud,'get_object',lambda *_:(_ for _ in ()).throw(service_error('AccessDenied')))
        expected='OSS_DATASET_REQUEST_FAILED'
    with pytest.raises((ValueError,ApplicationError)):driver.complete(uid,binding)
    observed=driver.read(uid,binding)
    assert observed['phase']=='VERIFICATION_FAILED' and observed['diagnostic']==expected
    assert 'private provider data' not in json.dumps(observed)
    assert datasets[1].list('demo')['assets']==[]
