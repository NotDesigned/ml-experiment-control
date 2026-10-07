"""Bounded SSH delivery into WYD shared storage before GPU admission."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shlex
import subprocess
import threading

from .application_errors import ApplicationError
from .checkpoint_registry import storage_scope
from .data_assets import AssetStore
from .data_delivery import digest

# No credentials or URL enter the backend. The API streams through bounded RAM;
# temporary archive and validated files exist only on backend shared storage.
REMOTE = '''
import json,sys,tempfile
from pathlib import Path
header=sys.stdin.buffer.read(4)
size=int.from_bytes(header,'big')
if len(header)!=4 or not 0<size<=8*1024*1024: raise ValueError('invalid metadata size')
metadata=json.loads(sys.stdin.buffer.read(size))
item=metadata['item'];cache=Path(metadata['cache']);destination=cache/item['asset_id']
try:
 if metadata['mode']=='check':
  if not destination.exists(): result={'status':'NOT_FOUND'}
  else: verify_tree(destination,item['files']);result={'status':'READY'}
 else:
  cache.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryFile(dir=cache) as archive:
   remaining=item['archive_bytes']
   while remaining:
    chunk=sys.stdin.buffer.read(min(remaining,1024*1024))
    if not chunk: raise ValueError('archive truncated')
    archive.write(chunk);remaining-=len(chunk)
   if sys.stdin.buffer.read(1): raise ValueError('archive oversized')
   deliver(item,None,cache,archive_stream=archive)
  result={'status':'READY'}
except Exception as error:
 result={'status':'FAILED','error_class':type(error).__name__}
print(json.dumps(result),flush=True)
'''


class WydDataStager:
    def __init__(self, runtime):
        self.runtime=runtime
        self.assets=AssetStore(Path(runtime.config.container_execution.artifact_store_file),runtime.config.project_registry_root_path())

    def metadata(self, project, asset_id, profile, mode):
        item=self.assets.read(project,asset_id)
        cache=profile['storage_root'].rstrip('/')+'/'+project+'/data-assets'
        return {'mode':mode,'cache':cache,'item':{key:item[key] for key in ('asset_id','sha256','archive_bytes','files')}}

    @contextmanager
    def process(self, profile, metadata):
        source=Path(__file__).with_name('data_input.py').read_text()
        transport=Path(__file__).with_name('worker_http.py').read_text()
        code="import types,sys;module=types.ModuleType('worker_http');sys.modules['worker_http']=module\n"+"exec("+repr(transport)+",module.__dict__)\n"+"exec("+repr(source)+",globals())\n"+REMOTE
        body=json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()
        if len(body)>8*1024*1024:
            raise ApplicationError('WYD data manifest is too large',code='WYD_DATA_METADATA_TOO_LARGE')
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',profile['backend']['ssh_alias'],shlex.join(['python3','-c',code])]
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        # Includes slow stdin writes, transport, unpacking and checksum reads.
        expired=threading.Event()
        def timeout():
            expired.set();process.kill()
        timer=threading.Timer(1800,timeout)
        timer.daemon=True;timer.start()
        try:
            process.stdin.write(len(body).to_bytes(4,'big')+body)
            yield process
            process.stdin.close()
            output=process.stdout.read(8193)
            code=process.wait(timeout=15)
            if expired.is_set() or code or len(output)>8192:
                raise ValueError('SSH data delivery did not return verified evidence')
            result=json.loads(output)
            if result.get('status') not in {'READY','NOT_FOUND'}:
                raise ValueError('backend data verification failed')
            process.result=result
        except (OSError,ValueError,subprocess.SubprocessError):
            raise ApplicationError('WYD data preparation failed; inspect the saved preparation, no automatic replay',code='WYD_DATA_PREPARATION_FAILED') from None
        finally:
            timer.cancel()
            if process.poll() is None: process.kill()
            process.wait()
            if not process.stdin.closed:process.stdin.close()
            process.stdout.close()

    def check(self, project, asset_id, profile):
        metadata=self.metadata(project,asset_id,profile,'check')
        with self.process(profile,metadata) as process:
            pass
        return process.result

    def publish(self, project, asset_id, profile):
        metadata=self.metadata(project,asset_id,profile,'publish')
        value,body=self.assets.archive(project,asset_id)
        try:
            with self.process(profile,metadata) as process:
                for chunk in body.iter_chunks(chunk_size=1024*1024):
                    process.stdin.write(chunk)
        finally:
            body.close()
        if process.result['status']!='READY':
            raise ApplicationError('WYD cache was not verified',code='WYD_DATA_NOT_READY')
        return self.receipt(project,asset_id,profile)

    def receipt(self, project, asset_id, profile):
        item=self.assets.read(project,asset_id)
        return {'status':'READY','asset_id':asset_id,'archive_sha256':item['sha256'],
                'files_sha256':digest(item['files']),
                'storage_scope':storage_scope(profile['backend'],profile['storage_root'].rstrip('/')+'/'+project),
                'transport':'ssh_stream_shared_storage'}
