"""Operator-only pinned CCR prewarm through a restricted HTTPS range relay.

No blob bytes are staged on the API host. Signed URLs and registry credentials
stay in memory. SSH sends only control and signed requests; the desktop receives
blob bytes over HTTPS. The adapter needs CCR_BLOB_CACHE=/cache and an owned
volume mounted there. No build or scheduler request is issued by this tool.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request

REGISTRY = 'registry.cn-sh-01.sensecore.cn'
HOST = 'aoss.cn-sh-01b.sensecoreapi-oss.cn'

# Runs inside the restricted adapter. Each complete blob is verified, then
# renamed atomically. Partial files are removed even on a short input stream.
WRITER = '''import fcntl,hashlib,json,os,pathlib,signal,sys,tempfile
digest,size,limit=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);root=pathlib.Path(os.environ.get('CCR_BLOB_CACHE','/cache'))
lock=(root/'.cache.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
used=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
if used+size>limit:raise SystemExit('cache capacity exceeded; remove selected owned blobs first')
stats=os.statvfs(root)
if stats.f_bavail*stats.f_frsize<size+2*1024**3:raise SystemExit('insufficient desktop cache storage')
def stalled(*args):raise ValueError('cache input stalled')
signal.signal(signal.SIGALRM,stalled);signal.signal(signal.SIGTERM,stalled)
fd,name=tempfile.mkstemp(prefix='.partial-',dir=root);h=hashlib.sha256();n=0
try:
 with os.fdopen(fd,'wb') as out:
  while n<size:
   signal.alarm(120)
   data=sys.stdin.buffer.read(min(1048576,size-n))
   if not data:raise ValueError('incomplete layer')
   out.write(data);h.update(data);n+=len(data)
  out.flush();os.fsync(out.fileno())
 signal.alarm(0)
 if h.hexdigest()!=digest:raise ValueError('layer checksum mismatch')
 os.chmod(name,0o444);os.replace(name,root/digest)
 receipt={'sha256':digest,'bytes':n}
 if 'emit' in globals():emit(receipt)
 else:print(json.dumps(receipt),flush=True)
finally:
 if os.path.exists(name):os.unlink(name)
 iterator=getattr(sys.stdin.buffer,'iterator',None)
 if iterator is not None:iterator.close()
'''

# Consume one small private control document from Docker exec stdin. Remote
# downloading bypasses bulk SSH traffic; only bounded progress returns via SSH.
DOWNLOADER = '''import io,json,os,sys,time,urllib.request,urllib.error,threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
private_input=os.fdopen(os.dup(0),'rb',buffering=0)
control=json.loads(private_input.readline(16385))
output_lock=threading.Lock()
def emit(value):
 with output_lock:print(json.dumps(value),flush=True)
def refresh():
 while line:=private_input.readline(16385):
  value=json.loads(line)
  if set(value)!= {'url'} or not isinstance(value['url'],str):return
  control['url']=value['url']
  emit({'signed_url_refreshed':True})
threading.Thread(target=refresh,daemon=True).start()
size=control['size'];chunk=2*1024*1024;workers=16;t=time.monotonic()
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args):return None
def fetch(start):
 end=min(size-1,start+chunk-1)
 for attempt in range(2):
  try:
   body=json.dumps({'url':control['url'],'start':start,'end':end,'size':size}).encode()
   request=urllib.request.Request(control['relay'],data=body,headers={'Authorization':'Bearer '+control['token'],'Content-Type':'application/json'})
   with urllib.request.build_opener(NoRedirect).open(request,timeout=45) as response:
    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('relay range identity mismatch')
    data=response.read(end-start+2)
   if len(data)!=end-start+1:raise OSError('incomplete relay range')
   return data
  except OSError:
   if attempt==1:raise ValueError('relay range unavailable') from None
def chunks():
 with ThreadPoolExecutor(max_workers=workers) as pool:
  for offset in range(0,size,workers*chunk):
   futures=[pool.submit(fetch,start) for start in range(offset,min(size,offset+workers*chunk),chunk)]
   for future in futures:yield future.result()
   done=min(size,offset+workers*chunk);elapsed=time.monotonic()-t
   emit({'downloaded_bytes':done,'total_bytes':size,'seconds':round(elapsed,2),'MB_per_s':round(done/elapsed/1e6,2)})
class Stream:
 def __init__(self):self.pending=b'';self.iterator=chunks()
 def read(self,n):
  if not self.pending:self.pending=next(self.iterator,b'')
  data,self.pending=self.pending[:n],self.pending[n:];return data
sys.stdin=SimpleNamespace(buffer=Stream())
sys.argv=['writer',control['digest'],str(size),str(control['limit'])]
'''


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def blob_url(repository, digest, auth):
    opener = urllib.request.build_opener(NoRedirect)
    url = 'https://' + REGISTRY + '/v2/' + repository + '/blobs/' + digest
    headers = {'Authorization': 'Basic ' + auth}
    def request():
        try:
            return opener.open(urllib.request.Request(url, headers=headers), timeout=20)
        except urllib.error.HTTPError as error:
            return error
    response = request()
    if response.code == 401:
        params = dict(re.findall(r'(\w+)="([^"]*)"', response.headers['WWW-Authenticate']))
        realm = params.pop('realm')
        if urllib.parse.urlsplit(realm).netloc != REGISTRY or not realm.startswith('https://'):
            raise ValueError('unexpected registry token endpoint')
        params['scope'] = 'repository:' + repository + ':pull'
        token_url = realm + '?' + urllib.parse.urlencode(params)
        with urllib.request.urlopen(urllib.request.Request(token_url, headers=headers), timeout=20) as result:
            token = json.load(result)
        headers = {'Authorization': 'Bearer ' + token.get('token', token.get('access_token'))}
        response.close()
        response = request()
    location = response.headers.get('Location', '')
    status = response.code
    response.close()
    parsed = urllib.parse.urlsplit(location)
    expected = '/registry/docker/registry/v2/blobs/sha256/' + digest[7:9] + '/' + digest[7:] + '/data'
    if status not in (302, 307) or parsed.netloc != HOST or parsed.scheme not in ('http', 'https') or parsed.path != expected:
        raise ValueError('unexpected CCR blob redirect')
    return urllib.parse.urlunsplit(('https', parsed.netloc, parsed.path, parsed.query, ''))


def prewarm(config, image, limit, relay, token):
    prefix = REGISTRY + '/ccr-zhicheng-02/'
    if not re.fullmatch(re.escape(prefix) + r'[A-Za-z0-9._/-]+@sha256:[0-9a-f]{64}', image):
        raise ValueError('prewarm requires a pinned approved CCR image')
    endpoint = config['docker_host']
    if not re.fullmatch(r'unix:///[-A-Za-z0-9_./]+', endpoint) or '..' in endpoint.split('/'):
        raise ValueError('prewarm requires the private Unix Docker endpoint')
    parsed = urllib.parse.urlsplit(relay)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path != '/ml-expd-builder-relay' or parsed.query or parsed.fragment
            or not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', token)):
        raise ValueError('invalid credential-free HTTPS relay or private token')
    authfile = config.get('registry_auth_file', '/root/.docker/config.json')
    command = [config.get('skopeo', '/usr/bin/skopeo'), 'inspect', '--authfile', authfile, '--raw', 'docker://' + image]
    result = subprocess.run(command, check=True, capture_output=True, timeout=60)
    if hashlib.sha256(result.stdout).hexdigest() != image.rsplit(':', 1)[1]:
        raise ValueError('base manifest identity mismatch')
    manifest = json.loads(result.stdout)
    if 'manifests' in manifest:
        raise ValueError('prewarm requires a linux/amd64 image manifest, not an index')
    layers = manifest['layers']
    if not layers or any(not re.fullmatch(r'sha256:[0-9a-f]{64}', l['digest']) or type(l['size']) is not int or l['size'] <= 0 for l in layers):
        raise ValueError('invalid base layer metadata')
    if sum(l['size'] for l in layers) > limit:
        raise ValueError('base layers exceed the dedicated cache budget')
    auth = json.loads(Path(authfile).read_text())['auths'][REGISTRY]['auth']
    base64.b64decode(auth, validate=True)  # Validate; never display credentials.
    docker = [config.get('docker', '/usr/bin/docker'), '--host', endpoint, 'exec', '-i', 'ml-expd-ccr-https']
    repository = image.split('/', 1)[1].split('@', 1)[0]
    for layer in layers:
        digest, size = layer['digest'][7:], layer['size']
        # Rehash before reuse. Cache metadata alone never proves an intact blob.
        check = "import hashlib,os,pathlib,sys;p=pathlib.Path(os.environ.get('CCR_BLOB_CACHE','/cache'))/sys.argv[1];h=hashlib.sha256();n=0\nif p.is_file() and not p.is_symlink():\n with p.open('rb') as f:\n  while b:=f.read(1048576):h.update(b);n+=len(b)\nprint(int(n==int(sys.argv[2]) and h.hexdigest()==sys.argv[1]))"
        existing = subprocess.run(docker + ['python3', '-c', check, digest, str(size)], capture_output=True, check=True, timeout=60)
        if existing.stdout.strip() == b'1':
            print(json.dumps({'layer': digest, 'bytes': size, 'cache': 'verified-hit'}), flush=True)
            continue
        url = blob_url(repository, layer['digest'], auth)
        control = {'url': url, 'digest': digest, 'size': size, 'limit': limit, 'relay': relay, 'token': token}
        child = subprocess.Popen(docker + ['python3', '-c', DOWNLOADER + WRITER],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        watchdog = threading.Timer(3600, child.kill)
        watchdog.start()
        finished = threading.Event()
        def refresh_url():
            # CCR signs each redirect for 1200 s. Slow cold layers must obtain
            # fresh URLs without restarting or sending blob bytes over SSH.
            while not finished.wait(300):
                try:
                    update = {'url': blob_url(repository, layer['digest'], auth)}
                    if finished.is_set():
                        return
                    child.stdin.write(json.dumps(update).encode() + b'\n')
                    child.stdin.flush()
                except Exception:
                    child.kill()  # Fail closed; never print the signed request.
                    return
        refresher = threading.Thread(target=refresh_url, daemon=True)
        try:
            child.stdin.write(json.dumps(control).encode() + b'\n')
            child.stdin.flush()
            refresher.start()
            receipt = None
            for line in child.stdout:
                value = json.loads(line)
                if 'sha256' in value:
                    receipt = value
                else:
                    # Never echo arbitrary worker text or its private control.
                    print(json.dumps({'layer': digest, **{k: value[k] for k in
                          ('downloaded_bytes', 'total_bytes', 'seconds', 'MB_per_s', 'signed_url_refreshed') if k in value}}), flush=True)
            if child.wait(timeout=60) != 0:
                raise ValueError('desktop cache checksum or write failed')
            if receipt != {'sha256': digest, 'bytes': size}:
                raise ValueError('desktop cache receipt mismatch')
            print(json.dumps({'layer': digest, 'bytes': size, 'cache': 'verified-new'}), flush=True)
        finally:
            finished.set()
            if refresher.ident is not None:
                refresher.join(timeout=60)
            child.stdin.close()
            watchdog.cancel()
            if child.poll() is None:
                child.kill()
            child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--relay-url', required=True)
    parser.add_argument('--relay-token-file', type=Path, required=True)
    parser.add_argument('--max-cache-gib', type=int, default=16, choices=range(1, 65))
    args = parser.parse_args()
    try:
        prewarm(json.loads(args.config.read_text()), args.image, args.max_cache_gib * 1024 ** 3,
                args.relay_url, args.relay_token_file.read_text().strip())
    except Exception as error:
        # HTTP errors may include signed URLs. Print only a class and static text.
        raise SystemExit('Prewarm failed (' + type(error).__name__ + '); no build was submitted. Inspect connection, capacity and pinned identity.') from None


if __name__ == '__main__':
    main()
