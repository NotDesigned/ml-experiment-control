"""Fixed OCI recipe and Slurm digest-to-SIF conversion contract."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from ml_exp_server.image_builder import ImageBuilder, MANIFEST_TYPE
from ml_exp_server.source_revisions import _tree_digest
from ml_exp_server.source_imports import seal_tree
from experiment_control.backends.wyd import WydSlurmBackend


def test_builder_uses_fixed_recipe_and_copies_exact_readonly_source(tmp_path):
    tree=tmp_path/'sources/demo/pending/tree';tree.mkdir(parents=True)
    (tree/'train.py').write_text('print(1)\n')
    digest=_tree_digest(tree);source='source.'+digest[7:]
    directory=tree.parent.with_name(source);tree.parent.rename(directory);tree=directory/'tree'
    (directory/'source.json').write_text(json.dumps({'project':'demo','tree_digest':digest}))
    seal_tree(directory)
    builder=ImageBuilder({'state_root':str(tmp_path/'builds'),'source_root':str(tmp_path/'sources'),'repository':'registry.example/results'})
    calls=[]
    def docker(args,**kwargs):
        calls.append(args)
        if args[0]=='build':
            context=Path(args[-1])
            recipe=(context/'Dockerfile').read_text()
            assert 'COPY source/ /workspace/' in recipe and 'RUN ' not in recipe
            assert '--network=none' in args and 'COPY worker.py' in recipe
            assert (context/'source/train.py').read_text()=='print(1)\n'
        return ''
    builder._docker=docker
    manifest = json.dumps({'mediaType': MANIFEST_TYPE, 'config': {'digest': 'sha256:'+'a'*64}})
    published_digest = 'sha256:'+hashlib.sha256(manifest.encode()).hexdigest()
    def skopeo(args, **kwargs):
        if args[0] == 'copy':
            Path(args[args.index('--digestfile')+1]).write_text(published_digest)
            return ''
        return manifest
    builder._skopeo=skopeo
    request={'operation':'build','project':'demo','source_id':source,'base_image':'registry.example/python@sha256:'+'a'*64}
    result=builder.request(request)
    assert result['image'].endswith(published_digest)
    assert builder.request({**request,'operation':'get'})==result
    assert len(calls)==2
    with pytest.raises(ValueError):builder.request({**request,'base_image':'registry.example/python:latest'})


def test_slurm_conversion_uses_digest_and_repairs_tampered_cache(tmp_path):
    backend=object.__new__(WydSlurmBackend)
    binpath=tmp_path/'bin';binpath.mkdir()
    tool=binpath/'apptainer'
    tool.write_text('#!/bin/sh\nprintf image > "$3"\nprintf build >> "'+str(tmp_path/'calls')+'"\n')
    tool.chmod(0o700)
    import os
    def remote(alias,command):
        return subprocess.run(shlex.split(command),check=True,env={**os.environ,'PATH':str(binpath)+':'+os.environ['PATH']})
    backend.remote_exec=remote
    sif=tmp_path/'images/sha.sif'
    run={'backend':{'ssh_alias':'cluster','oci_image':'registry.example/image@sha256:'+'a'*64,'sif_path':str(sif)}}
    backend._stage_oci_image(run)
    backend._stage_oci_image(run)
    assert (tmp_path/'calls').read_text()=='build'
    sif.write_text('tamper')
    backend._stage_oci_image(run)
    assert (tmp_path/'calls').read_text()=='buildbuild' and sif.read_text()=='image'


@pytest.mark.parametrize("failure", ["unpublished", "wrong-digest", "changed-config", "wrong-format"])
def test_publication_requires_matching_remote_manifest(tmp_path, failure):
    builder = ImageBuilder({'state_root':str(tmp_path/'builds'), 'repository':'registry.example/results'})
    builder._docker = lambda *args, **kwargs: ''
    original = {'mediaType':MANIFEST_TYPE, 'config':{'digest':'sha256:'+'a'*64}}
    remote = dict(original)
    if failure == 'changed-config':
        remote['config'] = {'digest':'sha256:'+'b'*64}
    if failure == 'wrong-format':
        remote['mediaType'] = 'application/vnd.oci.image.manifest.v1+json'
    raw = json.dumps(remote)
    digest = 'sha256:'+hashlib.sha256(raw.encode()).hexdigest()
    def skopeo(args, **kwargs):
        if args[0] == 'copy':
            Path(args[args.index('--digestfile')+1]).write_text(digest)
            return ''
        if args[-1].startswith('docker-archive:'):
            return json.dumps(original)
        if failure == 'unpublished':
            raise ValueError('remote image is absent')
        return raw+' ' if failure == 'wrong-digest' else raw
    builder._skopeo=skopeo
    with pytest.raises(ValueError):
        builder._publish('registry.example/results:bundle-test', tmp_path)
