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




@pytest.mark.parametrize('failure', [None, 'wrong-digest', 'changed-config', 'wrong-format', 'missing-config'])
def test_buildkit_streaming_publication_verifies_exact_remote_identity(tmp_path, failure):
    builder = ImageBuilder({'state_root':str(tmp_path/'builds'), 'repository':'registry.example/results'})
    config_digest = 'sha256:'+'a'*64
    remote = {'mediaType':MANIFEST_TYPE, 'config':{'digest':config_digest}}
    if failure == 'changed-config':remote['config'] = {'digest':'sha256:'+'b'*64}
    if failure == 'wrong-format':remote['mediaType'] = 'application/vnd.oci.image.manifest.v1+json'
    raw = json.dumps(remote)
    digest = 'sha256:'+hashlib.sha256(raw.encode()).hexdigest()
    def docker(args, **kwargs):
        assert args[:2] == ['buildx', 'build']
        assert '--provenance=false' in args and '--network=none' in args
        assert 'type=image,push=true,oci-mediatypes=false,compression=gzip' in args
        Path(args[args.index('--metadata-file')+1]).write_text(json.dumps({
            'containerimage.digest':digest,
            'containerimage.config.digest': '' if failure == 'missing-config' else config_digest,
        }))
        return ''
    builder._docker = docker
    builder._skopeo = lambda *args, **kwargs: raw+' ' if failure == 'wrong-digest' else raw
    if failure:
        with pytest.raises(ValueError):builder._publish_buildkit('registry.example/results:bundle-test',tmp_path)
    else:
        assert builder._publish_buildkit('registry.example/results:bundle-test',tmp_path).endswith(digest)
    assert not (tmp_path/'image.tar').exists()
