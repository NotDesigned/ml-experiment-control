"""Check the installed native-image bridge with the builder's own interpreter.

Run without PYTHONPATH/PYTHONHOME. This inspects the exact configured worker and
creates only a temporary capability marker under state_root; no provider call,
image build, Runtime edit or training submission is made.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bridge-sha256', required=True,
                        help='SHA256 of native_worker.py from the accepted source')
    parser.add_argument('--worker-sha256', required=True,
                        help='sha256: digest frozen in the server build configuration')
    parser.add_argument('--require-feature', action='append', default=[])
    args = parser.parse_args()
    if os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'):
        parser.error('remove PYTHONPATH/PYTHONHOME to inspect the installed package')
    from ml_exp_server.builds import native_worker
    module = Path(native_worker.__file__).resolve(strict=True)
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    if digest != args.bridge_sha256:
        parser.error('installed native_worker.py differs from the accepted source')
    config = json.loads(args.config.read_text())
    with tempfile.TemporaryDirectory(prefix='native-bridge-check-', dir=config['state_root']) as root:
        features = native_worker.freeze_features(Path(config['native_worker_bin']),
                                                 args.worker_sha256, Path(root) / 'features.json')
    if not set(args.require_feature) <= set(features):
        parser.error('configured worker lacks a required feature')
    print(json.dumps({'module': str(module), 'bridge_sha256': digest,
                      'worker_sha256': args.worker_sha256, 'features': features}))


if __name__ == '__main__':
    main()
