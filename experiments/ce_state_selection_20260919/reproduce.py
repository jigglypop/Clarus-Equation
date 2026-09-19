"""Verify the distributed files, rerun tests and compare regenerated results."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent

def compare(a, b, path='root'):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f'{path}: keys differ')
        for key in a:
            compare(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise AssertionError(f'{path}: list lengths differ')
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]')
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not math.isclose(a, b, rel_tol=1e-10, abs_tol=2e-13):
            raise AssertionError(f'{path}: {a!r} != {b!r}')
    elif a != b:
        raise AssertionError(f'{path}: values differ')

def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = ROOT / name
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise RuntimeError(f'Hash mismatch: {name}')
    print(f"SHA256: {len(manifest['sha256'])} files verified", flush=True)
    env = os.environ.copy()
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    with tempfile.TemporaryDirectory(prefix='ce_cs1_replay_') as work:
        output = Path(work) / 'results.json'
        subprocess.run([sys.executable, str(ROOT/'state_selection.py'),
                        '--output', str(output)], check=True, cwd=work, env=env)
        compare(json.loads((ROOT/'results.json').read_text()),
                json.loads(output.read_text()))
        print('Regenerated results agree (rtol=1e-10, atol=2e-13).')
        print('Byte-identical results:', output.read_bytes() == (ROOT/'results.json').read_bytes())

if __name__ == '__main__':
    main()
