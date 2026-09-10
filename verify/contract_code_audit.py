"""Fixed audit of existing contracts; no new candidate or observational fitting.

Print a receipt without overwriting historical candidate artifacts.
The eigendecomposition crosscheck is independent residual arithmetic, not an
independent physical forward model. See the dated contract audit in paper.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dimension_joint_candidate import joint_residuals, relative_flat_action
from ce_async_projection_memory import exact_audit, numeric_audit
from ce_embedding_einstein_variation import cosmological_counterexample, free_embedding
from ce_latest_common_epsilon_score import run as partial_score
from experiments.preregistration.validate_holdout_manifest import validate_manifest_path


def run():
    groups = ['quantum', 'muon_particle', 'cosmology', 'gravity']
    y = np.array([1., 2., 3., 4.])
    ref = y + [2., 1., 3., 2.]
    pred = y + [.5, .2, 1., .3]
    covariance = np.array([[2., .3, .1, .2], [.3, 1., -.1, .1],
                           [.1, -.1, 3., .4], [.2, .1, .4, 2.]])
    score = joint_residuals(y, ref, pred, covariance, groups, required_groups=groups)
    values, vectors = np.linalg.eigh(covariance)
    independent = [float(np.sqrt(np.sum((vectors.T @ (p-y))**2 / values)/4))
                   for p in (ref, pred)]
    error = max(abs(score['overall'][key]-value) for key, value in
                zip(('baseline_rmse', 'candidate_rmse'), independent))
    assert error < 1e-12
    guard = joint_residuals([0.]*4, [10., 10., 10., 1.], [0., 0., 0., 2.],
                           np.eye(4), groups, required_groups=groups)
    assert guard['overall']['strictly_reduced'] and not guard['joint_arithmetic_reduction']
    unchanged = joint_residuals([0.]*4, [2.]*4, [1., 1., 1., 2.],
                               np.eye(4), groups, required_groups=groups)
    assert not unchanged['joint_arithmetic_reduction']
    cutoff = [{"eta": eta, "cutoff": t, **relative_flat_action(t, eta=eta)}
              for eta in (0., .1) for t in (1e-6, 1e-12)]
    gr = [cosmological_counterexample(chi) for chi in (np.pi/4, np.pi/3, np.pi/2)]
    free = [free_embedding(.1, x) for x in (np.zeros(4), np.array([.1, -.2, .3, -.1]))]
    partial = partial_score()
    manifests = {domain: asdict(validate_manifest_path(
        ROOT / 'experiments' / 'preregistration' / f'{domain}_future_holdout_v2.json'))
        for domain in ('quantum', 'cosmology')}
    files = [Path(__file__), ROOT/'verify/dimension_joint_candidate.py',
             ROOT/'verify/ce_async_projection_memory.py',
             ROOT/'verify/ce_embedding_einstein_variation.py',
             ROOT/'verify/ce_latest_common_epsilon_score.py',
             ROOT/'experiments/preregistration/validate_holdout_manifest.py']
    return {
        'role': 'code_audit_not_physical_confirmation',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__},
        'entrypoint_sha256_not_transitive_input_manifest': {
            str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files},
        'synthetic_joint': score, 'eigen_rmse': independent, 'eigen_max_error': error,
        'synthetic_degradation_guard': guard, 'synthetic_unchanged_block': unchanged,
        'dimension_cutoff_scan': cutoff,
        'AM1': {'exact': exact_audit(), 'numeric': numeric_audit()},
        'GR6': {'counterexamples': gr, 'free_embedding': free},
        'seven_row_partial_score': partial,
        'holdout_artifact_validation': manifests,
        'full_observation_joint_rmse': None, 'scientific_success': False,
    }


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, ensure_ascii=True))
