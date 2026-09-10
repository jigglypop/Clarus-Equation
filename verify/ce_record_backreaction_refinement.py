"""Resolve chapter 77's two failed velocity convergence checks.

The first-pass result remains unchanged, including its failed gates. This
supplement refines only those cases; it does not change any physical input.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp

import ce_record_backreaction as base


def refine(row):
    mass, eta, theta = row['mass_over_V'], row['eta'], row['theta']
    y0, _ = base.initial_state(mass, eta, theta)
    base.rhs(0., y0, mass)
    old_fine, info = base.integrate(y0, mass, 1)
    start = time.perf_counter()
    solution = solve_ivp(lambda t, y: base.rhs(t, y, mass), (0., 60.), y0,
                         method='DOP853', t_eval=base.TIMES, rtol=2e-12, atol=2e-15)
    assert solution.success, solution.message
    new = solution.y
    energy, total, power, kinetic, residual = base.diagnostics(new, mass)
    eold, _, pold, _, _ = base.diagnostics(old_fine, mass)
    comparisons = {
        'fields_over_Aref': base.maximum(new[:5]-old_fine[:5]),
        'velocities_over_mref_Aref': base.maximum(new[5:]-old_fine[5:]),
        'qpower_absolute': base.maximum(power-pold),
        'Fi_partition_energy_over_initial_total': base.maximum((energy-eold)/total[0]),
        'total_energy_relative_drift': base.maximum(total/total[0]-1),
        'original_saved_qpower_trace_reproduction': base.maximum(pold[::10]-np.array(row['trace']['qpower']))}
    late = base.TIMES >= 50
    out = {'species': row['species'], 'mass_over_V': mass, 'eta': eta, 'theta': theta,
           'initial_failed_comparisons': row['numerical_comparisons'],
           'refined_comparisons': comparisons,
           'integration': [info, {'method': 'DOP853', 'rtol': 2e-12, 'atol': 2e-15,
                                  'function_evaluations': solution.nfev, 'seconds': time.perf_counter()-start}],
           'numerical_gate_passed': max(comparisons.values()) < 1e-6,
           'qpower_peak_sampled': float(max(power)),
           'qpower_late_min_max_sampled': [float(min(power[late])), float(max(power[late]))],
           'threshold_upward_tau_linear_interpolation': base.crossing_times(power, True),
           'threshold_downward_tau_linear_interpolation': base.crossing_times(power, False),
           'late_threshold_maintained_at_all_sampled_times': bool(np.all(power[late] >= 1)),
           'Higgs_Fi_partition_fraction_peak': float(max(energy[4]/total)),
           'Higgs_Fi_partition_fraction_late_min_max': [float(min(energy[4, late]/total[late])), float(max(energy[4, late]/total[late]))],
           'record_free_equation_scaled_residual_peak': float(max(residual)),
           'heavy_record_amplitude_over_initial_light_peak': float(max(abs(new[3]))*base.AREF/np.sqrt(base.N0/(2*mass)))}
    print(json.dumps({k: out[k] for k in ['species', 'eta', 'theta', 'refined_comparisons', 'numerical_gate_passed']}), flush=True)
    return out


def main():
    folder = Path(__file__).resolve().parent
    source = folder/'ce_record_backreaction.json'
    original = json.loads(source.read_text(encoding='utf-8'))
    assert original['run_complete'] and len(original['cases']) == 8
    assert original['preregistration_sha256'] == base.PREREG
    chapter = next((folder.parent/'paper').glob('06_*/77_*.md'))
    assert hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 77.2')[0].encode()).hexdigest() == base.PREREG
    for name, digest in original['source_hashes'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
    failed = [r for r in original['cases'] if not r['numerical_gate_passed']]
    result = {'schema_version': 1, 'candidate': 'CE-UR4-D3', 'scientific_success': False,
              'full_joint_rmse': None, 'new_observational_inputs_or_fits': False,
              'preregistration_sha256': base.PREREG,
              'source_hashes': {**original['source_hashes'], Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'initial_result_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'environment': original['environment'],
              'refinement_scope': 'only failed cases; original equations, state, parameters, interval, threshold and comparisons unchanged',
              'initial_passed_case_count': 8-len(failed), 'refined_cases': [], 'run_complete': False}
    destination = folder/'ce_record_backreaction_refinement.json'
    def save():
        result['refined_cases'].sort(key=lambda r: r['mass_over_V'])
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        temporary.replace(destination)
    save()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for job in as_completed([executor.submit(refine, row) for row in failed]):
            result['refined_cases'].append(job.result())
            save()
    result['run_complete'] = len(result['refined_cases']) == len(failed)
    result['combined_numerical_gate_passed'] = result['run_complete'] and all(r['numerical_gate_passed'] for r in result['refined_cases'])
    result['actual_instrument_derived'] = False
    save()
    assert result['combined_numerical_gate_passed'], 'precise pointer judgment remains withheld'
    print('all eight cases cleared, including the two separately preserved numerical refinements', flush=True)


if __name__ == '__main__':
    main()
