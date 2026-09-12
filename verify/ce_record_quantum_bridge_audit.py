"""Independent SU(5) trace and Gaussian-wavefunction audit of chapter 79.

No import of the generator, its reduced polynomial, or its Fock matrices.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from numpy.polynomial.hermite import hermgauss
import sympy as sy


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'verify/ce_record_quantum_bridge.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    for name,digest in data['source_sha256'].items():
        assert hashlib.sha256((root/'verify'/name).read_bytes()).hexdigest()==digest, name
    chapter = next((root/'paper/06_QFT_재설계').glob('79_*.md'))
    frozen = chapter.read_text(encoding='utf-8').split('## 79.2')[0]
    assert hashlib.sha256(frozen.encode()).hexdigest()==data['preregistration_sha256']
    ty = sy.diag(-2,-2,-2,3,3)/(2*sy.sqrt(15))
    coefficient = sy.simplify(2*(sy.trace(ty**4)-sy.trace(ty**2)**2/5))
    assert coefficient==sy.Rational(1,60)
    mp,mm = 1e-4*np.array([5+np.pi/2,5-np.pi/2])
    volume = mm**-3
    analytic = float(coefficient)/(2*mp*mm*volume)
    results = []
    for order in [4,5,7]:
        nodes,weights = hermgauss(order)
        ux,vx,uy,vy = np.meshgrid(nodes,nodes,nodes,nodes,indexing='ij')
        ww = np.einsum('a,b,c,d->abcd',weights,weights,weights,weights)/np.pi**2
        # Circular two-particle states in four real canonical oscillator coordinates.
        state_a = (ux-1j*vx)**2/np.sqrt(2)
        state_b = (uy-1j*vy)**2/np.sqrt(2)
        sp = (ux+1j*vx)/np.sqrt(2*mp*volume)
        sm = (uy+1j*vy)/np.sqrt(2*mm*volume)
        potential = float(coefficient)*volume*abs(sp**2+sm**2)**2
        element = np.sum(ww*state_b.conj()*potential*state_a)
        norm_a,norm_b = np.sum(ww*abs(state_a)**2),np.sum(ww*abs(state_b)**2)
        overlap = np.sum(ww*state_b.conj()*state_a)
        error = max(abs(element/analytic-1),abs(norm_a-1),abs(norm_b-1),abs(overlap))
        assert error < 1e-10, (order,error)
        results.append({'nodes_per_real_coordinate':order,
            'matrix_element_real_over_V':float(element.real),
            'matrix_element_imag_over_V':float(element.imag),
            'normalized_max_error':float(error)})
    matrix = data['quantum_matrix_element']
    assert abs(matrix['analytic_matrix_element_over_V']/analytic-1)<1e-10
    for row in matrix['cutoff_checks']:
        assert abs(row['matrix_element_over_V']/analytic-1)<1e-10
        for phase in row['phases']:
            phi = phase['phase_over_pi']*np.pi
            assert abs(phase['J_BA_over_mref']/(analytic/mm)+np.sin(phi))<1e-10
            assert abs(phase['coherence_energy_over_mref']/(analytic/mm)-np.cos(phi))<1e-10
    result = {'candidate':'CE-BR1-independent-audit',
        'preregistration_sha256':data['preregistration_sha256'],
        'generator_result_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'sympy':sy.__version__},
        'SU5_trace_coefficient_exact':str(coefficient),'Gaussian_checks':results,
        'H_BA_over_mref':float(analytic/mm),
        'audit_passed':True,'scientific_success':False,'full_joint_rmse':None}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'audit_passed':True,'max_Gaussian_error':max(r['normalized_max_error'] for r in results),
                      'H_BA_over_mref':analytic/mm}))


if __name__=='__main__':
    main()
