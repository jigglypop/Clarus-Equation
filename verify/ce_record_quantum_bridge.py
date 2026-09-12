"""Chapter 79: full-tensor/Fock witness against exact five-field quantum closure.

This computes a bare finite-volume Hamiltonian matrix element, not an on-shell
amplitude, a closed two-state evolution, or a physical measurement instrument.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import ast
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy import sparse
import sympy as sy

from ce_singlet_record_observability import build_model
from ce_supersymmetric_record import f_terms


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / 'paper/06_QFT_재설계/79_기록장의_양자_누설과_사건_사상의_연결.md'
PREREG = '005567e9c9a1d3d7763e60372ff0b1794649fbb451f82c287703b918777599aa'
M0 = 1e-4
MP, MM = M0 * np.array([5 + np.pi/2, 5 - np.pi/2])
VOLUME = MM**-3
TOL = 1e-10


def maximum(x):
    return float(np.max(np.abs(x)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symbolic_model():
    z, t, sp, up, sm, um, q, mp, mm = sy.symbols('z t sp up sm um q mp mm')
    x = [z, t, sp, up, sm, um, q]
    a = 1/(2*sy.sqrt(15))
    w = (-z*z/2-5*t*t/2+a*z**3/3+3*a*z*t*t
         -(3*a*z+t/2)*q*q/2)
    for s, u, m in [(sp, up, mp), (sm, um, mm)]:
        w += m*s*s/2+(m-4)*u*u/2+a*z*s*s+3*a*z*u*u+6*a*t*s*u
    f = sy.Matrix([sy.diff(w, v) for v in x])
    h = f.jacobian(x)
    c = np.array([[[complex(sy.diff(h[i,j], v)) for v in x]
                   for j in range(7)] for i in range(7)])
    barred = sy.symbols('zb tb spb upb smb umb qb')
    substitution = dict(zip(x, barred))
    pot = sy.expand(sum(v*v.xreplace(substitution) for v in f))
    coefficient = sy.Poly(pot, *x, *barred).coeff_monomial(sp**2*barred[4]**2)
    assert coefficient == sy.Rational(1,60)
    zero_minus_force = sy.simplify(sy.diff(pot, barred[4]).subs({sm:0,um:0,barred[4]:0,barred[5]:0}))
    zero_minus_triplet = sy.simplify(sy.diff(pot, barred[5]).subs({sm:0,um:0,barred[4]:0,barred[5]:0}))
    assert zero_minus_force == zero_minus_triplet == 0
    return x, (mp, mm), w, f, h, c, coefficient


def embedding():
    e = np.zeros((82,7), complex)
    e[11,0], e[10,1] = 1, 1
    for s, u, sign in [(2,3,1),(4,5,-1)]:
        e[35,s],e[59,s] = 1/np.sqrt(2), sign/np.sqrt(2)
        e[34,u],e[58,u] = 1/np.sqrt(2), sign/np.sqrt(2)
    e[75,6], e[80,6] = 1/np.sqrt(2), -1/np.sqrt(2)
    return e


def algebra():
    x, masses, polynomial, fs, hs, cs, coefficient = symbolic_model()
    _, _, reps, matrix, cubic, vacuum = build_model(M0)
    fv, w0 = f_terms(vacuum, matrix, cubic)
    e = embedding()
    errors = {'vacuum_F_absolute':maximum(fv),
              'kinetic_embedding':maximum(e.conj().T@e-np.eye(7))}
    full_c = np.einsum('ijk,ia,jb,kc->abc',cubic,e,e,e,optimize=True)
    errors['independent_seven_field_cubic'] = maximum(full_c-cs)
    f_linear = w0@e
    f_quadratic = np.einsum('ijk,ja,kb->iab',cubic,e,e,optimize=True)
    errors['all_transverse_F_linear_coefficients'] = maximum(f_linear-e@(e.conj().T@f_linear))
    errors['all_transverse_F_quadratic_coefficients'] = maximum(
        f_quadratic-np.einsum('ic,cab->iab',e,full_c,optimize=True))
    errors['all_projected_gauge_coefficients'] = maximum(
        np.einsum('ia,kij,jb->kab',e.conj(),reps,e,optimize=True))
    errors['vacuum_gauge_linear_coefficients'] = maximum(
        np.einsum('i,kij,jb->kb',vacuum.conj(),reps,e,optimize=True))
    sf = sy.lambdify([x, masses], fs, 'numpy')
    sh = sy.lambdify([x, masses], hs, 'numpy')
    reduced_mass = np.array(hs.subs(dict.fromkeys(x,0)).subs(dict(zip(masses,[MP,MM]))),complex)
    errors['independent_seven_field_mass'] = maximum(e.T@w0@e-reduced_mass)
    rng = np.random.default_rng(7901)
    for _ in range(4):
        xx = .003*(rng.normal(size=7)+1j*rng.normal(size=7))
        delta = e@xx
        ff = w0@delta+.5*np.einsum('ijk,j,k->i',cubic,delta,delta,optimize=True)
        ww = w0+np.einsum('ijk,k->ij',cubic,delta,optimize=True)
        rf = np.array(sf(xx,[MP,MM])).ravel()
        rw = np.array(sh(xx,[MP,MM]))
        for label, value in [
            ('F_closure_relative',(ff-e@rf)/maximum(ff)),
            ('force_closure_relative',(ww.conj().T@ff-e@(rw.conj().T@rf))/maximum(ww.conj().T@ff)),
            ('D_zero_absolute',np.einsum('i,aij,j->a',(vacuum+delta).conj(),reps,vacuum+delta,optimize=True))]:
            errors[label] = max(errors.get(label,0),maximum(value))
    # Execute only the already inspected symbolic function, without importing numba.
    original = ROOT/'verify/ce_record_backreaction.py'
    function = next(n for n in ast.parse(original.read_text(encoding='utf-8')).body
                    if isinstance(n,ast.FunctionDef) and n.name=='cubic_polynomial')
    namespace = {'np':np, 'sy':sy}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(original),'exec'),namespace)
    c5, f5, h5 = namespace['cubic_polynomial']()
    for indices,mass in [([0,1,2,3,6],MP),([0,1,4,5,6],MM)]:
        errors['chapter77_cubic_recovery'] = max(errors.get('chapter77_cubic_recovery',0),
            maximum(full_c[np.ix_(indices,indices,indices)]-c5))
        for _ in range(2):
            xx = .003*(rng.normal(size=5)+1j*rng.normal(size=5))
            yy = np.zeros(7,complex)
            yy[indices] = xx
            errors['chapter77_F_recovery'] = max(errors.get('chapter77_F_recovery',0),
                maximum(np.array(sf(yy,[MP,MM])).ravel()[indices]-np.array(f5(xx,mass)).ravel()))
    source_p = .5*np.einsum('ijk,j,k->i',cubic,e[:,2],e[:,2],optimize=True)
    source_m = .5*np.einsum('ijk,j,k->i',cubic,e[:,4],e[:,4],optimize=True)
    tensor_coefficient = np.vdot(source_m,source_p)
    errors['full_potential_mixed_coefficient_relative'] = abs(tensor_coefficient-complex(coefficient))/float(coefficient)
    assert max(errors.values()) < TOL, errors
    return {'errors':errors,'max_error':max(errors.values()),
            'potential_mixed_coefficient_exact':str(coefficient),
            'potential_mixed_coefficient_tensor':float(tensor_coefficient.real),
            'classical_zero_other_species_force_exact':True,
            'seven_complex_fields_classically_closed_only':True}, float(coefficient)


def ladder(cutoff, mode):
    size = cutoff+1
    a = sparse.diags(np.sqrt(np.arange(1,size)),1,shape=(size,size),format='csr')
    out = sparse.csr_matrix([[1.]])
    for k in range(4):
        out = sparse.kron(out,a if k==mode else sparse.eye(size,format='csr'),format='csr')
    return out


def fock_checks(coefficient):
    gamma = coefficient/(2*MP*MM*VOLUME)
    output = []
    for cutoff in [3,4,5]:
        ladders = [ladder(cutoff,k) for k in range(4)]
        sp = (ladders[0]+ladders[1].T)/np.sqrt(2*MP*VOLUME)
        sm = (ladders[2]+ladders[3].T)/np.sqrt(2*MM*VOLUME)
        pair = sp@sp+sm@sm
        contact = coefficient*VOLUME*(pair.T@pair)
        size = cutoff+1
        ai = np.ravel_multi_index((2,0,0,0),(size,)*4)
        bi = np.ravel_multi_index((0,0,2,0),(size,)*4)
        measured = float(contact[bi,ai])
        matrix_error = abs(measured/gamma-1)
        phase_results = []
        for phase in [0.,np.pi/2,np.pi,3*np.pi/2]:
            psi = np.zeros(size**4,complex)
            psi[ai],psi[bi] = 1/np.sqrt(2),np.exp(1j*phase)/np.sqrt(2)
            derivative = -1j*(contact@psi)
            probability_dot = 2*np.real(psi.conj()*derivative)
            current_ba = 2*np.imag(psi[bi].conjugate()*measured*psi[ai])
            current_ab = -current_ba
            rate_ba,rate_ab = max(current_ba,0)/.5,max(current_ab,0)/.5
            master_b = rate_ba*.5-rate_ab*.5
            coherence_energy = 2*np.real(psi[bi].conjugate()*measured*psi[ai])
            expected_j = -gamma*np.sin(phase)
            expected_e = gamma*np.cos(phase)
            err = max(abs(current_ba-expected_j),abs(probability_dot[bi]-current_ba),
                      abs(probability_dot[ai]+current_ba),abs(master_b-current_ba),
                      abs(coherence_energy-expected_e))/gamma
            assert err < TOL, (cutoff,phase,err)
            phase_results.append({'phase_over_pi':float(phase/np.pi),
                'J_BA_over_mref':float(current_ba/MM),
                'coherence_energy_over_mref':float(coherence_energy/MM),
                'rate_A_to_B_over_mref':float(rate_ba/MM),
                'rate_B_to_A_over_mref':float(rate_ab/MM),
                'independent_current_energy_error':float(err)})
        assert matrix_error < TOL
        output.append({'occupation_cutoff':cutoff,'matrix_dimension':size**4,
            'matrix_element_over_V':measured,'matrix_element_over_mref':measured/MM,
            'matrix_element_relative_error':matrix_error,'phases':phase_results})
    return {'m_plus_over_V':float(MP),'m_minus_over_V':float(MM),
            'box_volume_times_V3':float(VOLUME),'analytic_matrix_element_over_V':gamma,
            'analytic_matrix_element_over_mref':gamma/MM,
            'short_time_leakage_lower_coefficient_in_tau2':(gamma/MM)**2,
            'cutoff_checks':output,
            'bare_Fock_matrix_element_not_on_shell_amplitude':True,
            'spectator_condition':'same normalized spectator on both sides; no massless zero-mode free vacuum claimed',
            'contact_matrix_used_only_for_selected_elements_and_instantaneous_current':True,
            'two_state_subspace_assumed_invariant':False,
            'sampled_trajectories':0,'time_evolution_runs':0}


def phase_identification_check():
    s,eps = 1.,.1
    cycle = np.roll(np.eye(3),1,axis=0)
    k = M0*(5*np.eye(2)+np.pi/2*np.array([[0,1],[1,0]]))
    frozen = np.linalg.eigvalsh(k@k)
    rows = []
    for theta in [0.,np.pi]:
        mass_squared = s*np.eye(3)+eps*(np.exp(1j*theta/3)*cycle+np.exp(-1j*theta/3)*cycle.T)
        determinant = np.linalg.det(mass_squared)
        expected = s**3-3*s*eps**2+2*eps**3*np.cos(theta)
        u = np.diag([1,np.exp(1j*theta)])
        values = np.linalg.eigvalsh(u.conj().T@(k@k)@u)
        err = maximum((values-frozen)/frozen)
        assert abs(determinant-expected) < TOL and err < TOL
        rows.append({'phase_over_pi':float(theta/np.pi),
                     'cyclic_mass_squared_determinant':float(determinant.real),
                     'record_rephasing_spectral_relative_error':err})
    return {'checks':rows,'direct_state_phase_equals_action_phase_rejected':True,
            'arbitrary_larger_common_action_ruled_out':False}


def main():
    started = time.perf_counter()
    prereg = CHAPTER.read_text(encoding='utf-8').split('## 79.2')[0]
    assert hashlib.sha256(prereg.encode()).hexdigest()==PREREG, 'Preregistration changed'
    algebra_result, coefficient = algebra()
    fock = fock_checks(coefficient)
    phase = phase_identification_check()
    files = [Path(__file__),ROOT/'verify/ce_singlet_record_observability.py',
             ROOT/'verify/ce_supersymmetric_record.py',ROOT/'verify/ce_simple_group_record.py',
             ROOT/'verify/ce_record_backreaction.py',ROOT/'verify/ce_equivariant_record_events.py']
    result = {'candidate':'CE-BR1','preregistration_sha256':PREREG,
        'source_sha256':{p.name:sha(p) for p in files},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,
                       'scipy':scipy.__version__,'sympy':sy.__version__},
        'algebra':algebra_result,'quantum_matrix_element':fock,'phase_identification':phase,
        'verdict':{'exact_five_field_free_vacuum_quantum_closure':False,
                   'seven_field_quantum_closure_proven':False,
                   'physical_instrument_constructed':False,
                   'independent_algebra_and_Fock_checks_passed':True,
                   'scientific_success':False,'full_joint_rmse':None},
        'runtime_seconds':time.perf_counter()-started}
    path = Path(__file__).with_suffix('.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(path),'max_algebra_error':algebra_result['max_error'],
                      'H_BA_over_mref':fock['analytic_matrix_element_over_mref'],
                      'verdict':result['verdict'],'seconds':result['runtime_seconds']},ensure_ascii=False))


if __name__=='__main__':
    main()
