"""CE-LC1: positive cone Lorentz algebra, weighted frames and a flatness gate.

Frame changes, physical filters and a spacetime construction are separate.
Spinor factors are additional inputs, not identified with weak gauge factors.
"""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from ce_color_covariant_record import HF, HS, HA, REC, setup, unitary


I = np.eye(2, dtype=complex)
SX = np.array([[0,1],[1,0]], complex)
SY = np.array([[0,-1j],[1j,0]], complex)
SZ = np.diag([1.,-1.]).astype(complex)
SIGMA = [I, SX, SY, SZ]
ETA = np.diag([-1.,1.,1.,1.])
RHO = (I+.3*SX+.2*SY-.4*SZ)/2
EFFECT = (I+SX)/2
TOL = 1e-12


def error(a):
    return float(np.max(abs(a)))


def sqrt_positive(a):
    values, vectors = np.linalg.eigh(a)
    assert min(values) > -TOL
    return (vectors*np.sqrt(np.maximum(values, 0)))@vectors.conj().T


def normalize(a):
    return a/np.trace(a).real


def cone_audit(rapidity, angle):
    rotation = np.cos(angle/2)*I-1j*np.sin(angle/2)*SY
    boost = np.diag([np.exp(rapidity/2), np.exp(-rapidity/2)])
    s = rotation@boost
    inverse = np.linalg.inv(s)
    lorentz = np.array([[np.trace(a@s@b@s.conj().T).real/2 for b in SIGMA] for a in SIGMA])
    errors = {'spinor_determinant': float(abs(np.linalg.det(s)-1)),
              'Lorentz_form': error(lorentz.T@ETA@lorentz-ETA),
              'Lorentz_determinant': float(abs(np.linalg.det(lorentz)-1))}
    assert lorentz[0,0] >= 1-TOL
    states = [RHO, (I+SZ)/2, (I-SZ)/2, I/2]
    for rho in states:
        transformed = s@rho@s.conj().T
        vector = np.array([np.trace(a@rho).real/2 for a in SIGMA])
        changed = np.array([np.trace(a@transformed).real/2 for a in SIGMA])
        errors['cone_determinant'] = max(errors.get('cone_determinant',0.), float(abs(np.linalg.det(rho)+vector@ETA@vector)))
        errors['determinant_preserved'] = max(errors.get('determinant_preserved',0.), float(abs(np.linalg.det(transformed)-np.linalg.det(rho))))
        errors['four_vector_transform'] = max(errors.get('four_vector_transform',0.), error(changed-lorentz@vector))
        assert min(np.linalg.eigvalsh(transformed)) > -TOL and changed[0] > 0
    g = inverse.conj().T@inverse
    rho_new = s@RHO@s.conj().T
    effect_new = inverse.conj().T@EFFECT@inverse
    p_new = s@EFFECT@inverse
    root_g = sqrt_positive(g)
    effective_unitary = root_g@s
    physical_density = root_g@rho_new@root_g
    original_probability = float(np.trace(EFFECT@RHO).real)
    weighted_probability = float(np.trace(effect_new@rho_new).real)
    errors.update({
        'weighted_trace': float(abs(np.trace(g@rho_new)-1)),
        'effects_sum_to_G': error(effect_new+inverse.conj().T@(I-EFFECT)@inverse-g),
        'projector_idempotence': error(p_new@p_new-p_new),
        'projector_G_self_adjoint': error(g@p_new-p_new.conj().T@g),
        'effect_from_weighted_projector': error(effect_new-g@p_new),
        'probability_preserved': abs(weighted_probability-original_probability),
        'standardized_unitary': error(effective_unitary.conj().T@effective_unitary-I),
        'density_spectrum_preserved': error(np.linalg.eigvalsh(physical_density)-np.linalg.eigvalsh(RHO)),
    })
    assert min(np.linalg.eigvalsh(effect_new)) > -TOL
    assert min(np.linalg.eigvalsh(g-effect_new)) > -TOL
    plus, minus = states[1:3]
    filtered_mix = normalize(s@(I/2)@s.conj().T)
    mix_of_filtered = (normalize(s@plus@s.conj().T)+normalize(s@minus@s.conj().T))/2
    nonlinearity = float(np.sum(abs(np.linalg.eigvalsh(filtered_mix-mix_of_filtered)))/2)
    m = np.exp(-rapidity/2)*s
    n = np.diag([0., np.sqrt(1-np.exp(-2*rapidity))])
    isometry = np.vstack([m,n])
    success_probability = float(np.trace(m@(I/2)@m.conj().T).real)
    errors.update({'normalized_filter_nonlinearity_formula': abs(nonlinearity-np.tanh(rapidity)/2),
                   'Kraus_completeness': error(m.conj().T@m+n.conj().T@n-I),
                   'isometry_information_preservation': error(isometry.conj().T@isometry-I),
                   'filter_success_probability': abs(success_probability-(1+np.exp(-2*rapidity))/2)})
    assert max(errors.values()) < TOL, errors
    return s, g, {'rapidity': rapidity, 'rotation': angle, 'errors': errors,
                   'ordinary_trace_after_frame_change': float(np.trace(rho_new).real),
                   'fixed_trace_operator_defect': error(s.conj().T@s-I),
                   'physical_probability': weighted_probability,
                   'normalized_filter_nonlinearity_trace_distance': nonlinearity,
                   'filter_success_probability_for_maximally_mixed': success_probability}


def record_audit(s, g, rapidity, angle):
    _, _, transition = setup()
    initial = np.zeros(32, complex)
    initial[12] = initial[16] = 1/np.sqrt(2)
    psi = np.kron(I, unitary(transition, np.pi/4))@initial
    transform = np.kron(s,np.eye(16))
    metric = np.kron(g,np.eye(16))
    transformed = transform@psi
    record = np.kron(I,np.kron(np.eye(4),REC))
    system_h = np.kron(I,np.kron(HS,np.eye(4)))
    apparatus_h = np.kron(I,np.kron(np.eye(4),HA))
    total_h = np.kron(I,HF)
    values = [float(np.vdot(transformed, metric@op@transformed).real)
              for op in [np.eye(32),record,system_h,apparatus_h,total_h]]
    naive = float(np.vdot(transformed,record@transformed).real/np.vdot(transformed,transformed).real)
    errors = {'weighted_values': error(np.array(values)-[1.,.25,2.,1.,3.]),
              'all_state_frame_isometry': error(transform.conj().T@metric@transform-np.eye(32)),
              'all_state_record_covariance': error(transform.conj().T@metric@record@transform-record),
              'all_state_energy_covariance': error(transform.conj().T@metric@total_h@transform-total_h),
              'naive_probability_formula': abs(naive-(1+np.tanh(rapidity))/4)}
    assert max(errors.values()) < TOL, errors
    return {'rapidity': rapidity, 'rotation': angle, 'weighted_record_probability': values[1],
            'weighted_system_energy': values[2], 'weighted_apparatus_energy': values[3],
            'weighted_total_free_energy': values[4], 'naive_renormalized_record_probability': naive,
            'errors': errors}


def embedding(q):
    t, xi, u, v = q
    r = np.exp(xi)
    y = np.array([3+r*np.sinh(t),r*np.cosh(t),u,v])
    j = np.array([[r*np.cosh(t),r*np.sinh(t),0,0],
                  [r*np.sinh(t),r*np.cosh(t),0,0], [0,0,1,0], [0,0,0,1]])
    h = np.zeros((4,4,4))
    h[0,0,0] = h[0,1,1] = r*np.sinh(t)
    h[0,0,1] = h[0,1,0] = r*np.cosh(t)
    h[1,0,0] = h[1,1,1] = r*np.cosh(t)
    h[1,0,1] = h[1,1,0] = r*np.sinh(t)
    return y, j, h


def connection(q):
    _, j, h = embedding(q)
    return np.einsum('la,amn->lmn', np.linalg.inv(j),h)


def spacetime_audit(q):
    y, j, h = embedding(q)
    state_matrix = sum(value*sigma for value,sigma in zip(y,SIGMA))
    assert min(np.linalg.eigvalsh(state_matrix)) > 0
    g = j.T@ETA@j
    gamma = connection(q)
    derivative_metric = np.array([h[:,:,k].T@ETA@j+j.T@ETA@h[:,:,k] for k in range(4)])
    inverse_g = np.linalg.inv(g)
    from_metric = np.zeros((4,4,4))
    for lam, mu, nu in np.ndindex(4,4,4):
        from_metric[lam,mu,nu] = sum(inverse_g[lam,r]*(derivative_metric[mu,r,nu]+derivative_metric[nu,r,mu]-derivative_metric[r,mu,nu])/2 for r in range(4))
    errors = {'metric_formula': error(g-np.diag([-np.exp(2*q[1]),np.exp(2*q[1]),1.,1.])),
              'Christoffel_two_constructions': error(gamma-from_metric)}
    assert max(errors.values()) < TOL
    curvatures=[]
    for step in (.001,.0005):
        dgamma = np.array([(connection(q+step*np.eye(4)[k])-connection(q-step*np.eye(4)[k]))/(2*step) for k in range(4)])
        riemann=np.zeros((4,4,4,4))
        for lam, sigma, mu, nu in np.ndindex(4,4,4,4):
            riemann[lam,sigma,mu,nu] = dgamma[mu,lam,nu,sigma]-dgamma[nu,lam,mu,sigma]+sum(gamma[lam,mu,k]*gamma[k,nu,sigma]-gamma[lam,nu,k]*gamma[k,mu,sigma] for k in range(4))
        maximum=error(riemann)
        assert maximum < 1e-9
        curvatures.append({'step':step,'maximum_Riemann_residual':maximum})
    return {'coordinates':q.tolist(),'metric_eigenvalues':np.linalg.eigvalsh(g).tolist(),
            'minimum_positive_state_eigenvalue':float(min(np.linalg.eigvalsh(state_matrix))),
            'Jacobian_determinant':float(np.linalg.det(j)),
            'maximum_Christoffel_component':float(np.max(abs(gamma))),
            'curvature_checks':curvatures,'errors':errors}


def main():
    cone,record=[],[]
    for rapidity in (0.,.3,.7,1.2):
        for angle in (0.,.4):
            s,g,row=cone_audit(rapidity,angle)
            cone.append(row)
            record.append(record_audit(s,g,rapidity,angle))
    spacetime=[spacetime_audit(np.array(q)) for q in [(0.,0.,.1,-.1),(.2,.3,.1,-.1),(-.2,-.3,.1,-.1)]]
    here=Path(__file__).resolve()
    out={'candidate':'CE-LC1','algebra_tolerance':TOL,'curvature_tolerance':1e-9,
         'cone_and_frames':cone,'coupled_record':record,'state_coordinate_metric':spacetime,
         'maximum_algebra_error':max(max(r['errors'].values()) for r in cone+record+spacetime),
         'maximum_curvature_residual':max(c['maximum_Riemann_residual'] for r in spacetime for c in r['curvature_checks']),
         'environment':{'python':platform.python_version(),'numpy':np.__version__},
         'source_sha256':{name:hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
                           for name in [here.name,'ce_color_covariant_record.py','ce_isometric_color_frame.py']},
         'limits':['Hermitian 2x2 factor, observer pairing and spacetime map are supplied.',
                   'A passive frame change is different from a normalized physical filter.',
                   'Record energy accounting is the earlier supplied finite apparatus model.',
                   'A single invertible four-coordinate state map produces a flat pullback metric.',
                   'No Einstein action, local spacetime event law, actual outcome or four-force unification is obtained.']}
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'candidate':out['candidate'],'maximum_algebra_error':out['maximum_algebra_error'],
                      'maximum_curvature_residual':out['maximum_curvature_residual'],
                      'record':[r for r in record if r['rotation']==0.],
                      'spacetime':spacetime},indent=2))


if __name__=='__main__':
    main()
