"""CE-AM3: base-point embedding and coherent P/Q records, without outcome selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ce_common_isotropic_frame import weak_setup, block_diag


def partial_system(state):
    matrix = state.reshape(4, 4)
    return matrix @ matrix.conj().T


def run():
    eye = np.eye(2)
    x = np.array([[0., 1.], [1., 0.]])
    p = np.diag([1., 1., 1., 0.])
    q = np.eye(4) - p
    h = np.eye(4) + 4*q
    ks = [np.kron(x, eye), np.kron(eye, x)]
    us = [(np.eye(4)-1j*k)/np.sqrt(2) for k in ks]
    ws = [np.kron(p, np.eye(4)) + np.kron(q, flip)
          for flip in [np.kron(x, eye), np.kron(eye, x)]]
    events = [w @ np.kron(u, np.eye(4)) for w,u in zip(ws,us)]
    state = np.eye(4)[:,1].astype(complex)
    initial = np.kron(state, np.eye(4)[:,0])
    errors = {}
    # At q=0 all oscillator states are vacuum; keeping this support is exact for P(0).
    # The three orthogonal flavor channels remain, independently of their right basis.
    _, weak = weak_setup()
    vg = block_diag([np.ones((1,1)), weak, np.eye(4)[:,:3]])
    frame = np.kron(vg, np.eye(3))
    pg = frame @ frame.conj().T
    j = np.zeros((63,4), complex)
    for c in range(4):
        j[3*(17+c),c] = 1
    errors['embedding_isometry'] = float(np.linalg.norm(j.conj().T@j-np.eye(4)))
    errors['embedding_projection'] = float(np.linalg.norm(pg@j-j@p))
    errors['bare_event_commutator'] = float(np.linalg.norm(us[0]@us[1]-us[1]@us[0]))
    for i,w in enumerate(ws):
        errors[f'record_unitary_{i}'] = float(np.linalg.norm(w.conj().T@w-np.eye(16)))
        hh = np.kron(h,np.eye(4))
        errors[f'record_energy_commutator_{i}'] = float(np.linalg.norm(w@hh-hh@w))
        errors[f'kraus_completeness_{i}'] = float(np.linalg.norm(
            sum((r@us[i]).conj().T@(r@us[i]) for r in [p,q])-np.eye(4)))
    orders = []
    finals = []
    for order in [(0,1),(1,0)]:
        full = initial.copy()
        rho = np.outer(state,state.conj())
        for i in order:
            full = events[i]@full
            rho = sum(r@us[i]@rho@us[i].conj().T@r for r in [p,q])
        reduced = partial_system(full)
        errors[f'instrument_vs_full_{order}'] = float(np.linalg.norm(rho-reduced))
        errors[f'normalization_{order}'] = float(abs(np.vdot(full,full)-1))
        recovered = events[order[0]].conj().T@events[order[1]].conj().T@full
        errors[f'global_inverse_{order}'] = float(np.linalg.norm(recovered-initial))
        finals.append(reduced)
        orders.append({'order':[i+1 for i in order],
                       'system_probabilities':np.diag(reduced).real.tolist(),
                       'record_probabilities':np.sum(abs(full.reshape(4,4))**2,axis=0).tolist(),
                       'system_density_real':reduced.real.tolist(),
                       'system_density_imag':reduced.imag.tolist(),
                       'final_energy':float(np.trace(h@reduced).real)})
    distance = float(np.sum(abs(np.linalg.eigvalsh(finals[0]-finals[1])))/2)
    assert distance > .1
    v = np.array([-1j,1,-1,0])
    exact12 = np.eye(4,dtype=complex)/4
    exact12[0,1],exact12[1,0] = -1j/4,1j/4
    exact21 = (np.outer(v,v.conj())+q)/4
    errors['exact_density_12'] = float(np.linalg.norm(finals[0]-exact12))
    errors['exact_density_21'] = float(np.linalg.norm(finals[1]-exact21))
    errors['exact_order_distance'] = abs(distance-1/(2*np.sqrt(2)))
    # Distinct local observables and ticks on different tensor factors do commute.
    local_qs = [np.kron(np.diag([0.,1.]),eye),np.kron(eye,np.diag([0.,1.]))]
    local_events=[]
    for i,(lq,u) in enumerate(zip(local_qs,us)):
        flip = np.kron(x,eye) if i==0 else np.kron(eye,x)
        lw=np.kron(np.eye(4)-lq,np.eye(4))+np.kron(lq,flip)
        local_events.append(lw@np.kron(u,np.eye(4)))
    errors['local_event_commutator']=float(np.linalg.norm(
        local_events[0]@local_events[1]-local_events[1]@local_events[0]))
    # End-only record leaves the event-order equality intact.
    end_a=ws[0]@np.kron(us[1]@us[0],np.eye(4))@initial
    end_b=ws[0]@np.kron(us[0]@us[1],np.eye(4))@initial
    errors['end_only_order_error']=float(np.linalg.norm(end_a-end_b))
    assert max(errors.values()) < 1e-12,errors
    result={'candidate':'CE-AM3','errors':errors,'orders':orders,
            'system_order_trace_distance':distance,
            'recorded_event_commutator_norm':float(np.linalg.norm(events[0]@events[1]-events[1]@events[0])),
            'tick_energy_change_operator_norms':[float(np.linalg.norm(u.conj().T@h@u-h,2)) for u in us],
            'initial_energy':float((state.conj()@h@state).real),
            'status':'base-point embedding and unitary record; single outcome not derived'}
    here=Path(__file__).resolve()
    result['source_sha256']={name:hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
                            for name in [here.name,'ce_common_isotropic_frame.py','ce_isometric_color_frame.py']}
    here.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
