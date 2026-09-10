"""CE-GR3: local direct rotation of projections and finite weighted kinetic sums."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from ce_common_isotropic_frame import block_diag, coherent, cycle_matrix, line_frame, weak_frame
from ce_isometric_color_frame import frame as color_frame
from ce_projector_one_loop import coefficient


def common_frame(q, theta, cutoff):
    line,_,_=line_frame(q,cutoff)
    weak,_=weak_frame(q)
    vg=block_diag([line,weak,color_frame(q)])
    values,vectors=np.linalg.eigh(cycle_matrix(theta))
    columns=[]
    for kappa in values:
        pieces=[coherent(np.sqrt(kappa)*q[a],np.sqrt(kappa),cutoff)[0] for a in range(3)]
        columns.append(np.kron(np.kron(pieces[0],pieces[1]),pieces[2])[:,None])
    vf=block_diag(columns)@vectors.conj().T
    return np.kron(vg,vf)


def rotation_check(q,theta,cutoff):
    v0=common_frame([0.,0.,0.],theta,cutoff)
    v=common_frame(q,theta,cutoff)
    basis,singular,_=np.linalg.svd(np.column_stack([v0,v]),full_matrices=False)
    basis=basis[:,singular>1e-11]
    c0=basis.conj().T@v0;c=basis.conj().T@v
    p0=c0@c0.conj().T;p=c@c.conj().T
    eye=np.eye(len(p));delta=p-p0
    t=eye-delta@delta
    values,vectors=np.linalg.eigh(t)
    assert values.min()>0
    r=(p@p0+(eye-p)@(eye-p0))@((vectors/np.sqrt(values))@vectors.conj().T)
    rotated=r@c0
    errors={'unitarity':float(np.linalg.norm(r.conj().T@r-eye)),
            'projection_transport':float(np.linalg.norm(rotated@rotated.conj().T-p)),
            'intertwining':float(np.linalg.norm(r@p0-p@r))}
    assert max(errors.values())<1e-10,errors
    rank=int(np.count_nonzero(np.linalg.svd(r-eye,compute_uv=False)>1e-10))
    assert rank<=36
    return {'q':q,'theta':theta,'cutoff':cutoff,'ambient_rows':v.shape[0],
            'rotation_rank':rank,'chart_minimum_T':float(values.min()),'errors':errors}


def line_rotation(q,cutoff):
    v=coherent(q,1.,cutoff)[0]
    e=np.eye(cutoff+1)[:,0]
    c=float(v[0].real);w=v-c*e
    return np.eye(cutoff+1)+(c-1)*np.outer(e,e)-np.outer(w,w.conj())/(1+c)+np.outer(w,e)-np.outer(e,w.conj())


def oscillator_check(q,cutoff,epsilon,cost,masses):
    r=line_rotation(q,cutoff)
    dr=(line_rotation(q+epsilon,cutoff)-line_rotation(q-epsilon,cutoff))/(2*epsilon)
    raw=r.conj().T@dr
    generator=(raw-raw.conj().T)/2
    weights=abs(generator)**2
    kinetic=float(np.sum(np.triu(cost*weights,1)))
    bound=float(np.dot(masses,np.sum(weights,axis=0))/(32*np.pi**2))
    lh=float(np.sum(cost[0,1:]*weights[0,1:]))
    assert 0<kinetic<=bound+1e-13
    unitary=float(np.linalg.norm(r.conj().T@r-np.eye(cutoff+1)))
    assert unitary<1e-10
    return {'q':q,'cutoff':cutoff,'epsilon':epsilon,'full_kinetic':kinetic,
            'light_heavy':lh,'heavy_heavy':kinetic-lh,'weighted_upper_bound':bound,
            'raw_generator_antihermiticity_error':float(np.linalg.norm(raw+raw.conj().T)),
            'unitarity_error':unitary}


def main():
    common=[]
    for q in ([.1,-.05,.08],[.3,.2,-.1]):
        for theta in (0.,.7,math.pi):
            for cutoff in (2,4):common.append(rotation_check(q,theta,cutoff))
    lines=[]
    for cutoff in (16,32,64):
        masses=np.array([1.]+[5.+2*n for n in range(1,cutoff+1)])
        cost=np.zeros((cutoff+1,cutoff+1))
        for i in range(cutoff+1):
            for j in range(i+1,cutoff+1):cost[i,j]=cost[j,i]=coefficient(masses[i],masses[j])
        for q in (0.,.3,.7):
            for eps in (1e-4,5e-5):lines.append(oscillator_check(q,cutoff,eps,cost,masses))
    convergence=[]
    for q in (0.,.3,.7):
        selected=[r for r in lines if r['q']==q]
        reference=selected[-1]['full_kinetic']
        error=max(abs(r['full_kinetic']-reference)/reference for r in selected)
        assert error<1e-6,(q,error)
        convergence.append({'q':q,'maximum_relative_change':error})
    zero=next(r for r in lines if r['q']==0. and r['cutoff']==64 and r['epsilon']==5e-5)
    assert abs(zero['full_kinetic']/coefficient(1.,7.)-1)<1e-6
    here=Path(__file__).resolve()
    sources=[here.name,'ce_common_isotropic_frame.py','ce_isometric_color_frame.py','ce_projector_one_loop.py']
    out={'candidate':'CE-GR3','common_frames':common,'oscillator_cases':lines,'convergence':convergence,
         'source_sha256':{f:hashlib.sha256(here.with_name(f).read_bytes()).hexdigest() for f in sources},
         'limits':['General convergence is conditional on finite energy-weighted tangent norm',
                   'Oscillator coefficient table is a rank-one witness, not full GN3 coefficients',
                   'Reference-chart rule is supplied; absolute gravity remains unresolved']}
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'common_max_error':max(max(r['errors'].values()) for r in common),
                     'minimum_chart_T':min(r['chart_minimum_T'] for r in common),
                     'convergence':convergence,
                     'oscillator_results':[r for r in lines if r['cutoff']==64 and r['epsilon']==5e-5]},indent=2))


if __name__=='__main__':main()
