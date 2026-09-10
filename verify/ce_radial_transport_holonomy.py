"""CE-GR4: global prescribed-path transport and complement holonomy obstruction."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from ce_common_isotropic_frame import block_diag, cycle_matrix, weak_frame, line_frame
from ce_isometric_color_frame import KS,V0,frame as color_frame


def path_frame(q,theta):
    # Exact normalized L=1 family; no assertion about infinite-state tail accuracy.
    line,_,_=line_frame(q,1)
    weak,_=weak_frame(q)
    vg=block_diag([line,weak,color_frame(q)])
    values,vectors=np.linalg.eigh(cycle_matrix(theta))
    columns=[]
    for kappa in values:
        pieces=[]
        for qa in q:
            alpha=np.sqrt(kappa)*qa
            pieces.append(np.array([1.,alpha])/np.sqrt(1+alpha*alpha))
        columns.append(np.kron(np.kron(pieces[0],pieces[1]),pieces[2])[:,None])
    return np.kron(vg,block_diag(columns)@vectors.conj().T)


def origin_normals(theta):
    vl=np.zeros((4,1),complex);vl[0]=1
    dl=[np.zeros_like(vl) for _ in range(3)]
    dl[0][2]=1/np.sqrt(2);dl[1][2]=1j/np.sqrt(2);dl[2][1]=1/np.sqrt(2)
    vw,dw=weak_frame([0.,0.,0.]);dc=[1j*k@V0 for k in KS]
    vg=block_diag([vl,vw,V0]);dg=[block_diag([dl[a],dw[a],dc[a]]) for a in range(3)]
    values,vectors=np.linalg.eigh(cycle_matrix(theta))
    vf=np.zeros((24,3),complex);df=[np.zeros_like(vf) for _ in range(3)]
    for j,k in enumerate(values):
        vf[8*j,j]=1
        for a,index in enumerate([4,2,1]):df[a][8*j+index,j]=np.sqrt(k)
    vf=vf@vectors.conj().T;df=[d@vectors.conj().T for d in df]
    v=np.kron(vg,vf)
    derivatives=[np.kron(dg[a],vf)+np.kron(vg,df[a]) for a in range(3)]
    normals=[d-v@(v.conj().T@d) for d in derivatives]
    ng=np.array([0,1,1,2]+[0]*20)
    nf=np.tile([sum(t) for t in itertools.product([0,1],repeat=3)],3)
    number=(ng[:,None]+nf[None,:]).reshape(-1)
    return v,normals,number


def curvature_check(theta):
    v,bs,number=origin_normals(theta)
    rows=[]
    for a,b in [(0,1),(1,2),(2,0)]:
        aa0,bb0=bs[a][number==0],bs[b][number==0]
        aa1,bb1=bs[a][number==1],bs[b][number==1]
        cross=aa0@bb1.conj().T-bb0@aa1.conj().T
        norm2=float(np.vdot(cross,cross).real)
        commutator_squared=2*2**2*norm2
        assert abs(norm2-15)<1e-10,(theta,a,b,norm2)
        assert abs(commutator_squared-120)<1e-10
        rows.append({'directions':[a+1,b+1],'cross_curvature_HS_squared':norm2,
                     'mass_curvature_commutator_HS_squared':commutator_squared})
    return {'theta':theta,'pairs':rows}


def transport_step(old,new,states):
    basis,singular,_=np.linalg.svd(np.column_stack([old,new]),full_matrices=False)
    basis=basis[:,singular>1e-11]
    o=basis.conj().T@old;n=basis.conj().T@new
    p0=o@o.conj().T;p=n@n.conj().T;eye=np.eye(len(p))
    d=p-p0;t=eye-d@d
    values,vectors=np.linalg.eigh(t)
    assert values.min()>.1
    r=(p@p0+(eye-p)@(eye-p0))@((vectors/np.sqrt(values))@vectors.conj().T)
    return states+basis@((r-eye)@(basis.conj().T@states))


def path_checks():
    theta=.7;endpoint=np.array([math.pi/2,0.,0.])
    start=path_frame([0.,0.,0.],theta)
    end=path_frame(endpoint,theta)
    minimum=float(np.linalg.svd(start.conj().T@end,compute_uv=False).min())
    assert minimum<1e-12
    _,normals,_=origin_normals(theta)
    probe=normals[0][:,0]
    probe=probe/np.linalg.norm(probe)
    initial=np.column_stack([start,probe]);gram=initial.conj().T@initial
    results=[];finals=[]
    for steps in (16,32,64):
        old=start;states=initial.copy()
        for j in range(1,steps+1):
            new=path_frame(endpoint*j/steps,theta)
            states=transport_step(old,new,states);old=new
        error=float(np.linalg.norm(states.conj().T@states-gram))
        light=states[:,:18];dark=states[:,18]
        residual=float(np.linalg.norm(light-end@(end.conj().T@light)))
        leakage=float(np.linalg.norm(end.conj().T@dark))
        assert max(error,residual,leakage)<1e-10
        results.append({'steps':steps,'inner_product_error':error,
                        'light_projection_error':residual,'complement_leakage':leakage})
        finals.append(states)
    changes=[float(np.linalg.norm(finals[j+1]-finals[j])) for j in range(2)]
    assert changes[1]<changes[0]/2,changes
    return {'old_chart_minimum_overlap_singular_value':minimum,
            'results':results,'successive_changes':changes,'change_ratio':changes[1]/changes[0]}


def main():
    out={'candidate':'CE-GR4','curvature':[curvature_check(t) for t in (0.,.7,math.pi)],
         'path':path_checks(),
         'limits':['Prescribed radial preparation is an additional assumption',
                   'Complement curvature does not commute with the rising spectrum',
                   'Finite L=1 path is an algebra check, not an accurate infinite frame cutoff']}
    here=Path(__file__).resolve()
    sources=[here.name,'ce_common_isotropic_frame.py','ce_isometric_color_frame.py',
             'ce_finite_rank_transport.py','ce_projector_one_loop.py']
    out['source_sha256']={f:hashlib.sha256(here.with_name(f).read_bytes()).hexdigest() for f in sources}
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
