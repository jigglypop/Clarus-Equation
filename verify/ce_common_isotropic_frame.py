"""CE-GN3: common U(1), SU(2), SU(3) geometric transport and cyclic factor.

Coherent states are exact in the analytic construction. Finite Fock cutoffs are
only an error-controlled numerical check, not a replacement physical model.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np

try:
    from .ce_isometric_color_frame import KS, V0, Q0, SMALL, frame as color_frame, exact_rank
except ImportError:
    from ce_isometric_color_frame import KS, V0, Q0, SMALL, frame as color_frame, exact_rank


SIGMA=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1.,-1.])]
A=.5
POINTS=([0.,0.,0.],[.2,.3,.4],[1.,-.5,.7])
CUTOFFS=(4,8,16)
THETAS=(0.,.7,float(np.pi))


def block_diag(blocks):
    out=np.zeros((sum(b.shape[0] for b in blocks),sum(b.shape[1] for b in blocks)),complex)
    r=c=0
    for b in blocks:
        out[r:r+b.shape[0],c:c+b.shape[1]]=b;r+=b.shape[0];c+=b.shape[1]
    return out


def weak_setup():
    signs=np.array(list(itertools.product((-1,1),repeat=3)))
    effects=[(np.eye(2)+A*sum(s[a]*SIGMA[a] for a in range(3)))/8 for s in signs]
    blocks=[]
    for e in effects:
        values,vectors=np.linalg.eigh(e)
        assert values.min()>0
        blocks.append((vectors*np.sqrt(values))@vectors.conj().T)
    v0=np.vstack(blocks)
    assert np.max(np.abs(v0.conj().T@v0-np.eye(2)))<1e-12
    return signs,v0


def weak_frame(q):
    signs,v0=weak_setup()
    phases=np.repeat(np.exp(1j*(signs@q)),2)
    v=phases[:,None]*v0
    dv=[1j*np.repeat(signs[:,a],2)[:,None]*v for a in range(3)]
    return v,dv


def coherent(alpha,alpha_dot,cutoff):
    c=np.zeros(cutoff+1,complex);dc=np.zeros_like(c)
    c[0]=np.exp(-abs(alpha)**2/2)
    dc[0]=-np.real(np.conj(alpha)*alpha_dot)*c[0]
    for n in range(1,cutoff+1):
        c[n]=alpha*c[n-1]/np.sqrt(n)
        dc[n]=(alpha_dot*c[n-1]+alpha*dc[n-1])/np.sqrt(n)
    probability=float(np.vdot(c,c).real)
    normalized=c/np.sqrt(probability)
    derivative=(dc-c*np.vdot(c,dc).real/probability)/np.sqrt(probability)
    mu=abs(alpha)**2
    first=math.exp(-mu)*mu**(cutoff+1)/math.factorial(cutoff+1)
    assert mu<cutoff+2
    tail_bound=first/(1-mu/(cutoff+2))
    assert 1-probability<=tail_bound+1e-14
    return normalized,derivative,tail_bound


def line_frame(q,cutoff):
    alpha=(q[0]+1j*q[1])/np.sqrt(2);beta=q[2]/np.sqrt(2)
    a,da,ta=coherent(alpha,1/np.sqrt(2),cutoff)
    _,dia,_=coherent(alpha,1j/np.sqrt(2),cutoff)
    b,db,tb=coherent(beta,1/np.sqrt(2),cutoff)
    v=np.kron(a,b)[:,None]
    dv=[np.kron(da,b)[:,None],np.kron(dia,b)[:,None],np.kron(a,db)[:,None]]
    return v,dv,ta+tb


def qgt_from_frame(v,dv):
    assert np.max(np.abs(v.conj().T@v-np.eye(v.shape[1])))<1e-12
    # Gram subtraction avoids allocating an ambient identity/projector.
    normal=[d-v@(v.conj().T@d) for d in dv]
    return np.array([[a.conj().T@b for b in normal] for a in normal])


def analytic_gauge_tensor():
    line=.5*np.array([[1,1j,0],[-1j,1,0],[0,0,1]],complex)
    color=np.array([[V0.conj().T@a@Q0@b@V0 for b in KS] for a in KS])
    return np.array([[block_diag([np.array([[line[a,b]]]),
                     (np.eye(2) if a==b else np.zeros((2,2)))-A*A*SIGMA[a]@SIGMA[b],
                     color[a,b]]) for b in range(3)] for a in range(3)])


def cycle_matrix(theta):
    s=np.roll(np.eye(3,dtype=complex),1,axis=0)
    return np.eye(3)+.2*(np.exp(1j*theta/3)*s+np.exp(-1j*theta/3)*s.conj().T)


def finite_flavor_tensor(q,theta,cutoff):
    k=cycle_matrix(theta);values,vectors=np.linalg.eigh(k)
    assert values.min()>0
    diagonal=np.zeros((3,3))
    tails=[]
    for j,value in enumerate(values):
        total_tail=0.
        for a in range(3):
            v,d,t=coherent(np.sqrt(value)*q[a],np.sqrt(value),cutoff)
            diagonal[a,j]=np.vdot(d,d).real-abs(np.vdot(v,d))**2
            total_tail+=t
        tails.append(total_tail)
    result=np.zeros((3,3,3,3),complex)
    for a in range(3):result[a,a]=(vectors*diagonal[a])@vectors.conj().T
    return result,max(tails)


def common_mass(tg,tf):
    h=3*np.trace(tg,axis1=2,axis2=3).real+6*np.trace(tf,axis1=2,axis2=3).real
    inverse=np.linalg.inv(h)
    pg=np.einsum('ab,abij->ij',inverse,tg)
    pf=np.einsum('ab,abij->ij',inverse,tf)
    phi=np.kron(pg,np.eye(3))+np.kron(np.eye(6),pf)
    assert np.linalg.eigvalsh(h).min()>0 and np.linalg.eigvalsh(phi).min()>0
    assert abs(np.trace(phi)-3)<1e-11
    return h,pg,phi


def holonomy_audit(tg):
    connection=[block_diag([np.zeros((1,1)),-A*SIGMA[a],-SMALL[a]]) for a in range(3)]
    curvatures=[1j*(tg[a,b]-tg[b,a]) for a,b in ((0,1),(0,2),(1,2))]
    derivatives=[-1j*(a@f-f@a) for a in connection for f in curvatures]
    raw=curvatures+derivatives
    raw += [1j*(a@b-b@a) for a in curvatures for b in derivatives]
    # All coefficients here are exactly dyadic; scale 2 gives integer entries.
    initial_rank=exact_rank([2*m for m in raw])
    basis=[]
    def append_if_independent(m):
        rows=np.array([np.r_[x.real.ravel(),x.imag.ravel()] for x in basis+[m]])
        if np.linalg.matrix_rank(rows,tol=1e-10)>len(basis):basis.append(m)
    for m in raw:append_if_independent(m)
    for _ in range(12):
        old=list(basis)
        for a in old:
            for b in old:append_if_independent(1j*(a@b-b@a))
        if len(old)==len(basis):break
    # Numerical selection only picks a witness; exact rank certifies its span.
    rank=exact_rank([2*m for m in basis])
    assert rank==12
    repeated_rank=exact_rank([2*np.kron(m,np.eye(3)) for m in basis])
    assert repeated_rank==12
    assert all(abs(np.trace(m[1:3,1:3]))<1e-14 and abs(np.trace(m[3:,3:]))<1e-14 for m in basis)
    return dict(initial_generator_span_rank=initial_rank,exact_combined_rank=rank,rank_on_18_component_frame=repeated_rank,method='joint curvature, covariant derivatives and full commutator closure; rational witness elimination',
                group_choice_not_derived=True)


def analytic_audit():
    tg=analytic_gauge_tensor()
    hg=np.trace(tg,axis1=2,axis2=3).real
    g=sum(tg[a,a] for a in range(3))
    assert np.array_equal(hg,3*np.eye(3))
    assert np.array_equal(g,np.diag([1.5,2.25,2.25,1.,1.,1.]))
    rows=[]
    for theta in THETAS:
        k=cycle_matrix(theta)
        tf=np.array([[k if a==b else np.zeros((3,3)) for b in range(3)] for a in range(3)])
        h,pg,phi=common_mass(tg,tf)
        expected=(np.kron(g,np.eye(3))+np.kron(np.eye(6),3*k))/27
        assert np.max(np.abs(h-27*np.eye(3)))<1e-12
        assert np.max(np.abs(phi-expected))<1e-12
        angles=(theta+2*np.pi*np.arange(3))/3
        predicted=np.sort(np.concatenate([np.tile((value+3)/27+2*(.6/27)*np.cos(angles),count)
                                           for value,count in ((1.5,1),(2.25,2),(1.,3))]))
        assert np.max(np.abs(np.linalg.eigvalsh(phi)-predicted))<1e-12
        connection=[block_diag([np.zeros((1,1)),-A*SIGMA[a],-SMALL[a]]) for a in range(3)]
        assert max(np.linalg.norm(pg@a-a@pg) for a in connection)<1e-12
        rows.append(dict(theta=theta,common_metric_coefficient=27,
                         diagonal_mass_coefficients=[4.5/27,5.25/27,4/27],
                         shared_cyclic_epsilon=.6/27,mass_eigenvalues=np.linalg.eigvalsh(phi).tolist()))
    return dict(gauge_metric_coefficient=3,rows=rows,holonomy=holonomy_audit(tg))


def cutoff_audit():
    exact_g=analytic_gauge_tensor();rows=[]
    for theta in THETAS:
        k=cycle_matrix(theta)
        exact_f=np.array([[k if a==b else np.zeros((3,3)) for b in range(3)] for a in range(3)])
        exact_h,_,exact_phi=common_mass(exact_g,exact_f)
        for point in POINTS:
            q=np.array(point);series=[]
            for cutoff in CUTOFFS:
                vl,dl,tail_l=line_frame(q,cutoff)
                vw,dw=weak_frame(q);vc=color_frame(q);dc=[1j*kk@vc for kk in KS]
                # Direct-sum frame, including the truncated Abelian Hilbert space.
                v=block_diag([vl,vw,vc]);dv=[block_diag([dl[a],dw[a],dc[a]]) for a in range(3)]
                tg=qgt_from_frame(v,dv)
                tf,tail_f=finite_flavor_tensor(q,theta,cutoff)
                h,pg,phi=common_mass(tg,tf)
                color=pg[3:,3:];color_defect=np.linalg.norm(color-np.trace(color)*np.eye(3)/3)
                row=dict(theta=theta,q=point,cutoff=cutoff,ambient_gauge_dimension=v.shape[0],
                         discarded_norm_squared_upper_bound=max(tail_l,tail_f),
                         metric_max_error=float(np.max(np.abs(h-exact_h))),
                         mass_max_error=float(np.max(np.abs(phi-exact_phi))),
                         color_traceless_norm=float(color_defect))
                rows.append(row);series.append(row)
            assert series[-1]['metric_max_error']<1e-8
            assert series[-1]['mass_max_error']<1e-10
            if series[0]['metric_max_error']>1e-7:
                assert series[-1]['metric_max_error']<series[0]['metric_max_error']/10000
    return rows


def kinetic_identity_audit():
    # A modest finite cutoff permits a direct full tensor-product check.
    q=np.array([.2,.3,.4]);theta=.7;cutoff=4
    vl,dl,_=line_frame(q,cutoff);vw,dw=weak_frame(q)
    vc=color_frame(q);dc=[1j*kk@vc for kk in KS]
    vg=block_diag([vl,vw,vc]);dg=[block_diag([dl[a],dw[a],dc[a]]) for a in range(3)]
    values,vectors=np.linalg.eigh(cycle_matrix(theta))
    columns=[];derivatives=[[],[],[]]
    for value in values:
        pieces=[coherent(np.sqrt(value)*q[a],np.sqrt(value),cutoff) for a in range(3)]
        columns.append(np.kron(np.kron(pieces[0][0],pieces[1][0]),pieces[2][0])[:,None])
        for a in range(3):
            factors=[pieces[b][1 if b==a else 0] for b in range(3)]
            derivatives[a].append(np.kron(np.kron(factors[0],factors[1]),factors[2])[:,None])
    vf=block_diag(columns)@vectors.conj().T
    df=[block_diag(ds)@vectors.conj().T for ds in derivatives]
    v=np.kron(vg,vf)
    dv=[np.kron(dg[a],vf)+np.kron(vg,df[a]) for a in range(3)]
    total=qgt_from_frame(v,dv)
    h=np.trace(total,axis1=2,axis2=3).real;inverse=np.linalg.inv(h)
    phi=np.einsum('ab,abij->ij',inverse,total)
    factor_h,_,factor_phi=common_mass(qgt_from_frame(vg,dg),qgt_from_frame(vf,df))
    factor_error=float(np.max(np.abs(phi-factor_phi)))
    assert np.max(np.abs(h-factor_h))<1e-11 and factor_error<1e-12
    amplitude=np.arange(1,19)+1j*np.arange(18,0,-1);amplitude/=np.linalg.norm(amplitude)
    grad=[np.cos(np.arange(18)+a)+1j*np.sin(np.arange(18)-a) for a in range(3)]
    full_gradient=[v@grad[a]+dv[a]@amplitude for a in range(3)]
    covariant=[grad[a]+v.conj().T@dv[a]@amplitude for a in range(3)]
    lhs=sum(inverse[a,b]*np.vdot(full_gradient[a],full_gradient[b]) for a in range(3) for b in range(3))
    rhs=sum(inverse[a,b]*np.vdot(covariant[a],covariant[b]) for a in range(3) for b in range(3))+np.vdot(amplitude,phi@amplitude)
    residual=float(abs(lhs-rhs))
    assert residual<1e-11
    return dict(cutoff=cutoff,frame_shape=list(v.shape),factorization_max_error=factor_error,
                kinetic_identity_error=residual,scope='exact decomposition within this normalized finite-cutoff frame')


def main():
    result=dict(candidate='CE-GN3',status='conditional common isotropic geometry; no dynamical force unification claimed',
                analytic=analytic_audit(),cutoff=cutoff_audit(),
                kinetic_identity=kinetic_identity_audit(),
                not_derived=['choice of three blocks and chiral matter','independent Yang-Mills dynamics',
                             'physical generation of frame and gap','quantum effective action for infinite hidden sector',
                             'dynamic metric and Einstein limit','actual record selection','joint observational RMSE'],
                provenance=dict(python=platform.python_version(),numpy=np.__version__,
                                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                color_script_sha256=hashlib.sha256(Path(__file__).with_name('ce_isometric_color_frame.py').read_bytes()).hexdigest()))
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    summary=dict(candidate=result['candidate'],analytic=result['analytic'],
                 cutoff_cases=len(result['cutoff']),
                 final_cutoff_max_metric_error=max(r['metric_max_error'] for r in result['cutoff'] if r['cutoff']==16),
                 final_cutoff_max_mass_error=max(r['mass_max_error'] for r in result['cutoff'] if r['cutoff']==16))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
