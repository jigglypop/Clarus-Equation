"""CE-GN1: unchanged common frame away from its base point (NumPy only)."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


def block_diag(blocks):
    out = np.zeros((sum(b.shape[0] for b in blocks),sum(b.shape[1] for b in blocks)),complex)
    r=c=0
    for b in blocks:
        out[r:r+b.shape[0],c:c+b.shape[1]]=b
        r+=b.shape[0];c+=b.shape[1]
    return out


def inputs(theta):
    shift=np.roll(np.eye(3,dtype=complex),1,axis=0)
    z=np.diag(np.exp(2j*np.pi*np.arange(3)/3))
    one=[np.array([[1],[0]])/2,np.array([[1j],[0]])/2,np.array([[0],[1]])/2]
    two=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    three=[np.eye(3),z,shift@z@z]
    bg=[block_diag(b) for b in zip(one,two,three)]
    cf=np.eye(3)+.3*np.exp(1j*theta/3)*shift
    bf=[]
    for a in range(3):
        b=np.zeros((9,3),complex);b[3*a:3*a+3]=cf;bf.append(b)
    return bg,bf


def inverse_root(a):
    w,v=np.linalg.eigh(a)
    assert np.min(w)>0
    return (v/np.sqrt(w))@v.conj().T


def frame(bs,q):
    y=sum(x*b for x,b in zip(q,bs))
    return np.vstack([np.eye(y.shape[1]),y])@inverse_root(np.eye(y.shape[1])+y.conj().T@y)


def tensor(bs,q):
    y=sum(x*b for x,b in zip(q,bs))
    s=inverse_root(np.eye(y.shape[1])+y.conj().T@y)
    r=np.linalg.inv(np.eye(y.shape[0])+y@y.conj().T)
    return np.array([[s@a.conj().T@r@b@s for b in bs] for a in bs])


def lie_basis(generators,tol=1e-9):
    basis=[]
    def append(a):
        a=(a+a.conj().T)/2
        for _ in range(2):
            for b in basis:a-=np.vdot(b,a).real*b
        n=np.linalg.norm(a)
        if n>tol:basis.append(a/n)
    for a in generators:append(a.copy())
    for _ in range(36):
        old=len(basis)
        for a in list(basis):
            for b in list(basis):append(1j*(a@b-b@a))
        if len(basis)==old:return basis
    raise AssertionError('Lie closure did not stabilize')


def geometry(q,theta):
    bg,bf=inputs(theta)
    tg,tf=tensor(bg,q),tensor(bf,q)
    h=3*np.trace(tg,axis1=2,axis2=3).real+6*np.trace(tf,axis1=2,axis2=3).real
    inverse=np.linalg.inv(h)
    pg=np.einsum('ab,abij->ij',inverse,tg)
    pf=np.einsum('ab,abij->ij',inverse,tf)
    phi=np.kron(pg,np.eye(3))+np.kron(np.eye(6),pf)
    assert np.linalg.eigvalsh(h).min()>0
    assert np.linalg.eigvalsh(phi).min()>-1e-12
    assert abs(np.trace(phi)-3)<1e-12
    return h,pg,phi,tg


def point_audit(q,theta):
    h,pg,phi,tg=geometry(q,theta)
    color=(pg[3:,3:]+pg[3:,3:].conj().T)/2
    traceless=color-np.trace(color)*np.eye(3)/3
    curvature=[1j*(tg[a,b]-tg[b,a]) for a,b in ((0,1),(0,2),(1,2))]
    basis=lie_basis(curvature)
    # For this fixed block embedding, central directions are independent block traces.
    traces=np.array([[np.trace(g[:1,:1]).real,np.trace(g[1:3,1:3]).real,np.trace(g[3:,3:]).real] for g in basis])
    center=int(np.linalg.matrix_rank(traces,tol=1e-9))
    comm=max(np.linalg.norm(pg@g-g@pg) for g in basis)
    standard=[]
    for i,j in ((0,1),(0,2),(1,2)):
        e=np.zeros((3,3),complex);e[i,j]=1
        standard.extend([(e+e.conj().T)/2,(e-e.conj().T)/(2j)])
    standard.extend([np.diag([1.,-1.,0.])/2,np.diag([1.,1.,-2.])/(2*np.sqrt(3))])
    standard_comm=max(np.linalg.norm(color@g-g@color) for g in standard)
    eig=np.linalg.eigvalsh(color)
    weak=pg[1:3,1:3]
    assert np.linalg.norm(weak-np.trace(weak)*np.eye(2)/2)<1e-12
    return dict(q=list(q),theta=theta,color_eigenvalues=eig.tolist(),
                color_spread=float(eig[-1]-eig[0]),color_traceless_norm=float(np.linalg.norm(traceless)),
                mass_curvature_algebra_commutator_max=float(comm),
                standard_color_generator_commutator_max=float(standard_comm),
                curvature_algebra_dimension=len(basis),block_trace_rank=center,
                metric_min_eigenvalue=float(np.linalg.eigvalsh(h).min()),
                total_mass_trace=float(np.trace(phi).real))


def finite_difference_audit():
    theta=.7;q=np.array([.2,.2,0.]);bg,bf=inputs(theta)
    def total_frame(q):return np.kron(frame(bg,q),frame(bf,q))
    v=total_frame(q);normal=np.eye(156)-v@v.conj().T
    assert np.max(np.abs(v.conj().T@v-np.eye(18)))<1e-12
    expected_h,_,expected_phi,_=geometry(q,theta)
    rows=[]
    for step in (1e-3,5e-4,2.5e-4):
        deriv=[(total_frame(q+step*np.eye(3)[a])-total_frame(q-step*np.eye(3)[a]))/(2*step) for a in range(3)]
        direct=np.array([[a.conj().T@normal@b for b in deriv] for a in deriv])
        h=np.trace(direct,axis1=2,axis2=3).real
        phi=np.einsum('ab,abij->ij',np.linalg.inv(h),direct)
        rows.append(dict(step=step,metric_max_error=float(np.max(np.abs(h-expected_h))),
                         mass_max_error=float(np.max(np.abs(phi-expected_phi)))))
    assert rows[-1]['mass_max_error']<1e-7
    assert rows[-1]['metric_max_error']<rows[0]['metric_max_error']/10
    return rows


def analytic_direction_audit():
    # Along q=(u,u,0): h13=h23=0 and the color matrix is diagonal.
    # Its last two entries differ by h^33 * r1 * (r1-r0), strictly positive.
    rows=[]
    for u in (.1,.05,.025,.0125):
        h,pg,_,_=geometry([u,u,0.],0.)
        inverse=np.linalg.inv(h)
        r0=1/(1+4*u*u);r1=1/(1+u*u)
        exact=inverse[2,2]*3*u*u/((1+u*u)**2*(1+4*u*u))
        actual=float((pg[4,4]-pg[5,5]).real)
        assert max(abs(h[0,2]),abs(h[1,2]))<1e-12
        assert np.max(np.abs(pg[3:,3:]-np.diag(np.diag(pg[3:,3:]))))<1e-12
        assert actual>0 and abs(actual-exact)<1e-13
        rows.append(dict(u=u,color_diagonal_difference=actual,analytic_difference=exact,
                         difference_over_u_squared=actual/(u*u)))
    leading=Fraction(3)/(Fraction(63,4)+18*(1+Fraction(3,10)**2))
    assert leading==Fraction(100,1179)
    assert abs(rows[-1]['difference_over_u_squared']-float(leading))<1e-4
    return dict(exact_leading_coefficient=str(leading),rows=rows)


def covariance_audit():
    q=np.array([.2,.3,.4]);theta=.7
    h,pg,phi,_=geometry(q,theta)
    bg,bf=inputs(theta)
    change=np.array([[1.,.2,0.],[0.,1.,.1],[.1,0.,1.]])
    newq=np.linalg.solve(change,q)
    transformed=[[sum(change[b,a]*bs[b] for b in range(3)) for a in range(3)] for bs in (bg,bf)]
    tg,tf=[tensor(bs,newq) for bs in transformed]
    newh=3*np.trace(tg,axis1=2,axis2=3).real+6*np.trace(tf,axis1=2,axis2=3).real
    assert np.max(np.abs(newh-change.T@h@change))<1e-12
    inverse=np.linalg.inv(newh)
    newphi=np.kron(np.einsum('ab,abij->ij',inverse,tg),np.eye(3))+np.kron(np.eye(6),np.einsum('ab,abij->ij',inverse,tf))
    coordinate_error=float(np.max(np.abs(newphi-phi)))
    assert coordinate_error<1e-12
    fourier=np.exp(2j*np.pi*np.outer(np.arange(3),np.arange(3))/3)/np.sqrt(3)
    color=pg[3:,3:];rotated=fourier.conj().T@color@fourier
    spectrum_error=float(np.max(np.abs(np.linalg.eigvalsh(color)-np.linalg.eigvalsh(rotated))))
    assert spectrum_error<1e-12
    return dict(coordinate_mass_max_error=coordinate_error,color_basis_spectrum_max_error=spectrum_error)


def main():
    rows=[point_audit(q,theta) for theta in (0.,.7,float(np.pi))
          for q in ([0.,0.,0.],[.2,0.,0.],[.2,.2,0.],[.2,.3,.4])]
    for r in rows:
        if r['q']==[0.,0.,0.]:
            assert r['curvature_algebra_dimension']==12 and r['color_spread']<1e-12
        if r['q']==[.2,.2,0.]:assert r['color_spread']>1e-3
    result=dict(candidate='CE-GN1',status='base-point color degeneracy does not extend to a neighborhood',
                rows=rows,analytic_direction=analytic_direction_audit(),
                finite_difference=finite_difference_audit(),
                covariance=covariance_audit(),
                limitations=['fixed background mass condition, not gauge covariance violation',
                             'no gluon mass or new force derived','no observational refit or joint RMSE'],
                provenance=dict(python=platform.python_version(),numpy=np.__version__,
                                source_candidate_sha256=hashlib.sha256(Path(__file__).with_name('ce_joint_relation_representation.py').read_bytes()).hexdigest(),
                                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
