"""Check the common relation/geometry/action derivation in main-paper sec. 3.

Finite explicit witnesses for identities, not a Standard Model or cosmology fit.
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import sympy as sy


def simple_state(q):
    theta, phi, eta=q
    a=np.array([np.cos(theta/2),np.exp(1j*phi)*np.sin(theta/2)])
    at=np.array([-np.sin(theta/2),np.exp(1j*phi)*np.cos(theta/2)])/2
    ap=np.array([0,1j*np.exp(1j*phi)*np.sin(theta/2)])
    b=np.array([np.cos(eta/2),np.sin(eta/2)])
    be=np.array([-np.sin(eta/2),np.cos(eta/2)])/2
    return np.kron(a,b)[:,None],np.stack([np.kron(at,b),np.kron(ap,b),np.kron(a,be)])[:,:,None]


def geometry(V, derivatives):
    Q=np.eye(len(V))-V@V.conj().T
    tensor=np.array([[da.conj().T@Q@db for db in derivatives] for da in derivatives])
    metric=np.trace(tensor,axis1=2,axis2=3).real
    connection=np.array([1j*V.conj().T@d for d in derivatives])
    curvature=1j*(tensor-tensor.swapaxes(0,1))
    return metric,connection,curvature


def curl_check(frame,q,h):
    V,d=frame(q)
    _,A,F=geometry(V,d)
    result=np.zeros_like(F)
    for a in range(3):
        qa=q.copy();qa[a]+=h
        qb=q.copy();qb[a]-=h
        Ap=geometry(*frame(qa))[1];Am=geometry(*frame(qb))[1]
        for b in range(3):
            result[a,b]+=(Ap[b]-Am[b])/(2*h)
            result[b,a]-=(Ap[b]-Am[b])/(2*h)
    for a in range(3):
        for b in range(3):result[a,b]-=1j*(A[a]@A[b]-A[b]@A[a])
    return float(np.max(abs(result-F)))


def run():
    q=np.array([.8,.37,1.1])
    V,d=simple_state(q)
    h,A,F=geometry(V,d)
    expected=np.diag([1,np.sin(q[0])**2,1])/4
    err=float(np.max(abs(h-expected)))
    assert err<1e-14 and abs(F[0,1,0,0]+np.sin(q[0])/2)<1e-14
    ranks=[int(np.linalg.matrix_rank(h[:n,:n])) for n in [1,2,3]]
    assert ranks==[1,2,3]

    # Curvature of the induced 3-metric, computed from Christoffel symbols.
    t,p,e=sy.symbols('t p e',real=True);coords=[t,p,e]
    metric=sy.diag(sy.Rational(1,4),sy.sin(t)**2/4,sy.Rational(1,4));inv=metric.inv()
    G=[[[sy.simplify(sum(inv[k,l]*(sy.diff(metric[l,j],coords[i])+
          sy.diff(metric[l,i],coords[j])-sy.diff(metric[i,j],coords[l]))/2
          for l in range(3))) for j in range(3)] for i in range(3)] for k in range(3)]
    Ric=sy.zeros(3)
    for i in range(3):
        for j in range(3):
            Ric[i,j]=sy.simplify(sum(sy.diff(G[k][i][j],coords[k])-sy.diff(G[k][i][k],coords[j])+
                sum(G[k][i][j]*G[l][k][l]-G[l][i][k]*G[k][j][l] for l in range(3)) for k in range(3)))
    scalar=sy.simplify(sum(inv[i,j]*Ric[i,j] for i in range(3) for j in range(3)))
    assert scalar==8

    rng=np.random.default_rng(27031)
    generators=[]
    for _ in range(3):
        z=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
        generators.append((z+z.conj().T)/10)
    initial=np.eye(5,dtype=complex)[:,:2]
    def frame(q1):
        U=[expm(1j*v*g) for v,g in zip(q1,generators)]
        V=U[0]@U[1]@U[2]@initial
        ds=[]
        for i in range(3):
            factors=[1j*generators[j]@U[j] if i==j else U[j] for j in range(3)]
            ds.append(factors[0]@factors[1]@factors[2]@initial)
        return V,np.array(ds)
    curl_errors=[curl_check(frame,q,step) for step in [2e-4,1e-4]]
    assert curl_errors[1]<curl_errors[0]/3 and curl_errors[1]<1e-8
    V,d=frame(q);h,A,F=geometry(V,d)
    sigmay=np.array([[0,-1j],[1j,0]])
    G=expm(1j*q[0]*sigmay/2)
    dG=1j*sigmay@G/2
    vg=V@G;dg=np.array([x@G for x in d]);dg[0]+=V@dG
    hg,Ag,Fg=geometry(vg,dg)
    gauge_metric_error=float(np.max(abs(hg-h)))
    gauge_curvature_error=float(np.max(abs(Fg-np.array([[G.conj().T@x@G for x in row] for row in F]))))
    assert max(gauge_metric_error,gauge_curvature_error)<1e-14

    P=V@V.conj().T;Q=np.eye(5)-P
    E=np.diag([1,0,1,0,0]);EP=P@E@P
    defect=EP-EP@EP
    defect_error=float(np.max(abs(defect-P@E@Q@E@P)))
    eig=np.linalg.eigvalsh(defect)
    assert defect_error<1e-14 and min(eig)>-1e-14
    B=np.diag([0,1,2,-1,3]);C=np.diag([1,2,-1,3,0])
    bp=P@B@P;cp=P@C@P
    comm=bp@cp-cp@bp
    comm_error=float(np.max(abs(comm-(-P@B@Q@C@P+P@C@Q@B@P))))
    assert comm_error<1e-14 and np.linalg.norm(comm)>1e-3

    z=rng.normal(size=(7,7))+1j*rng.normal(size=(7,7))
    K=z.conj().T@z+np.eye(7)*2
    KP,B,KQ=K[:3,:3],K[:3,3:],K[3:,3:]
    Keff=KP-B@np.linalg.solve(KQ,B.conj().T)
    logdet=lambda x:np.linalg.slogdet(x)[1]
    determinant_error=float(abs(logdet(K)-logdet(KQ)-logdet(Keff)))
    inverse_error=float(np.max(abs(np.linalg.inv(K)[:3,:3]-np.linalg.inv(Keff))))
    assert max(determinant_error,inverse_error)<1e-12
    return dict(relation_coordinates=q.tolist(),simple_metric_error=err,
                spatial_ranks=ranks,scalar_curvature_in_units_ell_minus2=str(scalar),
                nonabelian_curl_errors=curl_errors,gauge_metric_error=gauge_metric_error,
                gauge_curvature_error=gauge_curvature_error,
                compressed_effect_defect_error=defect_error,
                compressed_effect_defect_eigenvalues=eig.tolist(),
                projected_commutator_norm=float(np.linalg.norm(comm)),
                projected_commutator_identity_error=comm_error,
                block_logdet_error=determinant_error,block_inverse_error=inverse_error,
                scope='Explicit relation geometry and Gaussian identities; no force identification, initial-state selection or cosmological prediction.',
                full_joint_rmse=None)


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
