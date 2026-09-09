"""Derive the CE cosine mass at a relation-frame base point, with curvature.

The hopping pattern and frame are explicit hypotheses, not selected by data.
The calculation does not predict fermion charges, early states, or joint RMSE.
"""
import json
from pathlib import Path

import numpy as np
import sympy as sy


def hermitian_vector(A):
    return np.r_[A.real.ravel(), A.imag.ravel()]


def lie_dimension(generators, n):
    basis=[]
    def add(A):
        A=(A+A.conj().T)/2
        A=A-np.trace(A)*np.eye(n)/n
        v=hermitian_vector(A)
        # Reorthogonalize for stable rank near degeneracy.
        for _ in range(2):
            for b in basis:v-=np.dot(v,hermitian_vector(b))*hermitian_vector(b)
        norm=np.linalg.norm(v)
        if norm>1e-9:
            basis.append((v[:n*n]+1j*v[n*n:]).reshape(n,n)/norm)
    for A in generators:add(A)
    for _ in range(n*n):
        old=len(basis)
        for A in list(basis):
            for B in list(basis):add(1j*(A@B-B@A))
        if len(basis)==old:break
    return len(basis)


def make_frame(B, q):
    Y=sum(qq*b for qq,b in zip(q,B));m=B[0].shape[1]
    w,u=np.linalg.eigh(np.eye(m)+Y.conj().T@Y)
    inverse_root=(u/np.sqrt(w))@u.conj().T
    return np.vstack([np.eye(m),Y])@inverse_root


def local_geometry(B):
    tensor=np.array([[a.conj().T@b for b in B] for a in B])
    metric=np.trace(tensor,axis1=2,axis2=3).real
    phi=np.einsum('ab,abij->ij',np.linalg.inv(metric),tensor)
    F=1j*(tensor-tensor.swapaxes(0,1))
    return metric,phi,F


def relation_metric(B,q):
    Y=sum(qq*b for qq,b in zip(q,B))
    left=np.linalg.inv(np.eye(Y.shape[1])+Y.conj().T@Y)
    right=np.linalg.inv(np.eye(Y.shape[0])+Y@Y.conj().T)
    return np.array([[np.trace(left@a.conj().T@right@b).real for b in B] for a in B])


def scalar_curvature(B,step=None):
    d=len(B);h=relation_metric(B,np.zeros(d));inverse=np.linalg.inv(h)
    dd=np.empty((d,d,d,d))
    for c in range(d):
        for e in range(d):
            if step is None:
                for a in range(d):
                    for b in range(d):
                        dd[a,b,c,e]=-np.trace(
                            (B[c].conj().T@B[e]+B[e].conj().T@B[c])@B[a].conj().T@B[b]+
                            B[a].conj().T@(B[c]@B[e].conj().T+B[e]@B[c].conj().T)@B[b]).real
            elif c==e:
                shift=np.zeros(d);shift[c]=step
                dd[:,:,c,e]=(relation_metric(B,shift)-2*h+relation_metric(B,-shift))/step**2
            else:
                u=np.zeros(d);u[c]=step;v=np.zeros(d);v[e]=step
                dd[:,:,c,e]=(relation_metric(B,u+v)-relation_metric(B,u-v)-
                             relation_metric(B,-u+v)+relation_metric(B,-u-v))/(4*step**2)
    # First derivatives of h vanish at q=0; Christoffel-square terms vanish.
    return float(np.einsum('ab,cd,adbc',inverse,inverse,dd)-np.einsum('ab,cd,abcd',inverse,inverse,dd))


def symbolic_curvature_check():
    v,u=sy.symbols('v u',nonzero=True)
    omega=(-1+sy.sqrt(3)*sy.I)/2
    S=sy.Matrix([[0,0,1],[1,0,0],[0,1,0]])
    Z=sy.diag(1,omega,omega**2);Ua=[sy.eye(3),Z,S*Z**2]
    # u is a formal unit phase; its conjugate is 1/u. v is real here.
    C=sy.eye(3)+v*u*S;Cd=sy.eye(3)+v/u*S.T
    B=[a*C for a in Ua];Bd=[Cd*a.conjugate().T for a in Ua]
    def dd(a,b,c,d):
        return -sy.trace((Bd[c]*B[d]+Bd[d]*B[c])*Bd[a]*B[b]+
                        Bd[a]*(B[c]*Bd[d]+B[d]*Bd[c])*B[b])
    numerator=sy.factor(sy.expand(sum(dd(a,c,a,c)-dd(a,a,c,c)
                                       for a in range(3) for c in range(3))))
    R=sy.factor(numerator/(9*(1+v*v)**2))
    assert sy.simplify(R-4-2*v*v/(1+v*v)**2)==0
    return dict(scalar_curvature=str(R),phase_independent=not R.has(u),
                identity_residual='0',method='Exact clock-shift matrices, formal unit phase, metric second derivatives.')


def run():
    S=np.roll(np.eye(3,dtype=complex),1,axis=0)
    omega=np.exp(2j*np.pi/3);Z=np.diag(omega**np.arange(3))
    assert np.max(abs(Z@S-omega*S@Z))<1e-14
    Ua=[np.eye(3),Z,S@Z@Z]
    rows=[]
    for v in [0.,.3,1.]:
        phase_rows=[]
        for theta in [0.,.7,np.pi]:
            C=np.eye(3)+v*np.exp(1j*theta/3)*S
            B=[u@C for u in Ua]
            h,phi,F=local_geometry(B)
            r=v/(1+v*v)
            expected=np.eye(3)+r*(np.exp(1j*theta/3)*S+np.exp(-1j*theta/3)*S.conj().T)
            matrix_error=float(np.max(abs(phi-expected)))
            expected_eigs=np.sort(1+2*r*np.cos((theta+2*np.pi*np.arange(3))/3))
            eigen_error=float(np.max(abs(np.linalg.eigvalsh(phi)-expected_eigs)))
            assert matrix_error<1e-14 and eigen_error<1e-14
            assert np.max(abs(h-3*(1+v*v)*np.eye(3)))<1e-13
            dimension=lie_dimension([F[0,1],F[0,2],F[1,2]],3)
            R=scalar_curvature(B)
            assert abs(R-(4+2*r*r))<1e-12
            curvature_errors=[abs(scalar_curvature(B,hstep)-R) for hstep in [1e-3,5e-4]]
            assert curvature_errors[1]<curvature_errors[0]/3
            # Check actual normalized frames and a differential kinetic action
            # away from the origin; this includes a nonzero connection.
            q=np.array([.12,-.07,.09]);step=1e-5
            V=make_frame(B,q);Q=np.eye(6)-V@V.conj().T
            d=[]
            for i in range(3):
                qp=q.copy();qm=q.copy();qp[i]+=step;qm[i]-=step
                d.append((make_frame(B,qp)-make_frame(B,qm))/(2*step))
            d=np.array(d)
            qt=np.array([[a.conj().T@Q@b for b in d] for a in d])
            metric=np.trace(qt,axis1=2,axis2=3).real;inverse=np.linalg.inv(metric)
            potential=np.einsum('ab,abij->ij',inverse,qt)
            connection=np.array([1j*V.conj().T@a for a in d])
            field=np.array([.7+.2j,-.3j,.1-.4j])
            field_derivative=np.array([[.1,.2j,.3],[-.2j,.4,.1],[.5,.2,-.3j]])
            full_derivative=np.array([a@field+V@f for a,f in zip(d,field_derivative)])
            cov_derivative=np.array([f-1j*A@field for f,A in zip(field_derivative,connection)])
            norm=lambda vectors:np.einsum('ab,ai,bi->',inverse,vectors.conj(),vectors).real
            direct=norm(full_derivative)
            split=norm(cov_derivative)+(field.conj()@potential@field).real
            kinetic_error=float(abs(direct-split))
            assert kinetic_error<1e-8 and abs(np.trace(potential)-3)<1e-12
            phase_rows.append(dict(theta=theta,matrix_error=matrix_error,eigen_error=eigen_error,
                                   geometric_mass_squared=np.linalg.eigvalsh(phi).tolist(),
                                   curvature_lie_algebra_dimension_at_origin=dimension,
                                   spatial_scalar_curvature=R,
                                   curvature_second_difference_errors=curvature_errors,
                                   kinetic_action_split_error=kinetic_error,
                                   trace_geometric_mass_away_from_origin=float(np.trace(potential).real)))
        rows.append(dict(hopping_ratio=v,r=r,phases=phase_rows))

    # Rank-2 frame with the same three relation directions, for comparison.
    pauli=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1,-1]).astype(complex)]
    h2,phi2,F2=local_geometry(pauli)
    dim2=lie_dimension([F2[0,1],F2[0,2],F2[1,2]],2)
    assert dim2==3 and np.max(abs(phi2-1.5*np.eye(2)))<1e-14
    assert all(p['curvature_lie_algebra_dimension_at_origin']==8 for row in rows[:2] for p in row['phases'])
    return dict(rows=rows,symbolic_curvature=symbolic_curvature_check(),
                rank2=dict(spatial_metric=h2.tolist(),mass_squared=phi2.real.tolist(),curvature_algebra_dimension=dim2),
                length_unit='ell=1; all masses are squared and scale as ell^-2',
                hypotheses=['normalized relation graph frame', 'cyclic hopping C=I+v exp(i theta/3) S',
                            'clock-shift leakage directions', 'metric from the same frame'],
                candidate_fitted_parameters=[],full_joint_rmse=None,
                scope='Local conditional mass/curvature realization. Does not establish a dynamical gauge-group transition or select v, ell, or the state.')


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
