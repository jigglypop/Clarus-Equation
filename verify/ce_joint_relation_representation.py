"""One relation metric for ranks 1+2+3, then conditional matter charges.

Frame, multiplet content and neutral singlet are explicit hypotheses.
No assertion of a derived Standard Model spectrum or a new joint-data fit.
"""
import json
from pathlib import Path

import numpy as np
import sympy as sy
from scipy.linalg import block_diag

from ce_relation_mass_force import local_geometry, scalar_curvature, hermitian_vector
from common_spectrum_muon import exact_em, Q2
from ce_obs32_profiled_matter import ce


def full_lie_basis(generators,n):
    """Real Hermitian Lie algebra; retain the Abelian center."""
    basis=[]
    def add(A):
        A=(A+A.conj().T)/2;v=hermitian_vector(A)
        for _ in range(2):
            for b in basis:v-=np.dot(v,hermitian_vector(b))*hermitian_vector(b)
        norm=np.linalg.norm(v)
        if norm>1e-9:basis.append((v[:n*n]+1j*v[n*n:]).reshape(n,n)/norm)
    for A in generators:add(A)
    for _ in range(n*n):
        old=len(basis)
        for A in list(basis):
            for B in list(basis):add(1j*(A@B-B@A))
        if old==len(basis):return basis
    raise RuntimeError('Lie closure did not stabilize')


def joint_frame_cases():
    S=np.roll(np.eye(3,dtype=complex),1,axis=0)
    Z=np.diag(np.exp(2j*np.pi*np.arange(3)/3));Ua=[np.eye(3),Z,S@Z@Z]
    B1=[np.array([[1],[0]],complex)/2,np.array([[1j],[0]])/2,np.array([[0],[1]])/2]
    B2=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1,-1]).astype(complex)]
    rows=[]
    for theta in [0.,.7,np.pi]:
        v=.3;t=1.;a=t*t+v*v;normalization=sy.Rational(9,4)+3*a
        normalization=float(normalization)
        C=t*np.eye(3)+v*np.exp(1j*theta/3)*S
        B3=[u@C for u in Ua]
        B=[block_diag(*blocks) for blocks in zip(B1,B2,B3)]
        h,phi,F=local_geometry(B)
        expected=block_diag(np.array([[.75]]),3*np.eye(2),3*C.conj().T@C)/normalization
        assert np.max(abs(h-normalization*np.eye(3)))<1e-13
        assert np.max(abs(phi-expected))<1e-14 and abs(np.trace(phi)-3)<1e-14
        basis=full_lie_basis([F[0,1],F[0,2],F[1,2]],6)
        assert len(basis)==12
        mass_commutator=max(float(np.max(abs(phi@g-g@phi))) for g in basis)
        assert mass_commutator>1e-3
        # Center commutes with every generator; this check preserves U(1).
        center=np.diag([1.,0,0,0,0,0])
        center_projection=sum(np.vdot(b,center).real*b for b in basis)
        center_error=float(np.max(abs(center-center_projection)))
        assert center_error<1e-12
        R=scalar_curvature(B)
        predicted_R=(195/4+36*a*a+18*t*t*v*v)/normalization**2
        assert abs(R-predicted_R)<1e-12
        rows.append(dict(theta=theta,metric_coefficient=normalization,
                         mass_squared=np.linalg.eigvalsh(phi).tolist(),mass_trace=float(np.trace(phi).real),
                         curvature_algebra_dimension=len(basis),center_projection_error=center_error,
                         mass_gauge_commutator_max=mass_commutator,
                         spatial_scalar_curvature=R))
    return rows


def charges_and_indices():
    q,u=sy.symbols('q u');d=-2*q-u;l=-3*q;e=6*q
    cubic=sy.factor(6*q**3+3*u**3+3*d**3+2*l**3+e**3)
    assert sy.expand(cubic+18*q*(u-2*q)*(u+4*q))==0
    # Choose the SM-like nonzero-q branch; the q=0 branch is documented.
    fields=[('Q',3,2,sy.Rational(1,6)),('uc',3,1,sy.Rational(-2,3)),
            ('dc',3,1,sy.Rational(1,3)),('L',1,2,sy.Rational(-1,2)),
            ('ec',1,1,sy.Integer(1)),('nc',1,1,sy.Integer(0))]
    Y={name:y for name,_,_,y in fields}
    anomalies=dict(SU3_SU3_Y=2*Y['Q']+Y['uc']+Y['dc'],
                   SU2_SU2_Y=3*Y['Q']+Y['L'],
                   gravity_Y=sum(c*w*y for _,c,w,y in fields),
                   Y_cubic=sum(c*w*y**3 for _,c,w,y in fields))
    assert all(x==0 for x in anomalies.values())
    dim=sum(c*w for _,c,w,y in fields)
    TY=sum(c*w*y*y for _,c,w,y in fields)
    T2=sum(sy.Rational(c,2) for _,c,w,y in fields if w==2)
    T3=sum(sy.Rational(w,2) for _,c,w,y in fields if c==3)
    EQ2=sum(c*sum((y+t3)**2 for t3 in ([sy.Rational(-1,2),sy.Rational(1,2)] if w==2 else [0])) for _,c,w,y in fields)
    assert (dim,TY,T2,T3,EQ2)==(16,sy.Rational(10,3),2,2,sy.Rational(16,3))
    assert sum(c for _,c,w,y in fields if w==2)%2==0
    return dict(cubic_factor=str(cubic),other_branch='q=0 with u=-d is not removed by local anomalies alone',
                charges={name:str(y) for name,_,_,y in fields},anomalies={k:str(v) for k,v in anomalies.items()},
                dimension=dim,TY=str(TY),T2=str(T2),T3=str(T3),electric_charge_squared_sum=str(EQ2))


def shared_epsilon_tensor_frame():
    # The gauge factor is unbroken (no color-dependent mass splitting).
    S=np.roll(np.eye(3,dtype=complex),1,axis=0)
    Z=np.diag(np.exp(2j*np.pi*np.arange(3)/3));Ua=[np.eye(3),Z,S@Z@Z]
    B1=[np.array([[1],[0]],complex)/2,np.array([[1j],[0]])/2,np.array([[0],[1]])/2]
    B2=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]]),np.diag([1,-1]).astype(complex)]
    Bg=[block_diag(*blocks) for blocks in zip(B1,B2,Ua)]
    Vg=np.vstack([np.eye(6),np.zeros((7,6))]);dg=[np.vstack([np.zeros((6,6)),b]) for b in Bg]
    rows=[]
    for theta in [0.,.7,np.pi]:
        t=1.;v=.3;a=t*t+v*v
        C=t*np.eye(3)+v*np.exp(1j*theta/3)*S
        Bf=[]
        for i in range(3):
            b=np.zeros((9,3),complex);b[3*i:3*i+3]=C;Bf.append(b)
        Vf=np.vstack([np.eye(3),np.zeros((9,3))]);df=[np.vstack([np.zeros((3,3)),b]) for b in Bf]
        V=np.kron(Vg,Vf)
        derivative=[np.kron(x,Vf)+np.kron(Vg,y) for x,y in zip(dg,df)]
        Q=np.eye(156)-V@V.conj().T
        qt=np.array([[x.conj().T@Q@y for y in derivative] for x in derivative])
        h=np.trace(qt,axis1=2,axis2=3).real
        normalization=3*(21/4)+6*3*a
        assert np.max(abs(h-normalization*np.eye(3)))<1e-12
        phi=np.einsum('ab,abij->ij',np.linalg.inv(h),qt)
        ag=np.diag([.75,3,3,3,3,3])
        expected=(np.kron(ag,np.eye(3))+np.kron(np.eye(6),3*C.conj().T@C))/normalization
        error=float(np.max(abs(phi-expected)))
        assert error<1e-14 and abs(np.trace(phi)-3)<1e-13
        F=1j*(qt-qt.swapaxes(0,1))
        basis=full_lie_basis([F[0,1],F[0,2],F[1,2]],18)
        assert len(basis)==12
        commutator=max(float(np.max(abs(phi@g-g@phi))) for g in basis)
        assert commutator<1e-12
        eps=3*t*v/normalization
        s_singlet=(.75+3*a)/normalization
        s_nonabelian=(3+3*a)/normalization
        expected_spectrum=np.sort(np.r_[s_singlet+2*eps*np.cos((theta+2*np.pi*np.arange(3))/3),
                                        np.tile(s_nonabelian+2*eps*np.cos((theta+2*np.pi*np.arange(3))/3),5)])
        assert np.max(abs(np.linalg.eigvalsh(phi)-expected_spectrum))<1e-13
        rows.append(dict(theta=theta,metric_coefficient=normalization,
                         shared_epsilon=eps,s_singlet=s_singlet,s_nonabelian=s_nonabelian,
                         mass_matrix_error=error,mass_gauge_commutator_max=commutator,
                         curvature_algebra_dimension=len(basis),mass_trace=float(np.trace(phi).real)))
    return dict(rows=rows,frame_shape=[156,18],
                scope='Gauge and cyclic factors share one relation metric; a local construction with supplied amplitudes, no fermion mass spectrum or cosmological state.')


def unchanged_observation_check(indices):
    # These checks keep the previous background and scalar statistics. The
    # charges justify the old indices conditionally; they are not new dynamics.
    old=json.loads(Path(__file__).with_name('ce_symmetric_bao_ruler.json').read_text())
    mu=exact_em(1000.,.35*(.014414e-9)**2/1e6,0.)*float(sy.Rational(indices['electric_charge_squared_sum']))/Q2
    assert abs(mu-old['muon_EM_component'])<1e-28
    z,y,kind,cov,*_=ce.load_data();chol=np.linalg.cholesky(cov)
    mu_res=(mu-385e-12)/(np.hypot(145,620)*1e-12)
    rows=[]
    for row in old['rows']:
        residual=np.linalg.solve(chol,np.array(row['prediction'])-y)
        score=float(np.sqrt((residual@residual+mu_res**2)/14))
        assert abs(score-row['partial_rmse_14'])<1e-12
        rows.append(dict(r=row['r'],theta_initial=row['theta_initial'],partial_rmse_14=score,
                         change_from_previous=score-row['partial_rmse_14']))
    return dict(muon_EM_component=mu,rows=rows,
                scope='Same previously supplied scalar representation and background; no new reduction from deriving identical group indices.')


if __name__=='__main__':
    indices=charges_and_indices()
    result=dict(joint_frame=joint_frame_cases(),matter_representation=indices,
                shared_epsilon_tensor_frame=shared_epsilon_tensor_frame(),
                observation_check=unchanged_observation_check(indices),
                assumptions=['block relation frames of ranks 1,2,3 with the specified relative amplitudes',
                             'one chiral SM-like family plus a neutral singlet',
                             'nonzero quark-doublet hypercharge; its unit fixed by convention',
                             'original CE complex scalars copy that representation for the loop check'],
                fitted_parameters=[],new_full_joint_rmse=None)
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
