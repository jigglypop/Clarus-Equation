"""CE-GN2: finite color frame, full holonomy and retained-gap dynamics.

All frames, coordinates, gap and flat spacetime are supplied. This verifies a
conditional construction, not dynamical Yang-Mills or quantum gravity.
"""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


X=np.array([[0,1],[1,0]],dtype=complex)
KS=[np.kron(X,np.eye(2)),np.kron(np.eye(2),X),np.kron(X,X)]
V0=np.eye(4,dtype=complex)[:,:3]
Q0=np.diag([0.,0.,0.,1.])
SMALL=[k[:3,:3] for k in KS]
CONNECTION=[-k for k in SMALL]


def comm(a,b):return a@b-b@a


def exact_rank(matrices):
    array=np.array([np.r_[m.real.ravel(),m.imag.ravel()] for m in matrices])
    assert np.max(np.abs(array-np.rint(array)))==0.
    rows=[[Fraction(int(x)) for x in row] for row in array]
    rank=0
    for col in range(len(rows[0])):
        pivot=next((i for i in range(rank,len(rows)) if rows[i][col]),None)
        if pivot is None:continue
        rows[rank],rows[pivot]=rows[pivot],rows[rank]
        divisor=rows[rank][col]
        rows[rank]=[x/divisor for x in rows[rank]]
        for i in range(len(rows)):
            if i!=rank:
                factor=rows[i][col]
                rows[i]=[x-factor*y for x,y in zip(rows[i],rows[rank])]
        rank+=1
        if rank==len(rows):break
    assert rank==np.linalg.matrix_rank(array,tol=1e-10)
    return rank


def frame(q):
    u=np.eye(4,dtype=complex)
    for value,k in zip(q,KS):u=u@(np.cos(value)*np.eye(4)+1j*np.sin(value)*k)
    return u@V0


def algebra_audit():
    assert all(np.max(np.abs(comm(a,b)))==0 for a in KS for b in KS)
    qgt=np.array([[V0.conj().T@a@Q0@b@V0 for b in KS] for a in KS])
    metric=np.trace(qgt,axis1=2,axis2=3).real
    assert np.array_equal(metric,np.eye(3))
    phi=sum(qgt[a,a] for a in range(3))
    assert np.array_equal(phi,np.eye(3))
    assert np.array_equal(sum(k@k for k in SMALL),2*np.eye(3))
    curvatures=[1j*(qgt[a,b]-qgt[b,a]) for a,b in ((0,1),(0,2),(1,2))]
    assert all(np.array_equal(f,-1j*comm(CONNECTION[a],CONNECTION[b])) for f,(a,b) in zip(curvatures,((0,1),(0,2),(1,2))))
    derivatives=[-1j*comm(a,f) for a in CONNECTION for f in curvatures]
    r0=exact_rank(curvatures);r1=exact_rank(curvatures+derivatives)
    assert r0==3 and r1==8
    assert all(np.trace(m)==0 for m in curvatures+derivatives)
    assert exact_rank(curvatures+[1j*comm(a,b) for a in curvatures for b in curvatures])==3
    currents=[]
    for b in range(3):
        j=sum((-1j*comm(CONNECTION[a],-1j*comm(CONNECTION[a],CONNECTION[b])) for a in range(3)),np.zeros((3,3),complex))
        assert np.array_equal(j,2*SMALL[b])
        currents.append(j)
    current_norm=float(sum(np.vdot(j,j).real for j in currents))
    assert current_norm==24.
    return dict(curvature_only_exact_rank=r0,curvature_and_covariant_derivative_exact_rank=r1,
                mass_matrix='I_3 / ell^2',metric='ell^2 I_3',
                source_free_Yang_Mills_current_squared_norm_at_ell_1=current_norm)


def finite_frame_audit():
    results=[]
    expected=np.array([[V0.conj().T@a@Q0@b@V0 for b in KS] for a in KS])
    for q in ([0.,0.,0.],[.2,.2,0.],[.2,.3,.4],[1.,-.5,.7]):
        v=frame(q);normal=np.eye(4)-v@v.conj().T
        derivative=[1j*k@v for k in KS]
        qgt=np.array([[a.conj().T@normal@b for b in derivative] for a in derivative])
        h=np.trace(qgt,axis1=2,axis2=3).real
        phi=np.einsum('ab,abij->ij',np.linalg.inv(h),qgt)
        errors=[float(np.max(np.abs(v.conj().T@v-np.eye(3)))),
                float(np.max(np.abs(qgt-expected))),float(np.max(np.abs(phi-np.eye(3))))]
        assert max(errors)<1e-12
        results.append(dict(q=q,normalization_error=errors[0],tensor_error=errors[1],mass_error=errors[2]))
    differences=[]
    q=np.array([.2,.3,.4]);v=frame(q);normal=np.eye(4)-v@v.conj().T
    for step in (1e-3,5e-4,2.5e-4):
        dv=[(frame(q+step*np.eye(3)[a])-frame(q-step*np.eye(3)[a]))/(2*step) for a in range(3)]
        qgt=np.array([[a.conj().T@normal@b for b in dv] for a in dv])
        h=np.trace(qgt,axis1=2,axis2=3).real
        phi=np.einsum('ab,abij->ij',np.linalg.inv(h),qgt)
        differences.append(dict(step=step,metric_error=float(np.max(np.abs(h-np.eye(3)))),
                                mass_error=float(np.max(np.abs(phi-np.eye(3))))))
    assert differences[-1]['metric_error']<differences[0]['metric_error']/10
    assert max(r['mass_error'] for r in differences)<1e-10
    return dict(points=results,finite_difference=differences)


def retained_gap_audit():
    rows=[]
    for momentum in ([0.,0.,0.],[.2,.3,.4],[.5,-.2,.1]):
        p=np.array(momentum);m0_squared=1.
        projected=sum(((p[a]*np.eye(3)+SMALL[a])@(p[a]*np.eye(3)+SMALL[a]) for a in range(3)),np.zeros((3,3),complex))+(1+m0_squared)*np.eye(3)
        approximate=np.linalg.eigvalsh(projected)
        errors=[]
        for gap_squared in (4.,16.,64.,256.):
            full=sum(((p[a]*np.eye(4)+KS[a])@(p[a]*np.eye(4)+KS[a]) for a in range(3)),np.zeros((4,4),complex))+m0_squared*np.eye(4)+gap_squared*Q0
            assert np.max(np.abs(full[:3,:3]-projected))<1e-12
            b=full[:3,3:];d=full[3,3].real;z=-1.
            sigma=b@b.conj().T/(z-d)
            direct=np.linalg.inv(z*np.eye(4)-full)[:3,:3]
            reduced=np.linalg.inv(z*np.eye(3)-projected-sigma)
            residual=float(np.max(np.abs(direct-reduced)))
            assert residual<1e-12
            eigenvalues=np.linalg.eigvalsh(full)
            assert eigenvalues[0]>0
            error=float(np.max(np.abs(eigenvalues[:3]-approximate)))
            errors.append(error)
            expected_norm=4*np.dot(p,p)/abs(z-d)
            assert abs(np.linalg.norm(sigma,2)-expected_norm)<1e-12
            rows.append(dict(momentum=momentum,gap_squared=gap_squared,
                             full_low_eigenvalues=eigenvalues[:3].tolist(),
                             projected_eigenvalues=approximate.tolist(),
                             max_projection_error=error,schur_resolvent_error=residual,
                             self_energy_norm_at_z_minus_1=expected_norm))
        if np.linalg.norm(p)>0:
            assert errors[-1]<errors[0]/25 and all(a>b for a,b in zip(errors,errors[1:]))
        else:assert max(errors)<1e-12
    return rows


def zero_gap_control():
    p=np.array([.2,.3,.4]);m0=1.
    full=sum(((p[a]*np.eye(4)+KS[a])@(p[a]*np.eye(4)+KS[a]) for a in range(3)),np.zeros((4,4),complex))+m0*np.eye(4)
    expected=np.sort([sum((p+np.array([a,b,a*b]))**2)+m0 for a in (-1,1) for b in (-1,1)])
    error=float(np.max(np.abs(np.linalg.eigvalsh(full)-expected)))
    assert error<1e-12
    return dict(full_four_mode_free_shift_spectrum_error=error,
                statement='No gap: full commuting connection is pure gauge; a three-mode projection is not an exact dynamics.')


def main():
    result=dict(candidate='CE-GN2',status='conditional color geometry survives; source-free Yang-Mills interpretation rejected',
                algebra=algebra_audit(),frame=finite_frame_audit(),gap_dynamics=retained_gap_audit(),
                zero_gap_control=zero_gap_control(),
                not_derived=['other two gauge sectors on the same metric','independent Yang-Mills boson dynamics',
                             'frame and gap preparation','physical record selection','dynamic metric and Einstein limit',
                             'joint observational RMSE'],
                provenance=dict(python=platform.python_version(),numpy=np.__version__,
                                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
