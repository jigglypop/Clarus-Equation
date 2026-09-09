"""Opposite scalar splitting from a specified holomorphic messenger block.

The spurion T and its F-component are supplied. This is a local charged-sector
construction, not a complete supersymmetric cosmology or a joint fit.
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from common_spectrum_vacuum_budget import shape, shape_integral
from common_spectrum_muon import ALPHA, M_MU, Q2


def run():
    with mp.workdps(180):
        mass=mp.mpf('1000'); s=mass**2; e=mp.mpf('.025'); eps=s*e
        theta=mp.mpf('.5'); d=16
        dark=mp.mpf('.027615e-9')**4*shape(mp.mpf('.15'),theta)
        def residual(t,integral=False):
            u=shape_integral if integral else shape
            return d*s*s*((1+t)**2*u(e/(1+t),theta)
                          +(1-t)**2*u(e/(1-t),theta)-2*u(e,theta))
        # d * b^2 U_ss is the leading *symmetric* residual, b=s*t.
        b0=(1-2*e)*(1+e)**2; k=2*e**3*(1+mp.cos(theta))
        quadratic_coefficient=d*s*s/(16*mp.pi**2)*mp.log1p(k/b0)
        estimate=mp.sqrt(dark/quadratic_coefficient)
        bound=mp.exp(mp.findroot(lambda q:mp.log(residual(mp.exp(q))/dark),
                                mp.log(estimate),tol=mp.mpf('1e-150')))
        m=mp.sqrt((s+mp.sqrt(s*s-4*eps*eps))/2); lam=eps/m
        f=mp.mpf('2.435e18')
        spurion_F=mp.sqrt(dark)  # Diagnostic energy input, NOT predicted.
        physical_b=spurion_F*lam/(3*f)
        probe=physical_b/s
        def determinant(a,angle):
            return a**3-3*a*eps**2+2*eps**3*mp.cos(angle)
        def log_threshold(a):
            return mp.log(determinant(a,theta)/determinant(a,mp.pi))
        def p1(a):
            return 3*(a*a-eps*eps)/determinant(a,theta)
        def gauge(t):
            return -mp.mpf(str(Q2))/(12*mp.pi)*(log_threshold(s*(1+t))+
                    log_threshold(s*(1-t))+4*log_threshold(s))
        def muon_leading(t):
            return (mp.mpf(str(ALPHA))/mp.pi)**2*mp.mpf(str(M_MU))**2*mp.mpf(str(Q2))* (
                (p1(s*(1+t))+p1(s*(1-t)))/360+p1(s)/45)
        visible=dict(gauge_threshold_at_degeneracy=float(gauge(0)),
            gauge_fractional_change_at_spurion_probe=float((gauge(probe)-gauge(0))/gauge(0)),
            muon_em_heavy_leading_at_degeneracy=float(muon_leading(0)),
            muon_em_leading_fractional_change_at_spurion_probe=float((muon_leading(probe)-muon_leading(0))/muon_leading(0)),
            scope="one-loop phase threshold and heavy-leading EM subset; not full observed couplings or g-2")
        rows=[]
        for t in [mp.mpf('1e-8'),mp.mpf('1e-16'),bound,probe]:
            v=residual(t); vi=residual(t,True)
            err=abs(v-vi)/abs(v)
            assert err<mp.mpf('1e-65')
            rows.append(dict(opposite_fractional_mass_squared_split=float(t),
                             residual_GeV4=float(v),residual_over_dark=float(v/dark),
                             independent_integral_relative_error=float(err),
                             quadratic_relative_error=float(abs(v-quadratic_coefficient*t*t)/abs(v))))
        # Independent diagonalization in the original flavor basis at a
        # resolvable splitting. Tiny diagnostic splittings need high precision.
        cycle=np.roll(np.eye(3,dtype=complex),1,axis=0)
        mat=float(m)*np.eye(3)+float(lam)*np.exp(1j*float(theta)/3)*cycle
        x=np.linalg.eigvalsh(mat.conj().T@mat)
        target=np.sort([float(s+2*eps*mp.cos((theta+2*mp.pi*j)/3)) for j in range(3)])
        b_test=float(s)*.02
        B=b_test*np.exp(1j*float(theta)/3)*cycle
        A=mat.conj().T@mat
        full=np.block([[A,B.conj().T],[B,A]])
        eigen=np.linalg.eigvalsh(full)
        scalar_target=np.sort(np.concatenate([target+b_test,target-b_test]))
        matrix_error=float(np.max(abs(eigen-scalar_target))/float(s))
        fermion_error=float(np.max(abs(x-target))/float(s))
        assert matrix_error<1e-12 and fermion_error<1e-12
        return dict(role="opposite_split_and_spurion_consistency_screen",
            assumptions=["T=i*f*theta at fixed real partner", "canonical messenger Kahler metric",
                         "M(T)=m*I+lambda*exp(T/(3*f))*S", "constant supplied F_T in relative subtraction",
                         "no additional common scalar soft mass", "flat one-loop relative potential"],
            inputs=dict(mass_GeV=1000.,epsilon_over_s=.025,theta=.5,representation_dimension=16,
                        dark_comparison_GeV4=float(dark),f_GeV=float(f)),
            factorization=dict(m_GeV=float(m),lambda_GeV=float(lam),
                               mass_squared_relative_error=fermion_error,scalar_6x6_relative_error=matrix_error),
            opposite_split_at_dark_scale=float(bound),
            spurion_probe=dict(F_GeV2=float(spurion_F),b_GeV2=float(physical_b),
                               b_over_s=float(probe),F_source="square root of supplied dark comparison; not a predicted vacuum"),
            visible_response=visible,
            checks=rows,decimal_precision=180,
            joint_rmse=None,scientific_success=False,
            missing=["origin and dynamics of T and F_T", "stabilization of real partner",
                     "full interacting symmetry including neutral/visible sectors and gravity",
                     "curvature, kinetic, and full muon matching", "common observational likelihood"])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
