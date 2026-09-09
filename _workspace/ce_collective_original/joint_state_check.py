"""Disclosed extension: simultaneous slow mode and a massive collective excitation.
A synthetic initial condition, not a fitted cosmological abundance.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.linalg import null_space
from scipy.optimize import root
from collective_phase_check import Loop,difference


def main():
    loop=Loop();N=4;q=3;f=.05;R=128.;D=difference(N,q);L=D.T@D
    ell=q**np.arange(N,-1,-1,dtype=float);N2=ell@ell;F2=f*f*N2
    B=null_space(ell[None,:]);vals,vec=np.linalg.eigh(B.T@L@B);modes=B@vec
    en=np.eye(5)[:,-1];z=np.linalg.pinv(L)@en
    phi0=np.pi-.6
    def grad(th):
        v,dv,_=loop.values(th[-1]);g=R*D.T@np.sin(D@th);g[-1]+=dv
        return g
    rr=root(lambda v:B.T@grad(ell*phi0+B@v), -B.T@z*loop.values(phi0)[1]/R,tol=1e-11)
    assert np.linalg.norm(rr.fun)<1e-8
    th0=ell*phi0+B@rr.x;vlo=loop.values(th0[-1])[0]
    injected=4*vlo # explicit preparation, not a predicted density ratio
    vel0=np.sqrt(2*injected)/f*modes[:,0]
    startV=2*R*np.sum(np.sin(D@th0/2)**2)+vlo
    H0=np.sqrt((injected+startV)/3);omega=np.sqrt(R*vals[0])/f
    tend=6/H0;grid=np.linspace(0,tend,18001)
    def rhs(t,y):
        th=y[:5];vv=y[5:10];H=y[11];kin=.5*f*f*(vv@vv)
        return np.r_[vv,-3*H*vv-grad(th)/(f*f),H,-kin]
    y0=np.r_[th0,vel0,0.,H0]
    sol=solve_ivp(rhs,(0,tend),y0,t_eval=grid,method='DOP853',rtol=2e-10,atol=2e-12)
    assert sol.success
    y=sol.y;th=y[:5];vel=y[5:10];a=np.exp(y[10]);H=y[11]
    phi=(ell@th)/N2;dphi=(ell@vel)/N2
    derivatives=np.array([loop.values(v) for v in phi]);U,dU,ddU=derivatives.T
    etap=-np.outer(z,dU/R);etadot=-np.outer(z,ddU*dphi/R)
    xx=modes.T@(th-ell[:,None]*phi-etap)
    vv=modes.T@(vel-ell[:,None]*dphi-etadot)
    Eg=.5*f*f*np.sum(vv*vv,axis=0)+.5*R*np.sum(vals[:,None]*xx*xx,axis=0)
    Pg=.5*f*f*np.sum(vv*vv,axis=0)-.5*R*np.sum(vals[:,None]*xx*xx,axis=0)
    kinetic=.5*f*f*np.sum(vel*vel,axis=0)
    totalV=2*R*np.sum(np.sin(D@th/2)**2,axis=0)+np.array([loop.values(v)[0] for v in th[-1]])
    rho=kinetic+totalV;p=kinetic-totalV
    constraint=np.max(abs(3*H*H-rho))/rho[0];assert constraint<2e-7
    # Twenty-oscillation windows; no single-epoch pressure claims.
    width=20*2*np.pi/omega;start=.10*tend;intervals=[]
    while start+width<=.95*tend:
        mask=(grid>=start)&(grid<=start+width)
        t=grid[mask]
        intervals.append(dict(t_mid=float((t[0]+t[-1])/2),a_mid=float(np.exp(np.interp((t[0]+t[-1])/2,grid,y[10]))),
          w_fast=float(np.trapezoid(Pg[mask],t)/np.trapezoid(Eg[mask],t)),
          averaged_comoving_fast_energy=float(np.trapezoid((a*a*a*Eg)[mask],t)/(t[-1]-t[0])),
          w_total=float(np.trapezoid(p[mask],t)/np.trapezoid(rho[mask],t))))
        start+=width
    energies=np.array([r['averaged_comoving_fast_energy'] for r in intervals]);assert len(energies)>2
    return dict(N=N,q=q,R=R,local_f=f,F_eff=np.sqrt(F2),initial_phi=phi0,
      initial_fast_over_loop_density=4,heavy_frequency_over_H0=omega/H0,final_a=float(a[-1]),
      initial_total_w=float(p[0]/rho[0]),final_loop_fraction_approx=float(U[-1]/rho[-1]),
      friedmann_max_relative_residual=float(constraint),
      comoving_fast_energy_relative_range=float((energies.max()-energies.min())/energies.mean()),
      maximum_abs_averaged_fast_w=float(max(abs(r['w_fast']) for r in intervals)),
      averaging_windows=intervals,
      limitation='Fast/slow split is an adiabatic diagnostic around a displaced transverse minimum; only total stress is exact. Abundances and preparation supplied. No halo or CMB comparison.')

if __name__=='__main__':
    r=main();Path(__file__).with_name('joint_state_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
