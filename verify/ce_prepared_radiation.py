"""Propagate QP27 mean abundance through a supplied post-annihilation radiation era.

No dark-abundance fit. Inflation, radiation, baryons, f and phase remain inputs.
Massless-neutrino radiation approximation; no recombination or perturbation likelihood.
"""
import json
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from ce_obs32_profiled_matter import ce
from ce_prepared_background import PreparedSpectrum

MP=2.435e18
HBAR=6.582119569e-25
MPC_KM=3.0856775814913673e19
RHO100=3*MP**2*(100/MPC_KM*HBAR)**2
T0=2.7255*8.617333262e-14
RHO_R=np.pi**2/15*T0**4*(1+(7/8)*(4/11)**(4/3)*3.046)


class RadiationBackground(ce.Background):
    def __init__(self,p,weights,scale,rho_c,method='DOP853'):
        self.p=p; self.sp=PreparedSpectrum(p,weights)
        self.sp.weighted_norm=self.sp.number_weights@np.sqrt(self.sp.eigs(p.theta_initial)[0])
        self.rb=.02237*RHO100/scale
        self.rc=rho_c/scale
        self.rr=RHO_R/scale
        self.Ni=np.log(p.ai)
        first=self.at(self.Ni,p.theta_initial,0.)
        velocity=-first['rho_c']*first['m1']/first['m']/(2*first['Z']*first['H2'])
        self.sol=solve_ivp(self.rhs,(self.Ni,0),[p.theta_initial,velocity],method=method,
            rtol=2e-10,atol=2e-12,max_step=.05,dense_output=True)
        assert self.sol.success
        self.H0=self.quantities(0)['H']

    def at(self,n,theta,v):
        U,U1,U2,m,m1,m2,Z,Z1=self.sp.evaluate(theta)
        rb=self.rb*np.exp(-3*n);rc=self.rc*np.exp(-3*n)*m;rr=self.rr*np.exp(-4*n)
        denominator=3-.5*Z*v*v
        if denominator<=0:raise ValueError('Friedmann denominator')
        h2=(rb+rc+rr+U)/denominator
        if h2<=0:raise ValueError('Nonpositive H squared')
        kinetic=.5*Z*h2*v*v;hn=-(rb+rc+4*rr/3+2*kinetic)/(2*h2)
        return dict(U=U,U1=U1,U2=U2,m=m,m1=m1,m2=m2,Z=Z,Z1=Z1,
            rho_b=rb,rho_c=rc,rho_r=rr,H=np.sqrt(h2),H2=h2,kinetic=kinetic,
            hN=hn,theta=theta,v=v,omega_phi=(U+kinetic)/(3*h2),w_phi=(kinetic-U)/(kinetic+U))

    def continuity(self):
        errors=[]
        for n in np.linspace(self.Ni,0,101):
            b=self.quantities(n);v=b['v'];vp=self.rhs(n,[b['theta'],v])[1]
            derivative=(-3*b['rho_b']+b['rho_c']*(-3+b['m1']/b['m']*v)-4*b['rho_r']
                +b['U1']*v+.5*b['Z1']*b['H2']*v**3+b['Z']*b['H2']*v*(vp+b['hN']*v))
            expected=-3*b['rho_b']-3*b['rho_c']-4*b['rho_r']-6*b['kinetic']
            errors.append(abs(derivative-expected)/(3*b['H2']))
        return max(errors)


def run():
    scales=json.loads(Path(__file__).with_name('ce_obs32_common_mass.json').read_text())
    _,_,_,_,zs,obs,cov=ce.load_data();chol=np.linalg.cholesky(cov)
    rows=[]
    for r,mass_ev in [(.15,.027615),(.35,.014414)]:
        scale=next(x['potential_scale_GeV4'] for x in scales['cases'] if x['r']==r)
        for duration in [60,10**11]:
            with mp.workdps(60):
                s=(mp.mpf(str(mass_ev))*mp.mpf('1e-9'))**2
                x=[s*(1+2*mp.mpf(str(r))*mp.cos((mp.mpf('.5')+2*mp.pi*j)/3)) for j in range(3)]
                hi=mp.mpf('1e7');hr=mp.mpf('8.5471961e-45')
                variance=[3*hi**4/(8*mp.pi**2*a)*(-mp.expm1(-2*a*duration/(3*hi**2))) for a in x]
                number=[4*mp.gamma(mp.mpf('1.25'))**2/mp.pi*hr**mp.mpf('1.5')*2*v/a**mp.mpf('.25') for v,a in zip(variance,x)]
                rho_c=float(sum(n*mp.sqrt(a) for n,a in zip(number,x)))
                weights=[float(n/sum(number)) for n in number]
            p=ce.Parameters(r,.5,ai=1e-8,s_over_Mp2=(mass_ev*1e-9/MP)**2)
            bg=RadiationBackground(p,weights,scale,rho_c)
            check=RadiationBackground(p,weights,scale,rho_c,'Radau')
            pred=np.array([bg.distances(z)['F_AP'] for z in zs])
            other=np.array([check.distances(z)['F_AP'] for z in zs])
            err=float(np.max(abs(pred-other)));assert err<1e-8
            end=bg.quantities(0);white=np.linalg.solve(chol,pred-obs)
            continuity=bg.continuity();assert continuity<1e-12
            rows.append(dict(r=r,N_pre=duration,prepared_omega_c_h2=rho_c/RHO100,
                evolved_omega_c_h2=end['rho_c']*scale/RHO100,
                conditional_H0=bg.H0*np.sqrt(scale)/MP/HBAR*MPC_KM,
                theta_today=end['theta'],theta_N_today=end['v'],AP_rmse=float(np.linalg.norm(white)/np.sqrt(6)),
                conservation_error=continuity,solver_max_AP_difference=err))
    return dict(scope='conditional mean preparation plus post-annihilation background',
        H_I_GeV=1e7,formation_H_R_GeV=8.5471961e-45,omega_b_h2=.02237,
        T0_K=2.7255,Neff_massless=3.046,initial_theta=.5,ai=1e-8,
        cases=rows,fitted_parameters=0,joint_rmse=None,
        missing=['inflation and reheating dynamics','massive neutrino transition','recombination','primordial perturbations','full quantum-sector backreaction'])


if __name__=='__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
