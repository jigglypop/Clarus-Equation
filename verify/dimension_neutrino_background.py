"""Continuum scalar + collisionless massive neutrinos in homogeneous GR.

Diagnostic branch: only neutrino masses depend on Q, as m_i=m_i0 exp(beta_i Q/Mpl).
Cold matter and photons are minimally coupled. Couplings and zero initial scalar
state are supplied. No perturbations, apparatus likelihood or observational fit.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_laguerre
from dimension_frequency_quadrature import FrequencyProbe


class NeutrinoBackground:
    def __init__(self, *, betas=(0.,.1,.1), cells=20, momentum_order=64,
                 mass_over_href=3., a_initial=.01, rtol=2e-9, atol=2e-11):
        self.betas=np.asarray(betas,float)
        if self.betas.shape!=(3,) or not np.isfinite(self.betas).all():
            raise ValueError('three finite couplings required')
        if not 0<a_initial<1 or not np.isfinite(mass_over_href) or mass_over_href<=0:
            raise ValueError('valid initial scale and positive scalar mass required')
        self.m0=np.array([0.,np.sqrt(.0000749),np.sqrt(.002513)])
        self.t0=1.68e-4
        self.mpl_eV,self.href_eV=2.435e27,1.4377e-33
        self.density_unit=(self.mpl_eV*self.href_eV)**2
        probe=FrequencyProbe(cells=cells,coupling=0.)
        x,weight,self.quadrature=probe.positive_quadrature(32,8)
        self.v=np.sqrt(weight);self.d=probe.m**2+probe.a*x
        self.n=len(x);self.mass2=mass_over_href**2
        y,w=roots_laguerre(momentum_order)
        self.y=y;self.fw=w/(1+np.exp(-y))*y*y/np.pi**2
        self.start=np.log(a_initial);self.rtol=rtol;self.atol=atol
        self.dust0=.9;self.photon0=.00015

    def neutrinos(self, n, q):
        temperature=self.t0*np.exp(-n)
        masses=self.m0*np.exp(self.betas*q)
        x=masses/temperature
        energy=np.hypot(self.y[None,:],x[:,None])
        pref=temperature**4/self.density_unit
        rho=pref*(energy@self.fw)
        pressure=pref*((self.y[None,:]**2/(3*energy))@self.fw)
        trace=pref*((x[:,None]*(x[:,None]/energy))@self.fw)
        return rho.sum(),pressure.sum(),float(self.betas@trace),masses

    def quantities(self,n,state,lam):
        field,speed=state[:self.n],state[self.n:2*self.n]
        q=float(self.v@field)
        rho,p,source,masses=self.neutrinos(n,q)
        dust=self.dust0*np.exp(-3*n);photons=self.photon0*np.exp(-4*n)
        kinetic=float(speed@speed)
        potential=.5*self.mass2*float((self.d*field)@field)
        h2=(dust+photons+rho+.5*kinetic+potential+lam)/3
        if not np.isfinite(h2) or h2<=0:
            raise ValueError('finite expanding branch required')
        return np.sqrt(h2),dust+4*photons/3+rho+p+kinetic,source,q,masses

    def integrate(self,lam):
        state=np.zeros(2*self.n+2)
        state[-1]=self.quantities(self.start,state,lam)[0]
        def rhs(n,state):
            h,enthalpy,source,_,_=self.quantities(n,state,lam)
            field,speed=state[:self.n],state[self.n:2*self.n]
            return np.r_[speed/h,-3*speed-(self.mass2*self.d*field+self.v*source)/h,
                         np.exp(-n)/h,-enthalpy/(2*h)]
        sol=solve_ivp(rhs,(self.start,0),state,method='DOP853',
                      rtol=self.rtol,atol=self.atol,max_step=.03,dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    def calibrate(self):
        lam=3-self.dust0-self.photon0-self.neutrinos(0,0)[0]
        for _ in range(10):
            sol=self.integrate(lam)
            h=self.quantities(0,sol.y[:,-1],lam)[0]
            if abs(h-1)<1e-11:
                break
            lam+=3*(1-h*h)
        else:
            raise RuntimeError('H0 boundary calibration did not converge')
        self.lam,self.solution=lam,sol
        return self

    def diagnostics(self):
        errors=[]
        for n in np.linspace(self.start,0,101):
            state=self.solution.sol(n)
            h=self.quantities(n,state,self.lam)[0]
            errors.append(abs(state[-1]/h-1))
        h,_,_,q,masses=self.quantities(0,self.solution.y[:,-1],self.lam)
        distances=[]
        for z in (.38,.698,1.48):
            state=self.solution.sol(-np.log1p(z))
            distances.append(float(self.solution.y[-2,-1]-state[-2]))
        return {'H0_over_Href':h,'q_today':q,'masses_today_eV':masses.tolist(),
                'DM_Href_over_c_at_z_038_0698_148':distances,
                'maximum_raychaudhuri_constraint_error':max(errors),'Lambda_input':self.lam}


def report():
    rows=[]
    for beta in (0.,.1,1.):
        base=NeutrinoBackground(betas=(0,beta,beta)).calibrate()
        fine=NeutrinoBackground(betas=(0,beta,beta),cells=40,momentum_order=96,
                                rtol=2e-11,atol=2e-13).calibrate()
        b,f=base.diagnostics(),fine.diagnostics()
        rows.append({'beta':beta,'base':b,'refined':f,
          'relative_distance_refinement':(np.array(f['DM_Href_over_c_at_z_038_0698_148'])/
                                         b['DM_Href_over_c_at_z_038_0698_148']-1).tolist()})
    return {'rows':rows,'status':'supplied_neutrino_only_exponential_branch_no_fit',
      'initial_state':'scalar positions and velocities zero at a=0.01',
      'input_masses':'NuFIT60 IC24+SK NO rank-two conditional masses at q=0',
      'normalization_inputs':{'Mpl_eV':2.435e27,'Href_eV':1.4377e-33,'Tnu0_eV':1.68e-4,
                              'Omega_cold':.3,'Omega_photon':.00005,'Mstar_over_Href':3},
      'scope':'homogeneous adiabatic collisionless neutrinos; independent initial/boundary inputs',
      'joint_rmse':None,'scientific_success':False}


if __name__=='__main__':
    result=report()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
