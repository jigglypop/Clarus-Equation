#!/usr/bin/env python3
"""CE-CR1: spatial Gaussian records and the finite-work preparation gate.
Three complex CE mediators. hbar=c=1; no observational fitting.
This is NOT a prediction of the portal, the volume, the preparation or one outcome.
"""
from __future__ import annotations
import argparse
import json
import math
import platform
import unittest
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

PI = math.pi


def masses(kappa: float = 1., u: float = .2, s0: float = .5,
           epsilon: float = .15) -> np.ndarray:
    x = s0 + kappa*u + np.array([-2*epsilon, epsilon, epsilon])
    if kappa < 0 or epsilon < 0 or np.min(x) <= 0:
        raise ValueError('Require kappa, epsilon >= 0 and positive squared masses')
    return x


def positive_pair(xa, xb):
    a, b = np.atleast_1d(xa).astype(float), np.atleast_1d(xb).astype(float)
    if a.shape != b.shape or np.min(a) <= 0 or np.min(b) <= 0:
        raise ValueError('Matching arrays of positive squared masses required')
    return a, b


def log_pair_overlap(k, xa, xb):
    """Minus log overlap of one complex mode; stable also near zero contrast."""
    r = .25*np.log1p((xa-xb)/(np.asarray(k)**2+xb))
    return np.log1p(2*np.sinh(r/2)**2)


def gamma_density(xa, xb, cutoff=np.inf):
    a, b = positive_pair(xa, xb)
    if cutoff <= 0:
        raise ValueError('Positive cutoff required')
    value = error = 0.
    for x, y in zip(a, b):
        f = lambda k: float(k*k*log_pair_overlap(k, x, y)/(2*PI**2))
        z, e = quad(f, 0., cutoff, epsabs=2e-15, epsrel=2e-10, limit=300)
        value += z; error += e
    return value, error


def small_contrast_coefficient(x):
    x = np.asarray(x, float)
    if np.min(x) <= 0:
        raise ValueError('Positive squared masses required')
    return float(np.sum(1/np.sqrt(x))/(256*PI))


def required_volume(gamma, error=.01):
    if gamma <= 0 or not 0 < error < .5:
        raise ValueError('Need gamma>0 and 0<error<1/2')
    return -math.log(4*error*(1-error))/(2*gamma)


def helstrom_error(gamma, volume):
    if gamma < 0 or volume < 0:
        raise ValueError('Nonnegative arguments required')
    z = math.exp(-2*gamma*volume)
    return z/(2*(1+math.sqrt(-math.expm1(-2*gamma*volume))))


def sphere_sum_density(xa, xb, length, cutoff):
    a, b = positive_pair(xa, xb)
    n = int(math.ceil(cutoff*length/(2*PI)))
    kv = 2*PI*np.arange(-n,n+1)/length
    yz = kv[:,None]**2+kv[None,:]**2
    total = 0.
    for kx in kv:
        k2 = kx*kx+yz
        k = np.sqrt(k2[k2 <= cutoff**2])
        for x,y in zip(a,b):
            total += float(np.sum(log_pair_overlap(k,x,y)))
    return total/length**3


def occupation(k, xa, xb, tau=0.):
    if xa <= 0 or xb <= 0 or tau < 0:
        raise ValueError('Positive masses and nonnegative switching time required')
    wa, wb = math.sqrt(k*k+xa), math.sqrt(k*k+xb)
    dw = (xb-xa)/(wa+wb)
    if dw == 0: return 0.
    if tau == 0: return dw*dw/(4*wa*wb)
    # log sinh avoids overflow for large momentum and slow preparation.
    def ls(x):
        if x < 1e-5: return math.log(x)+math.log1p(x*x/6)
        return x+math.log1p(-math.exp(-2*x))-math.log(2)
    logn = 2*ls(PI*tau*abs(dw)/2)-ls(PI*tau*wa)-ls(PI*tau*wb)
    return math.exp(logn) if logn > -745 else 0.


def excitation_energy(xa, xb, tau, cutoff=np.inf):
    a,b = positive_pair(xa,xb)
    if tau < 0 or (tau == 0 and not np.isfinite(cutoff)):
        raise ValueError('Sudden continuum quench needs finite cutoff')
    value = error = 0.
    for x,y in zip(a,b):
        f = lambda k: k*k*math.sqrt(k*k+y)*occupation(k,x,y,tau)/PI**2
        z,e = quad(f,0.,cutoff,epsabs=2e-14,epsrel=2e-9,limit=300)
        value+=z; error+=e
    return value,error


def max_adiabaticity(xa, xb, tau):
    if xa <= 0 or xb <= 0 or tau <= 0:
        raise ValueError('Positive arguments required')
    c,d=(xa+xb)/2,(xb-xa)/2
    if d == 0: return 0.
    y=-3*d/(2*c+math.sqrt(4*c*c-3*d*d))
    return abs(xb-xa)*(1-y*y)/(4*tau*(c+d*y)**1.5)


def mode_ode(k, xa, xb, tau, method='DOP853'):
    """Prescribed tanh mass ramp, one real mode; complex work is twice this."""
    if tau <= 0: raise ValueError('Positive ramp time required')
    wa,wb=math.sqrt(k*k+xa),math.sqrt(k*k+xb)
    f0=1/math.sqrt(2*wa); p0=-1j*wa*f0
    tmax=16*tau
    def rhs(t,y):
        z=math.tanh(t/tau)
        w2=k*k+(xa+xb)/2+(xb-xa)*z/2
        dx=(xb-xa)*(1-z*z)/(2*tau)
        f,p=complex(y[0],y[1]),complex(y[2],y[3])
        dp=-w2*f
        return [p.real,p.imag,dp.real,dp.imag,.5*dx*abs(f)**2]
    sol=solve_ivp(rhs,[-tmax,tmax],[f0,0.,p0.real,p0.imag,0.],
                  method=method,rtol=2e-11,atol=2e-13)
    if not sol.success: raise RuntimeError(sol.message)
    y=sol.y[:,-1];f,p=complex(y[0],y[1]),complex(y[2],y[3])
    # Stable late negative-frequency amplitude (a harmless phase is omitted).
    beta=(wb*f-1j*p)/math.sqrt(2*wb)
    e=.5*(abs(p)**2+wb*wb*abs(f)**2)
    return {'occupation':abs(beta)**2,'work_balance_error':abs(e-wa/2-y[4]),
            'wronskian_error':abs(f*np.conj(p)-np.conj(f)*p-1j),
            'work_complex':2*y[4], 'final_energy_complex':2*e}


class Checks(unittest.TestCase):
    def test_01_positive_spectrum(self):
        np.testing.assert_allclose(masses(),[.4,.85,.85])
    def test_02_zero_contrast(self):
        self.assertEqual(gamma_density(masses(),masses())[0],0.)
    def test_03_symmetry(self):
        a,b=masses(),masses(u=.8)
        self.assertAlmostEqual(gamma_density(a,b)[0],gamma_density(b,a)[0],places=13)
    def test_04_complex_gaussian_integral(self):
        for a,b in [(.4,1.),(.85,1.45),(1.,1.01)]:
            wa,wb=np.sqrt(a),np.sqrt(b)
            q,_=quad(lambda x:(wa*wb)**.25/math.sqrt(PI)*np.exp(-(wa+wb)*x*x/2),-np.inf,np.inf)
            self.assertAlmostEqual(-math.log(q*q),float(log_pair_overlap(0.,a,b)),places=12)
    def test_05_uv_asymptote(self):
        a,b=.4,1.; k=1e4
        self.assertAlmostEqual(float(log_pair_overlap(k,a,b))*k**4,(b-a)**2/32,places=8)
    def test_06_small_contrast(self):
        x=masses(u=.5);d=1e-3
        g,_=gamma_density(x-d/2,x+d/2)
        self.assertLess(abs(g/d**2/small_contrast_coefficient(x)-1),3e-7)
    def test_07_continuum_tail(self):
        a,b=masses(),masses(u=.8);g,_=gamma_density(a,b)
        gs,_=gamma_density(a,b,100.)
        tail=np.sum((a-b)**2)/(64*PI**2*100)
        self.assertLess(abs((g-gs)/tail-1),2e-4)
    def test_08_volume_threshold(self):
        g=gamma_density(masses(),masses(u=.8))[0]
        for e in [.1,.01,1e-4]:
            self.assertAlmostEqual(helstrom_error(g,required_volume(g,e)),e,places=13)
    def test_09_no_record_without_portal(self):
        self.assertEqual(gamma_density(masses(kappa=0),masses(kappa=0,u=.8))[0],0.)
    def test_10_volume_scaling(self):
        g=gamma_density(masses(),masses(u=.8))[0]
        self.assertAlmostEqual(math.exp(-g*20)**2,math.exp(-g*40),places=14)
    def test_11_mass_scaling(self):
        a,b=masses(),masses(u=.8);g=gamma_density(a,b)[0]
        self.assertAlmostEqual(gamma_density(4*a,4*b)[0]/g,8.,places=9)
    def test_12_finite_box(self):
        a,b=masses(),masses(u=.8);g=gamma_density(a,b,8.)[0]
        self.assertLess(abs(sphere_sum_density(a,b,32.,8.)/g-1),5e-4)
    def test_13_tanh_sudden_limit(self):
        for k in [0.,1.,4.]:
            self.assertLess(abs(occupation(k,.4,1.,1e-6)/occupation(k,.4,1.,0.)-1),1e-8)
    def test_14_mode_solution(self):
        for k in [0.,.7,2.]:
            for tau in [.25,1.]:
                d=mode_ode(k,.4,1.,tau)
                self.assertLess(abs(d['occupation']-occupation(k,.4,1.,tau)),3e-10)
    def test_15_work_balance(self):
        for tau in [.25,1.,2.]:
            d=mode_ode(.7,.4,1.,tau)
            self.assertLess(d['work_balance_error'],2e-9)
            self.assertLess(d['wronskian_error'],2e-9)
    def test_16_integrator_independence(self):
        a=mode_ode(.7,.4,1.,.5,'DOP853');b=mode_ode(.7,.4,1.,.5,'RK45')
        self.assertLess(abs(a['occupation']-b['occupation']),2e-9)
    def test_17_sudden_energy_log_divergence(self):
        a,b=masses(),masses(u=.8)
        lo=excitation_energy(a,b,0.,100.)[0];hi=excitation_energy(a,b,0.,200.)[0]
        slope=np.sum((b-a)**2)/(16*PI**2)
        self.assertLess(abs((hi-lo)/(slope*math.log(2))-1),2e-4)
    def test_18_finite_ramp_energy(self):
        a,b=masses(),masses(u=.8)
        x=excitation_energy(a,b,.5)[0];y=excitation_energy(a,b,.5,20.)[0]
        self.assertGreater(x,0);self.assertLess(abs(x-y),1e-11)
    def test_19_adiabatic_suppression(self):
        a,b=masses(),masses(u=.8)
        e=[excitation_energy(a,b,t)[0] for t in [.1,.5,1.,2.]]
        self.assertTrue(np.all(np.diff(e)<0))
    def test_20_exact_adiabatic_maximum(self):
        y=np.linspace(-1,1,20001);a,b=.4,1.;tau=.5
        sampled=np.max((b-a)*(1-y*y)/(4*tau*((a+b)/2+(b-a)*y/2)**1.5))
        self.assertLess(abs(sampled/max_adiabaticity(a,b,tau)-1),1e-7)
    def test_21_gap_boundary_slowdown(self):
        e=[max_adiabaticity(a,1.,1.) for a in [.1,.01,.001]]
        self.assertTrue(np.all(np.diff(e)>0))
    def test_22_record_does_not_select_coupling(self):
        vols=[]
        for k in [.25,.5,1.,2.]:
            g=gamma_density(masses(kappa=k),masses(kappa=k,u=.8))[0]
            vols.append(required_volume(g));self.assertGreater(g,0)
        self.assertTrue(np.all(np.isfinite(vols)))
    def test_23_positive_epsilon_not_required_for_record(self):
        self.assertGreater(gamma_density(masses(epsilon=0),masses(epsilon=0,u=.8))[0],0)
    def test_24_invalid_domain(self):
        with self.assertRaises(ValueError):masses(epsilon=1.)
        with self.assertRaises(ValueError):excitation_energy([.4],[1.],0.)
        with self.assertRaises(ValueError):required_volume(0.)

    def test_25_record_threshold_is_unique_given_resources(self):
        goal=-math.log(4*.01*.99)/(2*1000.)
        f=lambda k: gamma_density(masses(kappa=k),masses(kappa=k,u=.8))[0]-goal
        root=brentq(f,.01,4.,xtol=1e-12)
        self.assertGreater(root,0)
        self.assertLess(f(root*.99),0);self.assertGreater(f(root*1.01),0)
        self.assertAlmostEqual(helstrom_error(goal,1000.),.01,places=13)
    def test_26_ultraviolet_overlap_tail_bound(self):
        a,b=masses(),masses(u=.8);g=gamma_density(a,b)[0]
        for K in [1.,8.,100.]:
            missing=g-gamma_density(a,b,K)[0]
            bound=float(np.sum((a-b)**2))/(64*PI**2*K)
            self.assertGreaterEqual(missing,-1e-14)
            self.assertLessEqual(missing,bound+1e-14)

    def test_27_general_smooth_ramp_beta_bound(self):
        # Integration by parts of exact Bogoliubov evolution; tanh norms inserted.
        dx=.6
        for k in [.7,1.,2.,5.]:
            for tau in [.1,.5,1.]:
                bound=math.exp(dx/(4*k*k))*(dx/(8*tau*k**3)+7*dx*dx/(96*tau*k**5))
                self.assertLessEqual(math.sqrt(occupation(k,.4,1.,tau)),bound)


def diagnostics():
    a,b=masses(),masses(u=.8);g,err=gamma_density(a,b)
    ode=[{'k':k,'tau':t,'exact_occupation':occupation(k,.4,1.,t),**mode_ode(k,.4,1.,t)}
         for k in [0.,.7,2.] for t in [.25,1.]]
    return {'study':'CE-CR1','baseline_main':'ac06cc8f1a8508d36d550b2ed8709c02c84d397c',
      'inputs':{'s0':.5,'epsilon':.15,'kappa':1.,'ua':.2,'ub':.8,'x_a':a.tolist(),'x_b':b.tolist(),
                'units':'hbar=c=1; internal masses, not electroweak or cosmological predictions'},
      'gamma':g,'quadrature_estimated_error':err,
      'small_contrast_coefficient':small_contrast_coefficient(masses(u=.5)),
      'record_volumes':[{'target_error':e,'volume':required_volume(g,e)} for e in [.1,.01,1e-4]],
      'finite_volume':{'cutoff':8.,'continuum_at_cutoff':gamma_density(a,b,8.)[0],
          'boxes':[{'length':L,'density':sphere_sum_density(a,b,L,8.)} for L in [8.,16.,32.]]},
      'sudden_quench':{'log_energy_coefficient':float(np.sum((b-a)**2)/(16*PI**2)),
         'cutoffs':[{'cutoff':K,'excitation_energy_density':excitation_energy(a,b,0.,K)[0]} for K in [10.,30.,100.,300.]]},
      'smooth_ramps':[{'tau':t,'excitation_energy_density':excitation_energy(a,b,t)[0],
                        'max_adiabaticity':max(max_adiabaticity(x,y,t) for x,y in zip(a,b))}
                      for t in [.1,.25,.5,1.,2.]],
      'mode_checks':ode,
      'nonunique_portals':[{'kappa':k,'gamma':(gk:=gamma_density(masses(kappa=k),masses(kappa=k,u=.8))[0]),
                            'volume_for_error_01':required_volume(gk)} for k in [.25,.5,1.,2.]],
      'conditional_resource_window':{
        'given_volume':1000.,'given_optimal_discrimination_error':.01,'given_tau':3.,
        'given_adiabaticity_ceiling':.1,
        'kappa_min_at_epsilon_015':brentq(lambda k:gamma_density(masses(kappa=k),masses(kappa=k,u=.8))[0]+math.log(4*.01*.99)/(2*1000.),.01,4.,xtol=1e-12),
        'epsilon_max_at_kappa_1':brentq(lambda e:max_adiabaticity(masses(epsilon=e)[0],masses(epsilon=e,u=.8)[0],3.)-.1,0.,.249999,xtol=1e-12),
        'note':'Two separate slices, not one jointly optimized point; chosen resource thresholds are not physical constants.'},
      'epsilon_zero_gamma':gamma_density(masses(epsilon=0),masses(epsilon=0,u=.8))[0],
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
      'limits':{'original_finite_environment_replaced':False,'new_preparation':'conditional vacuum/adiabatic branch; prescribed tanh ramp for energy diagnostic',
                'single_outcome_derived':False,'irreversible_redundant_local_records_derived':False,
                'all_constants_derived':False,'full_backreaction_solved':False,'observational_fitting':False,'full_joint_rmse':None}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results.json'))
    args=parser.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not r.wasSuccessful():return 1
    out=diagnostics();out['verification']={'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(args.output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
