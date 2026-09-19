"""CE-TH1: energy-resolved collective thermalization; no observational fitting.
Units: hbar=k_B=1; beta and all diagnostic energies are dimensionless.
Run: OPENBLAS_NUM_THREADS=1 python thermal_origin.py --output results.json
"""
from __future__ import annotations
import argparse
import json
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import expm, null_space
from scipy.integrate import solve_ivp

ALPHA_S = 0.11789
P = 4 * ALPHA_S ** (4/3)
DELTA = P * (1-P)
MASS = np.sqrt([1-(1+np.sqrt(3))*.15, 1+(np.sqrt(3)-1)*.15, 1.3])
I3 = np.eye(3, dtype=complex)
I9 = np.eye(9, dtype=complex)
SWAP = np.zeros((9,9), complex)
for i in range(3):
    for j in range(3):
        SWAP[3*j+i,3*i+j] = 1
PP, PM = (I9+SWAP)/2, (I9-SWAP)/2
RHO1 = np.diag([P,1-P,0]).astype(complex)
INITIAL = np.kron(RHO1,RHO1)

def tr(a):
    return float(np.trace(a).real)

def pair_h(energies=MASS):
    h = np.diag(energies)
    return np.kron(h,I3)+np.kron(I3,h)

def collective(a, sign=1):
    return np.kron(a,I3)+sign*np.kron(I3,a)

def lindblad(j):
    jdj = j.conj().T@j
    return np.kron(j.conj(),j)-.5*(np.kron(I9,jdj)+np.kron(jdj.T,I9))

def generator(beta: float, energies=MASS, leakage: float=0.0):
    """Thermal qubit collisions; beta=0, leakage=0 recovers CS1 exactly.
    Leakage is an independently labeled diagnostic: local rather than
    collective thermal jumps. Its value is not used to fit DELTA.
    """
    if beta < 0 or leakage < 0:
        raise ValueError('beta and leakage must be nonnegative')
    h = pair_h(energies)
    out = -1j*(np.kron(I9,h)-np.kron(h.T,I9))
    for low in range(3):
        for high in range(low+1,3):
            gap = float(energies[high]-energies[low])
            if gap <= 0:
                raise ValueError('energies must be strictly increasing')
            pe = 1/(1+np.exp(beta*gap))
            a = np.zeros((3,3),complex); a[high,low]=1
            up = collective(a)
            out += pe*lindblad(up)+(1-pe)*lindblad(up.conj().T)
            if leakage:
                for local in (np.kron(a,I3),np.kron(I3,a)):
                    out += leakage*(pe*lindblad(local)+(1-pe)*lindblad(local.conj().T))
    for diagonal in (np.diag([1,-1,0])/2, np.diag([1,1,-2])/(2*np.sqrt(3))):
        out += lindblad(collective(diagonal))
    return out

def sector_states(beta: float, energies=MASS):
    """Stable restricted Gibbs states; relative energies avoid overflow."""
    h = pair_h(energies)
    e = np.diag(h).real
    boltz = np.diag(np.exp(-beta*(e-e.min())))
    zp, zm = tr(PP@boltz), tr(PM@boltz)
    return PP@boltz/zp, PM@boltz/zm

def state(beta: float, weight=DELTA, energies=MASS):
    rp,rm = sector_states(beta,energies)
    return (1-weight)*rp+weight*rm

def gibbs(beta: float, energies=MASS):
    w=np.exp(-beta*(energies-np.min(energies))); w/=w.sum()
    return np.diag(w).astype(complex)

def stats(beta: float, energies=MASS, weight=DELTA):
    h=pair_h(energies); rp,rm=sector_states(beta,energies)
    ep,em=tr(rp@h),tr(rm@h)
    vp,vm=tr(rp@h@h)-ep*ep,tr(rm@h@h)-em*em
    ebar=(1-weight)*ep+weight*em
    rho=gibbs(beta,energies); rho2=gibbs(2*beta,energies)
    purity=tr(rho@rho)
    uprime=2*purity*(tr(rho@np.diag(energies))-tr(rho2@np.diag(energies)))
    return dict(beta=float(beta), energy_plus=ep, energy_minus=em,
                energy_mean=ebar, variance_plus=vp, variance_minus=vm,
                variance_ratio=vp/vm if vm>1e-15 else None,
                event_probability=float(weight), energy_fraction=weight*em/ebar,
                relative_energy_bias=em/ebar-1,
                gibbs_purity=purity, gibbs_selected_weight=(1-purity)/2,
                split_from_purity=2*uprime/(1-purity*purity))

def entropy(rho):
    w=np.linalg.eigvalsh((rho+rho.conj().T)/2)
    w=w[w>1e-15]
    return float(-np.sum(w*np.log(w)))

def free_energy(beta, energies=MASS, weight=DELTA):
    if beta<=0:
        raise ValueError('free energy uses finite positive beta')
    rho=state(beta,weight,energies)
    return tr(rho@pair_h(energies))-entropy(rho)/beta

def collision(beta=.7, dt=.1, energies=MASS, high=2, low=0):
    """Exact finite energy-preserving 9 x 2 collision, not a master approximation."""
    h=pair_h(energies); gap=energies[high]-energies[low]
    a=np.zeros((3,3),complex); a[high,low]=1
    up=collective(a)
    minus=np.array([[0,1],[0,0]],complex)
    v=np.kron(up,minus)+np.kron(up.conj().T,minus.conj().T)
    hb=np.diag([0,gap]).astype(complex)
    hf=np.kron(h,np.eye(2))+np.kron(I9,hb)
    pe=1/(1+np.exp(beta*gap)); rb=np.diag([1-pe,pe])
    initial=np.kron(INITIAL,rb)
    unit=expm(-1j*np.sqrt(dt)*v)
    final=unit@initial@unit.conj().T
    rs=np.trace(final.reshape(9,2,9,2),axis1=1,axis2=3)
    re=np.trace(final.reshape(9,2,9,2),axis1=0,axis2=2)
    ds=tr(h@rs)-tr(h@INITIAL)
    db=tr(hb@re)-tr(hb@rb)
    return dict(energy_commutator=float(np.linalg.norm(v@hf-hf@v)),
                unitarity=float(np.linalg.norm(unit.conj().T@unit-np.eye(18))),
                system_heat=ds, bath_heat=db, energy_balance=ds+db,
                swap_change=tr(PM@rs)-DELTA)

def finite_collision_error(beta,dt,high=2,low=0):
    a=np.zeros((3,3),complex); a[high,low]=1
    up=collective(a); gap=MASS[high]-MASS[low]
    pe=1/(1+np.exp(beta*gap))
    minus=np.array([[0,1],[0,0]],complex)
    v=np.kron(up,minus)+np.kron(up.conj().T,minus.conj().T)
    rb=np.diag([1-pe,pe]); initial=np.kron(INITIAL,rb)
    unit=expm(-1j*np.sqrt(dt)*v)
    y=unit@initial@unit.conj().T
    rs=np.trace(y.reshape(9,2,9,2),axis1=1,axis2=3)
    l=pe*lindblad(up)+(1-pe)*lindblad(up.conj().T)
    deriv=(l@INITIAL.reshape(-1,order='F')).reshape(9,9,order='F')
    return float(np.linalg.norm(rs-INITIAL-dt*deriv))

def evolution(beta=1.0,t=16.0):
    gen=generator(beta); vec=INITIAL.reshape(-1,order='F')
    y=(expm(t*gen)@vec).reshape(9,9,order='F')
    sol=solve_ivp(lambda tt,v:gen@v,[0,t],vec,method='DOP853',rtol=2e-11,atol=2e-13)
    yy=sol.y[:,-1].reshape(9,9,order='F')
    target=state(beta)
    evals=np.linalg.eigvals(gen)
    decays=-evals.real[evals.real<-1e-9]
    return dict(beta=beta,time=t,stationary_residual=float(np.linalg.norm(gen@target.reshape(-1,order='F'))),
                stationary_error=float(np.linalg.norm(y-target)),
                expm_ode_difference=float(np.linalg.norm(y-yy)),
                zero_eigenvalues=int(np.sum(np.abs(evals)<1e-9)),
                spectral_gap=float(decays.min()),minimum_state_eigenvalue=float(np.linalg.eigvalsh(y).min()),
                trace_error=abs(tr(y)-1),swap_error=abs(tr(PM@y)-DELTA))

def pressure_demo(beta=1.0,volume=2.0,momentum=.3,weight=DELTA):
    energies=np.sqrt(MASS*MASS+momentum**2*volume**(-2/3))
    e_v=-momentum**2*volume**(-5/3)/(3*energies)
    h=pair_h(energies); hv=pair_h(e_v)
    rp,rm=sector_states(beta,energies); rho=(1-weight)*rp+weight*rm
    press=-tr(rho@hv)
    cov=0.0
    for w,r in ((1-weight,rp),(weight,rm)):
        cov+=w*(tr(r@h@hv)-tr(r@h)*tr(r@hv))
    step=volume*1e-4
    def ee(v):
        en=np.sqrt(MASS*MASS+momentum**2*v**(-2/3))
        rr=state(beta,weight,en)
        return tr(rr@pair_h(en))
    def ff(v):
        en=np.sqrt(MASS*MASS+momentum**2*v**(-2/3))
        return free_energy(beta,en,weight)
    def minus_deriv(fun):
        return -(fun(volume-2*step)-8*fun(volume-step)+8*fun(volume+step)-fun(volume+2*step))/(12*step)
    return dict(pressure=press,minus_free_energy_derivative=minus_deriv(ff),
                minus_energy_derivative=minus_deriv(ee),covariance_term=beta*cov,
                predicted_minus_energy_derivative=press+beta*cov,
                w_total=press*volume/tr(rho@h))

class Tests(unittest.TestCase):
    def test_projectors(self):
        for r in (PP,PM): self.assertLess(np.linalg.norm(r@r-r),1e-14)
        self.assertEqual((tr(PP),tr(PM)),(6.,3.))
    def test_initial_weight(self): self.assertAlmostEqual(tr(PM@INITIAL),DELTA,14)
    def test_zero_inverse_temperature_recovers_cs1(self):
        self.assertLess(np.linalg.norm(state(0)-((1-DELTA)*PP/6+DELTA*PM/3)),1e-14)
    def test_partition_trace(self):
        for b in (.1,1.,5.):
            z=np.sum(np.exp(-b*MASS)); z2=np.sum(np.exp(-2*b*MASS))
            h=pair_h(); boltz=expm(-b*h)
            self.assertAlmostEqual(tr(PP@boltz),.5*(z*z+z2),13)
            self.assertAlmostEqual(tr(PM@boltz),.5*(z*z-z2),13)
    def test_stationarity(self):
        for b in (0.,.1,1.,5.,10.):
            self.assertLess(np.linalg.norm(generator(b)@state(b).reshape(-1,order='F')),2e-14)
    def test_trace_and_swap(self):
        for b in (0.,1.,5.):
            l=generator(b)
            for obs in (I9,PM): self.assertLess(np.linalg.norm(l.conj().T@obs.reshape(-1,order='F')),1e-13)
    def test_kernel_dimension(self):
        for b in (0.,1.,5.): self.assertEqual(null_space(generator(b),rcond=1e-10).shape[1],2)
    def test_cs1_spectrum(self):
        # Only imaginary Hamiltonian shifts occur, since the collective
        # isotropic dissipator commutes with the free collective evolution.
        ev=np.linalg.eigvals(generator(0)).real
        for v,cnt in ((0,2),(-1.5,32),(-3,20),(-4,27)):
            self.assertEqual(np.sum(abs(ev-v)<1e-8),cnt)
    def test_energy_split(self):
        for b in (.1,1.,5.,10.):
            s=stats(b); self.assertGreater(s['energy_minus'],s['energy_plus'])
            self.assertAlmostEqual(s['energy_minus']-s['energy_plus'],s['split_from_purity'],12)
    def test_variance_ratio(self): self.assertAlmostEqual(stats(0)['variance_ratio'],2.5,11)
    def test_bias_sign(self):
        for b in (.1,1.,5.,10.): self.assertGreater(stats(b)['energy_fraction'],DELTA)
    def test_high_temperature_derivative(self):
        vh=np.mean(MASS*MASS)-np.mean(MASS)**2
        predicted=(1-DELTA)*1.5*vh/(2*np.mean(MASS))
        numerical=(stats(1e-4)['relative_energy_bias']-stats(-1e-4)['relative_energy_bias'])/(2e-4)
        self.assertLess(abs(numerical-predicted),1e-7)
    def test_collision_conservation(self):
        s=collision()
        for key in ('energy_commutator','unitarity','energy_balance','swap_change'):
            self.assertLess(abs(s[key]),1e-13)
    def test_collision_limit(self):
        errors=[finite_collision_error(1,d) for d in (.02,.01,.005)]
        for a,b in zip(errors[:-1],errors[1:]): self.assertTrue(3.8<a/b<4.2)
    def test_thermal_stationarity_finite_collision(self):
        b=1.; high=2; low=0; a=np.zeros((3,3),complex); a[high,low]=1
        up=collective(a); gap=MASS[high]-MASS[low]; pe=1/(1+np.exp(b*gap))
        minus=np.array([[0,1],[0,0]],complex)
        v=np.kron(up,minus)+np.kron(up.conj().T,minus.conj().T)
        rr=np.kron(state(b),np.diag([1-pe,pe])); uu=expm(-.4j*v)
        self.assertLess(np.linalg.norm(uu@rr@uu.conj().T-rr),1e-14)
    def test_free_energy_minimum(self):
        b=1.; eq=state(b); candidate=INITIAL; h=pair_h()
        f0=tr(candidate@h)-entropy(candidate)/b
        fe=tr(eq@h)-entropy(eq)/b
        vals,vec=np.linalg.eigh(eq); logeq=(vec*np.log(vals))@vec.conj().T
        rel=-entropy(candidate)-tr(candidate@logeq)
        self.assertGreater(rel,0)
        self.assertAlmostEqual(f0-fe,rel/b,12)
    def test_independent_bath_selects_gibbs(self):
        for b in (0.,1.,5.):
            target=np.kron(gibbs(b),gibbs(b)); l=generator(b,leakage=.03)
            self.assertLess(np.linalg.norm(l@target.reshape(-1,order='F')),2e-14)
            self.assertEqual(null_space(l,rcond=1e-10).shape[1],1)
            self.assertAlmostEqual(tr(PM@target),stats(b)['gibbs_selected_weight'],13)
    def test_pressure_derivative(self):
        s=pressure_demo()
        self.assertLess(abs(s['pressure']-s['minus_free_energy_derivative']),2e-11)
    def test_heat_correction(self):
        s=pressure_demo()
        self.assertLess(abs(s['minus_energy_derivative']-s['predicted_minus_energy_derivative']),2e-11)
    def test_pressure_sign(self):
        for b in (.1,1.,5.): self.assertTrue(0<pressure_demo(beta=b)['w_total']<1/3)
    def test_ode_crosscheck(self): self.assertLess(evolution()['expm_ode_difference'],2e-10)
    def test_global_bias_bound(self):
        for b in (.01,.1,1.,5.,10.):
            bound=(1-DELTA)*b*(MASS[-1]-MASS[0])**2/(2*MASS[0])
            self.assertLessEqual(stats(b)['relative_energy_bias'],bound+1e-14)
    def test_free_energy_decreases(self):
        b=1.; l=generator(b); vec=INITIAL.reshape(-1,order='F'); h=pair_h()
        vals=[]
        for t in (0.,.01,.1,1.,4.):
            r=(expm(t*l)@vec).reshape(9,9,order='F')
            vals.append(tr(r@h)-entropy(r)/b)
        self.assertTrue(np.all(np.diff(vals)<1e-12))
    def test_no_energy_origin_selection(self):
        shift=2.0
        r=state(1.,energies=MASS+shift)
        self.assertLess(np.linalg.norm(r-state(1.)),1e-14)
        self.assertAlmostEqual(stats(1.,energies=MASS+shift)['energy_mean']-stats(1.)['energy_mean'],2*shift,12)

def results():
    vh=np.mean(MASS*MASS)-np.mean(MASS)**2
    return dict(model='CE-TH1',date='2026-09-19',source_main='cbbc1cd930eb6c7ff42e0437aca261334f2a271d',
        scope='Conditional finite-dimensional thermal collision model; no cosmological or flavour fit.',
        fixed_inputs=dict(alpha_s=ALPHA_S,p=P,delta=DELTA,masses=MASS.tolist()),
        diagnostics=[stats(b) for b in (0.,.1,1.,5.,10.)],
        energy_collision=collision(),evolution=[evolution(b) for b in (0.,1.,5.)],
        pressure=pressure_demo(),
        high_temperature_relative_bias_per_beta=(1-DELTA)*1.5*vh/(2*np.mean(MASS)),
        global_relative_bias_bound_per_beta=(1-DELTA)*(MASS[-1]-MASS[0])**2/(2*MASS[0]),
        approximation_limit=[dict(dt=d,error=finite_collision_error(1.,d)) for d in (.02,.01,.005)],
        tests=24,
        unresolved=['Initial delta and alpha_s matching are not predicted.',
                    'Thermal beta and diagnostic masses are supplied, not fit.',
                    'Bath supply, Markov limit and collective coupling are assumptions.',
                    'No PMNS/Yukawa, abundance, vacuum amplitude or CMB likelihood calculation.',
                    'Finite-temperature extension is a new model assumption, not a forced correction to all CE physics.'])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='results.json')
    args=parser.parse_args()
    res=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not res.wasSuccessful(): raise SystemExit(1)
    Path(args.output).write_text(json.dumps(results(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
