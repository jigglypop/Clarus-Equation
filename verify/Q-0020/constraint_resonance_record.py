"""실제 장치 포함 제약의 생성자로 유한 분해능 기록을 구성한다.

F_tot의 고정 총에너지 주기 정의역·시계·포인터·결합·준비는 공급한다.
K의 계수 pi/(2gamma)는 외부 시간이 아닌 무차원 결합면적이다.
원래 고정 b 제약 F_s나 0D 시간 발생을 유도하지 않는다.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

import regge_record_constraint as prior
import relational_split_clock as clock

HERE=Path(__file__).resolve().parent
SX=np.array([[0.,1.],[1.,0.]])
SZ=np.diag([1.,-1.])


@lru_cache(maxsize=12)
def source(size,beta):
    return prior.fiber(size,1.,beta,"length","constraint")


def response(eigenvalues,gamma,fraction=1.):
    if not np.isfinite(gamma) or gamma<=0 or not np.isfinite(fraction):
        raise ValueError("양의 유한 결합과 유한 결합 비율이 필요하다")
    lam=np.asarray(eigenvalues,dtype=float)
    if not np.all(np.isfinite(lam)):
        raise ValueError("제약 고윳값은 유한해야 한다")
    radius=np.hypot(lam,gamma)
    return (gamma/radius*np.sin(math.pi*fraction*radius/(2*gamma)))**2


class Detector:
    def __init__(self,size=17,beta=5.,twist=0.,ratio=.25):
        if isinstance(size,bool) or not isinstance(size,int) or size<3 or size%2==0:
            raise ValueError("푸리에 절단은 3 이상의 홀수여야 한다")
        if not np.isfinite(ratio) or ratio<=0 or not np.isfinite(twist) or not np.isfinite(beta):
            raise ValueError("분해능·경계 위상·작용 계수를 확인해야 한다")
        data=source(size,float(beta))
        self.n=size;self.beta=beta;self.twist=twist
        self.length=float(prior.geometry.full.limit(1.))
        self.gap=2*math.pi/self.length
        self.gamma=ratio*self.gap
        self.frequencies=np.rint(np.fft.fftfreq(size)*size).astype(int)
        self.eigenvalues=(2*math.pi*self.frequencies+twist)/self.length
        unit=(np.arange(size)+.5)/size
        dressing=data["j"]*math.sqrt(size)
        self.modes=dressing[:,None]*np.exp(1j*unit[:,None]*
            (2*math.pi*self.frequencies[None,:]+twist))/math.sqrt(size)
        self.system_energy=data["levels"]

    def generator(self,state):
        lam=self.eigenvalues[None,:]
        first=lam*state[:,:,0]+self.gamma*state[:,:,1]
        second=self.gamma*state[:,:,0]-lam*state[:,:,1]
        return math.pi/(2*self.gamma)*np.stack((first,second),axis=-1)

    def rotate_fraction(self,fraction,state):
        radius=np.hypot(self.eigenvalues,self.gamma)
        angle=math.pi*fraction*radius/(2*self.gamma)
        return np.cos(angle)[None,:,None]*state-1j*(
            2*self.gamma*np.sin(angle)/(math.pi*radius))[None,:,None]*self.generator(state)

    def rotate(self,x,state):
        return self.rotate_fraction(float(clock.autonomous.profile(x)[0]),state)

    def coefficients(self):
        c=np.zeros(self.n,dtype=complex)
        c[0]=math.sqrt(.3);c[1]=math.sqrt(.4);c[-1]=1j*math.sqrt(.3)
        return c

    def physical(self,state):
        return np.einsum("ij,tjr->tir",self.modes,state)


def spectral_check(model):
    gamma=model.gamma;lam=model.eigenvalues
    c=model.modes.conj().T@np.ones(model.n)/math.sqrt(model.n)
    c/=np.linalg.norm(c)
    weights=abs(c)**2
    r=response(lam,gamma)
    pzero=float(weights[0]);click=float(weights@r)
    leakage=float(weights[1:]@r[1:]/click) if click>0 else None
    eps=gamma**2/(model.gap**2+gamma**2)
    prior_bound=eps*(1-pzero)/(pzero+eps*(1-pzero))
    return {"size":model.n,"beta":model.beta,"gamma_over_gap":gamma/model.gap,
            "zero_response":float(r[0]),"maximum_off_response":float(max(r[1:])),
            "off_envelope":eps,"actual_prior_zero":pzero,"click_probability":click,
            "posterior_off":leakage,"posterior_bound":prior_bound}


def matrix_check(model):
    f=(model.modes*model.eigenvalues)@model.modes.conj().T
    d=np.kron(f,SZ)+model.gamma*np.kron(np.eye(model.n),SX)
    direct=expm(-1j*math.pi*d/(2*model.gamma))
    c=model.coefficients();initial=np.zeros((1,model.n,2),complex);initial[0,:,0]=c
    output=model.rotate_fraction(1.,initial)
    physical=model.physical(initial).reshape(-1)
    expected=model.physical(output).reshape(-1)
    selected=output[0,:,1]
    before=abs(model.physical(initial))**2;after=abs(model.physical(output))**2
    delta=after-before
    system=float(np.sum(delta*model.system_energy[None,:,None]))
    battery=float(np.sum(delta*(2.8-model.system_energy)[None,:,None]))
    full_f=np.kron(f,np.eye(2))
    return {
        "size":model.n,"direct_exponential_error":float(np.linalg.norm(direct@physical-expected)),
        "constraint_commutator":float(np.linalg.norm(direct@full_f-full_f@direct)),
        "constraint_distribution_error":float(np.max(abs(np.sum(abs(output)**2,axis=2)-np.sum(abs(initial)**2,axis=2)))),
        "pointer_response_error":abs(float(np.linalg.norm(selected)**2)-float(abs(c)**2@response(model.eigenvalues,model.gamma))),
        "system_energy_change":system,"battery_energy_change":battery,"energy_balance_error":abs(system+battery),
        "minimum_battery_energy":float(2.8-max(model.system_energy)),
        "norm_error":abs(float(np.linalg.norm(output)**2)-1),
    }


def clock_audit(model,x=.43,step=2e-4):
    totals=np.linspace(2.2,3.8,5);gap=.4;mass=1.3
    amplitude=np.sin(math.pi*np.arange(1,6)/6);amplitude/=np.linalg.norm(amplitude)
    a=np.zeros((len(totals),model.n,2),complex)
    a[:,:,0]=amplitude[:,None]*model.coefficients()[None,:]
    kinetic=gap+totals[-1]-totals
    p=np.sqrt(2*mass*kinetic);velocity=p/mass
    def wave(z):
        free=a*(np.exp(1j*p*z)/np.sqrt(velocity))[:,None,None]
        return model.rotate(z,free)
    value,first,second=clock.differences(wave,x,step)
    _,first2,second2=clock.differences(wave,x,step/2)
    first=(16*first2-first)/15;second=(16*second2-second)/15
    fraction,slope=clock.autonomous.profile(x)
    curvature=60*x*(1-x)*(1-2*x) if 0<x<1 else 0.
    kval=model.generator(value);square=model.generator(kval)
    pi=-1j*first+slope*kval
    pi2=-second-2j*slope*model.generator(first)-1j*curvature*kval+slope*slope*square
    residual=pi2/(2*mass)-kinetic[:,None,None]*value
    currents=[float(np.vdot(value[:,:,r],pi[:,:,r]).real/mass) for r in (0,1)]
    expected=float(abs(model.coefficients())**2@response(model.eigenvalues,model.gamma,float(fraction)))
    position=float(np.sum(abs(value[:,:,1])**2)/np.linalg.norm(value)**2)
    return {"x":x,"gamma_over_gap":model.gamma/model.gap,
            "constraint_residual":float(np.linalg.norm(residual)/np.linalg.norm(value)),
            "flux_error":abs(sum(currents)-1),"click_flux_error":abs(currents[1]-expected),
            "position_flux_error":abs(position-expected),
            "omitted_square_residual":float(np.linalg.norm(residual-slope*slope*square/(2*mass))/np.linalg.norm(value))}


def differential_constraint(mode,twist,beta=5.,fraction=.43):
    geometry=prior.geometry;constraint=prior.constraint
    lengths=constraint.geometry_lengths(1.)
    lo,hi=constraint.fine_interval(lengths);width=hi-lo;upper=geometry.full.limit(1.)
    e=lo+fraction*width;t=2.8;b=t-1-e/upper
    def packet(total):
        return math.sin(math.pi*(total-2.2)/1.6) if 2.2<total<3.8 else 0.
    def wave(edge,battery):
        return constraint.wave(lengths,edge,"length",mode=mode,twist=twist,beta=beta)*packet(1+edge/upper+battery)
    step=2e-5
    def derivative(axis):
        def central(h):
            return ((wave(e+h,b)-wave(e-h,b))/(2*h) if axis==0 else
                    (wave(e,b+h)-wave(e,b-h))/(2*h))
        return (4*central(step/2)-central(step))/3
    value=wave(e,b);_,gradient=constraint.phase_data(lengths,e)
    system=-1j*derivative(0)+beta*gradient*value
    compensation=1j*derivative(1)/upper
    eigenvalue=(2*math.pi*mode+twist)/width
    return {"mode":mode,"twist":twist,"beta":beta,"fraction":fraction,
            "eigenvalue":eigenvalue,"total_eigen_residual":abs(system+compensation-eigenvalue*value),
            "system_only_residual":abs(system-eigenvalue*value),"minimum_battery":b}


def preparation_control(model):
    r=float(response(model.gap,model.gamma))
    rows=[]
    for gap in (1.,1e-4,1e-12):
        weights=np.array([1/math.sqrt(gap+1),1/math.sqrt(gap)])
        # 두 원자: 아래 에너지에는 비영모드, 위 에너지에는 영모드.
        atomic=float(weights@np.array([r,1.])/sum(weights))
        # 연속 균등 준비는 끝점 특이성이 적분 가능하다.
        denom=2*(math.sqrt(gap+1)-math.sqrt(gap))
        upper=2*(math.sqrt(gap+.5)-math.sqrt(gap))
        continuous=r+(1-r)*upper/denom
        y0=math.sqrt(gap);ym=math.sqrt(gap+.5);y1=math.sqrt(gap+1)
        direct=(quad(lambda y:2.,y0,ym)[0]+quad(lambda y:2*r,ym,y1)[0])/denom
        rows.append({"gap":gap,"flux":(1+r)/2,"atomic_position":atomic,
                     "continuous_position":continuous,"continuous_quad_error":abs(continuous-direct)})
    factorized=[]
    modes=model.coefficients();expected=float(abs(modes)**2@response(model.eigenvalues,model.gamma))
    for gap in (1.,1e-4,1e-16):
        w=np.sqrt(gap/(gap+np.linspace(1.,0.,5)))
        weights=np.array([1.,4.,6.,4.,1.])/16
        actual=float(np.sum(w*weights*expected)/np.sum(w*weights))
        factorized.append({"gap":gap,"record":actual,"flux":expected,"difference":abs(actual-expected)})
    return {"correlated":rows,"factorized":factorized,
            "continuous_limit":r+(1-r)/math.sqrt(2),"atomic_limit":1.}


def twist_control():
    rows=[]
    base=Detector(ratio=.1)
    for relative in (1.,.1,.01,.001):
        twist=relative*base.gamma*base.length
        lam=twist/base.length
        rows.append({"twist":twist,"minimum_abs_eigenvalue":lam,
                     "lambda_over_gamma":relative,"exact_zero_mode":False,
                     "click":float(response(lam,base.gamma))})
    # 간격 없이 0에 축적하는 비영 고윳값의 반례.
    accumulation=[{"lambda_over_gamma":r,"click":float(response(r,1.))}
                  for r in (1.,.1,.01,.001,1e-5)]
    return {"twisted":rows,"accumulation":accumulation}


def branch_budget(model):
    # 실제 작용 위상과 다른 균등 계 준비를 같은 총에너지 섬유에 넣는다.
    c=model.modes.conj().T@np.ones(model.n)/math.sqrt(model.n)
    c/=np.linalg.norm(c)
    initial=np.zeros((1,model.n,2),complex);initial[0,:,0]=c
    output=model.physical(model.rotate_fraction(1.,initial))[0]
    before=model.physical(initial)[0]
    total=2.8;rows=[]
    for outcome in (0,1):
        density=abs(output[:,outcome])**2
        probability=float(sum(density))
        system=float(density@model.system_energy)
        battery=float(density@(total-model.system_energy))
        rows.append({"record":outcome,"interpretation":"미검출" if outcome==0 else "검출",
                     "probability":probability,"system_energy_unnormalized":system,
                     "battery_energy_unnormalized":battery,"total_energy_unnormalized":system+battery,
                     "conditional_total_energy":(system+battery)/probability if probability>0 else None})
    before_density=np.sum(abs(before)**2,axis=1)
    return {"size":model.n,"beta":model.beta,"gamma_over_gap":model.gamma/model.gap,"branches":rows,
            "probability_sum_error":abs(sum(r["probability"] for r in rows)-1),
            "total_energy_sum_error":abs(sum(r["total_energy_unnormalized"] for r in rows)-total),
            "initial_system_energy":float(before_density@model.system_energy),
            "initial_battery_energy":float(before_density@(total-model.system_energy)),
            "nonselected_branch_discarded":False}


def run():
    matrix=[matrix_check(Detector(n)) for n in (17,33,65)]
    model=Detector()
    paths=[Path(__file__),HERE/"regge_record_constraint.py",HERE/"regge_quantum_constraint_transfer.py",
           HERE/"relational_split_clock.py",HERE/"autonomous_split_clock.py"]
    return {
        "model":"장치 포함 제약 생성자의 공명 기록과 스펙트럼 간격",
        "dependencies":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "spectra":[spectral_check(Detector(n,beta=beta,ratio=ratio)) for n in (17,33,65)
                   for beta in (0.,5.) for ratio in (.5,.3,.25,.17,.1,.07,.05,.01037,.01)],
        "matrix":matrix,"clock":[clock_audit(Detector(ratio=ratio),x=x)
                                for ratio in (.5,.25) for x in (.2,.43,.8)],
        "differential":[differential_constraint(mode,twist,beta,fraction)
                        for mode in (0,1,-2) for twist in (0.,.17)
                        for beta,fraction in ((0.,.31),(5.,.67))],
        "preparation":preparation_control(model),"twist":twist_control(),
        "branch_budget":branch_budget(Detector(65,ratio=.01037)),
        "scope":{"periodic_fixed_total_energy_domain_supplied":True,
                 "first_order_constraint_generator_used":True,
                 "constraint_and_energy_strong_commutation_assumed_in_continuum":True,
                 "original_fixed_b_constraint_preserved":False,"physical_clock_origin_derived":False,
                 "physical_duration_or_control_strength_cost_derived":False,
                 "common_metric_forces_or_cosmology_derived":False,
                 "conditional_click_is_not_exact_projection_without_gap_and_prior":True},
    }


if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=True,indent=2,allow_nan=False))
