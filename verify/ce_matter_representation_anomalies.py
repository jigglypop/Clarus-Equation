"""CE-MR1: conditional matter representations and anomaly cancellation.

The Weyl content is supplied, not derived from the scalar frame action.
"""
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from ce_common_isotropic_frame import block_diag, SIGMA


def sums(charges,nu=0):
    q,u,d,l,e=charges
    return {'SU3_cubic':2-1-1,'SU3_squared_U1':2*q+u+d,
            'SU2_squared_U1':3*q+l,
            'gravity_squared_U1':6*q+3*u+3*d+2*l+e+nu,
            'U1_cubic':6*q**3+3*u**3+3*d**3+2*l**3+e**3+nu**3,
            'SU2_doublet_parity':(3+1)%2}


def color_generators():
    result=[]
    for a,b in [(0,1),(0,2),(1,2)]:
        x=np.zeros((3,3),complex);x[a,b]=x[b,a]=.5
        y=np.zeros((3,3),complex);y[a,b]=-.5j;y[b,a]=.5j
        result.extend([x,y])
    result.extend([np.diag([1.,-1.,0.])/2,np.diag([1.,1.,-2.])/(2*np.sqrt(3))])
    return result


def matrices(charges):
    reps=[block_diag([y*np.eye(d) for y,d in zip(charges,[6,3,3,2,1])])]
    for sigma in SIGMA:
        t=sigma/2
        reps.append(block_diag([np.kron(np.eye(3),t),np.zeros((3,3)),
                                np.zeros((3,3)),t,np.zeros((1,1))]))
    for t in color_generators():
        reps.append(block_diag([np.kron(t,np.eye(2)),-t.T,-t.T,
                                np.zeros((2,2)),np.zeros((1,1))]))
    return reps


def matrix_anomalies(charges):
    reps=matrices(charges)
    cubic=max(abs(np.trace(a@(b@c+c@b))) for a,b,c in itertools.product(reps,repeat=3))
    linear=max(abs(np.trace(a)) for a in reps)
    hermitian=max(np.linalg.norm(a-a.conj().T) for a in reps)
    cross=max(np.linalg.norm(a@b-b@a) for a in reps[1:4] for b in reps[4:])
    assert hermitian<1e-10 and cross<1e-10
    return {'max_symmetric_triple_trace':float(cubic),'max_linear_trace':float(linear),
            'hermiticity_error':float(hermitian),'commuting_factors_error':float(cross)}


def multiply(p,q):
    result={}
    for (a,b),c in p.items():
        for (d,e),f in q.items():result[a+d,b+e]=result.get((a+d,b+e),0)+c*f
    return {key:value for key,value in result.items() if value}


def cubic_polynomial():
    result={}
    for weight,(a,b) in zip([6,3,3,2,1],[(1,0),(0,1),(-2,-1),(-3,0),(6,0)]):
        linear={(1,0):a,(0,1):b}
        cube=multiply(multiply(linear,linear),linear)
        for key,value in cube.items():result[key]=result.get(key,0)+weight*value
    result={key:value for key,value in result.items() if value}
    factored=multiply(multiply({(1,0):-18},{(1,0):4,(0,1):1}),{(1,0):-2,(0,1):1})
    assert result==factored
    return {f'q^{i} u^{j}':value for (i,j),value in result.items()}


def kernel():
    # Centers: U(1) parameter k/6, SU(2) exponent s, SU(3) exponent r.
    charges=[1,-4,2,-3,6];doublets=[1,0,0,1,0];triality=[1,-1,-1,0,0]
    result=[]
    for k,s,r in itertools.product(range(6),range(2),range(3)):
        phases=[Fraction(y*k,6)+Fraction(d*s,2)+Fraction(t*r,3)
                for y,d,t in zip(charges,doublets,triality)]
        if all(p.denominator==1 for p in phases):result.append([k,s,r])
    assert result==[[k,k%2,k%3] for k in range(6)]
    return result


def main():
    standard=(1,-4,2,-3,6);swapped=(1,2,-4,-3,6);bad=(1,-4,2,-3,5)
    assert all(v==0 for v in sums(standard).values())
    assert all(v==0 for v in sums(swapped).values())
    for t in range(-4,5):assert all(v==0 for v in sums((0,t,-t,0,0)).values())
    good=matrix_anomalies(standard);swap=matrix_anomalies(swapped);negative=matrix_anomalies(bad)
    assert max(good.values())<1e-10 and max(swap.values())<1e-10
    assert sums(bad)['U1_cubic']==-91 and negative['max_symmetric_triple_trace']==182
    y=[1,-4,2,-3,6,0];bl=[1,-1,-1,-3,3,3];weights=[6,3,3,2,1,1]
    mixed=[sum(w*math.comb(3,p)*a**p*b**(3-p) for w,a,b in zip(weights,y,bl)) for p in range(4)]
    assert mixed==[0,0,0,0]
    extensions=[]
    for alpha,beta in itertools.product(range(-2,3),repeat=2):
        charges=[alpha*a+beta*b for a,b in zip(y,bl)]
        anomalies=sums(charges[:5],charges[5]);assert all(v==0 for v in anomalies.values())
        extensions.append({'alpha':alpha,'beta':beta,'charges':charges})
    # Higgs L^3 W: singlet hypercharge sums for Q H u^c, Q H* d^c, l H* e^c.
    yukawa=[standard[0]+3+standard[1],standard[0]-3+standard[2],standard[3]-3+standard[4]]
    assert yukawa==[0,0,0]
    here=Path(__file__).resolve()
    out={'candidate':'CE-MR1','standard_anomalies':sums(standard),
         'matrix_standard':good,'matrix_swapped':swap,'negative_control':negative,
         'cubic_after_linear_constraints':cubic_polynomial(),'center_kernel':kernel(),
         'right_neutrino_mixed_cubic_coefficients':mixed,'right_neutrino_families':extensions,
         'Yukawa_charge_sums':yukawa,
         'three_generations':{'internal_Weyl_components':45,'SU2_doublets':12},
         'source_sha256':{name:hashlib.sha256(here.with_name(name).read_bytes()).hexdigest()
                           for name in [here.name,'ce_common_isotropic_frame.py','ce_isometric_color_frame.py']},
         'limits':['Matter content, chirality, spin geometry and Higgs are supplied',
                   'Matter kernel does not fix the full theory global gauge group',
                   'No independent gauge vectors, Yukawa values or Einstein dynamics derived']}
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:out[key] for key in ['standard_anomalies','matrix_standard','negative_control',
          'cubic_after_linear_constraints','center_kernel','right_neutrino_mixed_cubic_coefficients']},indent=2))


if __name__=='__main__':main()
