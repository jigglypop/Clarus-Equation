"""CE-GR2: a trace-class internal tower and the cost of transporting it.

No observation fitting. Finite proper time is distinct from a UV-complete action.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from ce_common_isotropic_frame import block_diag, cycle_matrix, weak_frame
from ce_isometric_color_frame import KS, V0
from ce_projector_one_loop import coefficient

A, B, OMEGA = 1., 5., 2.


def heat(s):
    denom = -math.expm1(-OMEGA*s)
    return 18*math.exp(-A*s) + math.exp(-B*s)*(3/denom**5+60/denom**3-18)


def finite_heat(s, cutoff):
    return 18*math.exp(-A*s)+math.exp(-B*s)*math.fsum(
        (3*math.comb(n+4,4)+60*math.comb(n+2,2)-(18 if n==0 else 0))
        *math.exp(-OMEGA*s*n) for n in range(cutoff+1))


def tail(s, cutoff):
    r = math.exp(-OMEGA*s)
    total = 0.
    for d, multiplier in [(5,3),(3,60)]:
        n=cutoff+1
        prefix=0.
        while r*(n+d)/(n+1)>=1:
            prefix+=math.comb(n+d-1,d-1)*r**n
            n+=1
        first = math.comb(n+d-1,d-1)*r**n
        ratio = r*(n+d)/(n+1)
        assert ratio<1
        total += multiplier*(prefix+first/(1-ratio))
    # Frozen cases whose powers underflow have a true tail below the least
    # positive float. Report that positive floor instead of an exact zero.
    return max(float(np.nextafter(0.,1.)), math.exp(-B*s)*total)


def normal_split(theta):
    # At q=0 occupation 0 or 1 per oscillator contains the exact first derivatives.
    vl=np.zeros((4,1),complex);vl[0]=1
    dl=[np.zeros_like(vl) for _ in range(3)]
    dl[0][2]=1/np.sqrt(2);dl[1][2]=1j/np.sqrt(2);dl[2][1]=1/np.sqrt(2)
    vw,dw=weak_frame([0.,0.,0.])
    vc=V0;dc=[1j*k@vc for k in KS]
    vg=block_diag([vl,vw,vc])
    dg=[block_diag([dl[a],dw[a],dc[a]]) for a in range(3)]
    values,vectors=np.linalg.eigh(cycle_matrix(theta))
    vf=np.zeros((24,3),complex);df=[np.zeros_like(vf) for _ in range(3)]
    for j,kappa in enumerate(values):
        vf[8*j,j]=1
        for a,index in enumerate([4,2,1]):df[a][8*j+index,j]=np.sqrt(kappa)
    vf=vf@vectors.conj().T;df=[d@vectors.conj().T for d in df]
    v=np.kron(vg,vf)
    derivatives=[np.kron(dg[a],vf)+np.kron(vg,df[a]) for a in range(3)]
    ng=np.array([0,1,1,2]+[0]*20)
    nf=np.tile([sum(t) for t in itertools.product([0,1],repeat=3)],3)
    number=(ng[:,None]+nf[None,:]).reshape(-1)
    normals=[d-v@(v.conj().T@d) for d in derivatives]
    parts=[]
    for level in (0,1):
        part=[d[number==level,:] for d in normals]
        parts.append(np.array([[np.vdot(x,y).real for y in part] for x in part]))
    errors={'frame_isometry':float(np.linalg.norm(v.conj().T@v-np.eye(18))),
            'number_annihilates_frame':float(np.linalg.norm(number[:,None]*v)),
            'normal_above_level_one':float(max(np.linalg.norm(d[number>1,:]) for d in normals)),
            'level_zero_metric':float(np.linalg.norm(parts[0]-7.5*np.eye(3))),
            'level_one_metric':float(np.linalg.norm(parts[1]-19.5*np.eye(3))),
            'total_metric':float(np.linalg.norm(sum(parts)-27*np.eye(3)))}
    assert max(errors.values())<1e-10,errors
    return {'theta':theta,'errors':errors,'kappa_min':float(values.min()),
            'light_heavy_kinetic_coefficient':7.5*coefficient(A,B)+19.5*coefficient(A,B+OMEGA)}


def main():
    traces=[]
    for s in (.1,.5,1.,2.):
        exact=heat(s)
        for cutoff in (16,64,256,512):
            partial=finite_heat(s,cutoff);bound=tail(s,cutoff)
            slack=1e-12*exact
            assert -slack<=exact-partial<=bound+slack
            if cutoff==512:assert abs(partial-exact)/exact<1e-10
            traces.append({'s':s,'cutoff':cutoff,'exact':exact,'partial':partial,
                           'omitted_tail_bound':bound,'relative_error':abs(partial-exact)/exact})
    metrics=[normal_split(theta) for theta in (0.,.7,math.pi)]
    # A single Q-color, single flavor channel, single oscillator chain suffices
    # for a positive divergent lower subseries of the unregulated kinetic sum.
    x,w=np.polynomial.legendre.leggauss(64);x=(x+1)/2;w=w/2
    terms=[];errors=[]
    for n in range(1,1025):
        a=B+OMEGA*n;b=a+OMEGA
        c=coefficient(a,b)
        quad=OMEGA**2/(16*math.pi**2)*np.dot(w,x*(1-x)/(x*a+(1-x)*b))
        errors.append(abs(c-quad)/c)
        terms.append((n+1)*c)  # per unit flavor kappa
    assert max(errors)<1e-10
    limit=OMEGA/(96*math.pi**2)
    chains=[{'cutoff':cutoff,'positive_partial_sum_per_kappa':math.fsum(terms[:cutoff]),
             'last_term_per_kappa':terms[cutoff-1],
             'last_term_over_limit':terms[cutoff-1]/limit} for cutoff in (16,64,256,1024)]
    assert all(terms[n]>0 for n in range(len(terms)))
    assert .99<terms[-1]/limit<1
    uv=[{'s':s,'scaled_heat_to_limit_one':heat(s)*(OMEGA*s)**5/3}
        for s in (.1,.01,.001,.0001)]
    here=Path(__file__).resolve()
    sources=[here.name,'ce_common_isotropic_frame.py','ce_isometric_color_frame.py','ce_projector_one_loop.py']
    out={'candidate':'CE-GR2','inputs':{'a':A,'b':B,'omega':OMEGA},
         'source_sha256':{f:hashlib.sha256(here.with_name(f).read_bytes()).hexdigest() for f in sources},
         'heat_traces':traces,'metric_splits':metrics,'uv_scaling':uv,'heavy_chain':chains,
         'chain_term_limit_per_kappa':limit,'pair_coefficient_max_relative_error':max(errors),
         'status':'finite internal heat trace at positive proper time; full UV action not finite',
         'limitations':['Supplied increasing spectrum and full unitary extension',
                        'Additional heavy-heavy kinetic sum diverges without a UV prescription',
                        'No physical Einstein limit or gauge vector dynamics established']}
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'candidate':out['candidate'],
                     'pair_error':max(errors),'metrics':metrics,'uv_scaling':uv,'heavy_chain':chains},indent=2))


if __name__=='__main__':main()
