"""CE-GF3: quadratic Maxwell plus induced scalar metric audit."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from ce_projector_one_loop import coefficient


def match_error(actual,expected):
    return min(max(abs(a-b) for a,b in zip(actual,p)) for p in itertools.permutations(expected))


def main():
    c_loop=coefficient(1.,5.);kappa=1.
    rows=[]
    for label,c in [('loop',c_loop),('half',.5),('threshold',1.),('twice',2.)]:
        hessian=np.array([c,c+kappa,c+kappa,c+kappa])
        assert np.all(hessian>0)
        for k in (.01,.05,.1):
            # p0=a(t) cos(kz), pz=b(t) sin(kz); rows are (a,b,adot,bdot).
            flow=np.array([[0,0,1,0],[0,0,0,1],
                           [(kappa-c)*k*k/c,0,0,kappa*k/c],
                           [0,-c*k*k/(c+kappa),-kappa*k/(c+kappa),0]],float)
            actual=np.linalg.eigvals(flow)
            speed2=(c-kappa)/(c+kappa)
            extra=np.sqrt(complex(-speed2))*k
            expected=[1j*k,-1j*k,extra,-extra]
            error=match_error(actual,expected)
            assert error<1e-10,(label,k,error)
            if c<kappa:assert max(actual.real)>0
            # Direct determinant coefficients versus factorized dispersion.
            polynomial=[c*(c+kappa),-2*c*c*k*k,c*(c-kappa)*k**4]
            factored=[c*(c+kappa),-c*((c+kappa)+(c-kappa))*k*k,c*(c-kappa)*k**4]
            assert np.max(abs(np.array(polynomial)-factored))<1e-12
            rows.append({'case':label,'C':c,'k':k,'velocity_hessian':hessian.tolist(),
                         'longitudinal_extra_speed_squared':speed2,
                         'growth_rate':float(max(actual.real)),
                         'eigenvalue_match_error':float(error),
                         'static_p0_gradient_energy_coefficient':(c-kappa)/2})
    overlaps=[]
    for delta in (0.,.1,.5,1.):
        alpha=1j*delta/np.sqrt(2)
        v=np.array([np.exp(-abs(alpha)**2/2)*alpha**n/math.sqrt(math.factorial(n)) for n in range(33)])
        overlap=abs(v[0])**2;expected=math.exp(-delta**2/2)
        assert abs(overlap-expected)<1e-12
        assert abs(np.vdot(v,v)-1)<1e-12
        overlaps.append({'delta_p_norm':delta,'state_overlap_squared':float(overlap),
                         'projector_HS_distance_squared':2*(1-expected)})
    backgrounds=[]
    for ell,g in [(1.,1.),(100.,.01)]:
        kap=1/(g*g*ell*ell)
        assert abs(kap-1)<1e-12
        backgrounds.append({'ell':ell,'g_A':g,'kappa':kap,'background_gradient':1/ell})
    here=Path(__file__).resolve()
    out={'candidate':'CE-GF3','C_loop':c_loop,'backgrounds':backgrounds,'modes':rows,
         'state_overlaps':overlaps,'pure_Maxwell_velocity_rank':3,
         'combined_p_velocity_rank':4,
         'source_sha256':{f:hashlib.sha256(here.with_name(f).read_bytes()).hexdigest()
                           for f in [here.name,'ce_projector_one_loop.py']},
         'limits':['Quadratic two-derivative model on supplied flat spacetime',
                   'Uniform-gap relative loop coefficient, not the rising-spectrum theory',
                   'No complete determinant, gravity or observational parameter claim']}
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'C_loop':c_loop,'loop_results':rows[:3],
                     'maximum_eigenvalue_error':max(r['eigenvalue_match_error'] for r in rows),
                     'overlaps':overlaps},indent=2))


if __name__=='__main__':main()
