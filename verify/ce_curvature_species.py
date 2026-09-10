"""CE-GR1: regulated curvature response, not a derivation of dynamical gravity."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def integral(c,L,n,method):
    x,w=np.polynomial.legendre.leggauss(n)
    u=(x+1)/2;w=w/2
    if method==0:
        return float(L*np.dot(w,np.exp(-c/(L*u))))
    s=1/L+u/(c*(1-u))
    return float(np.dot(w,np.exp(-c*s)/s**2/(c*(1-u)**2)))


def closed(c,L):
    z=c/L
    # E1 series, convergent for positive z; frozen cases have z <= 5.
    terms=[];term=1.
    for k in range(1,150):
        term*=-z/k;terms.append(term/k)
    e1=-0.5772156649015328606-math.log(z)-math.fsum(terms)
    return L*math.exp(-z)-c*e1


def main():
    rows=[];errors=[];table=[]
    for L in (1.,10.,100.,1000.):
        vals=[]
        for c in (1.,5.):
            results=[integral(c,L,n,method) for method in (0,1) for n in (256,512)]
            expected=closed(c,L)
            err=max(abs(v-expected)/expected for v in results)
            assert expected>0 and err<1e-8,(c,L,err)
            errors.append(err);vals.append(expected)
            rows.append({'mass_squared':c,'cutoff_squared':L,'closed':expected,
                         'quadratures':results,'max_relative_error':err})
        ia,ib=vals
        for N in (18,36,180,1800):
            for xi in (0.,1/6,1/3):
                weight=18*ia+(N-18)*ib
                mpl2=2*(1/6-xi)*weight/(16*math.pi**2)
                relative=2*(1/6-xi)*18*(ia-ib)/(16*math.pi**2)
                assert (mpl2>0 if xi==0 else mpl2==0 if xi==1/6 else mpl2<0)
                table.append({'N':N,'cutoff_squared':L,'xi':xi,
                              'regulated_EH_mass_squared':mpl2,
                              'all_heavy_subtracted_EH_mass_squared':relative})
    # Exact isospectrality under constant orientation, in a finite witness.
    p=np.diag([1.,1.,0.,0.]);x=5*np.eye(4)-4*p
    angle=.7;u=np.eye(4);u[0,0]=u[2,2]=math.cos(angle)
    u[0,2]=-math.sin(angle);u[2,0]=math.sin(angle)
    rotated=u@x@u.T
    eigerr=float(np.max(abs(np.linalg.eigvalsh(rotated)-np.linalg.eigvalsh(x))))
    assert eigerr<1e-12
    here=Path(__file__).resolve()
    out={'candidate':'CE-GR1','script_sha256':hashlib.sha256(here.read_bytes()).hexdigest(),
         'integrals':rows,'coefficients':table,'max_quadrature_relative_error':max(errors),
         'constant_orientation_eigenvalue_error':eigerr,
         'log_cutoff_derivative_at_1000':1000*(math.exp(-1/1000)-math.exp(-5/1000)),
         'log_cutoff_derivative_limit':4,
         'limits':['Supplied metric and curvature coupling; no Einstein limit established',
                   'Infinite equal heavy multiplicity diverges at finite gap and cutoff',
                   'Same-rank orientation subtraction cancels pure curvature response']}
    here.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:out[k] for k in ('candidate','max_quadrature_relative_error',
          'constant_orientation_eigenvalue_error','log_cutoff_derivative_at_1000')},indent=2))
    for row in table:
        if row['cutoff_squared']==100 and row['xi']==0:print(json.dumps(row))


if __name__=='__main__':main()
