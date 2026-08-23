# Concrete P0 counterexample instance: d*=3, eta=1e-3 draw with min_d s_d > tau_class
import numpy as np, json
from math import comb
D_SET=[1,2,3]; N=14; MS=900030
def mex(d): return [(a,b) for s in range(d+1) for a in range(s+1) for b in [s-a]]
def phi(Z,d): return np.stack([Z[:,0]**a*Z[:,1]**b for (a,b) in mex(d)],axis=1)
def loo_mean(P,Y):
    G=P.T@P; C=np.linalg.lstsq(P,Y,rcond=None)[0]
    h=np.einsum('ij,jk,ik->i',P,np.linalg.inv(G),P)
    E=(Y-P@C)/(1.0-h)[:,None]
    return float(np.mean(np.linalg.norm(E,axis=1))), h
rng=np.random.default_rng(MS+1000*3+1000)  # cell_sim(3, 1e-3) stream
eta=1e-3; tau=8*eta
for t in range(200):
    Z=rng.standard_normal((N,2)); zq=rng.standard_normal((1,2))
    Cs=rng.uniform(-1,1,(comb(5,2),6))
    Y=phi(Z,3)@Cs+eta*rng.standard_normal((N,6))
    sv={}; hmax={}
    for d in D_SET:
        sv[d],h=loo_mean(phi(Z,d),Y); hmax[d]=float(h.max())
    if min(sv.values())>tau:
        ce={'rng_stream':'default_rng(900030+3000+1000), draw_index t=%d'%t,
            'eta':eta,'tau_class':tau,
            's_d':{str(d):sv[d] for d in D_SET},
            'min_s_over_tau':min(sv.values())/tau,
            'hmax_d3':hmax[3],
            'Z':Z.tolist(),'Cstar':Cs.tolist()}
        print(json.dumps(ce,indent=2)); break
