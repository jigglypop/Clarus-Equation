# R-D confirmation sim: N=24 + studentized PRESS + tau=max(1e-8,8eta) + parsimony floor 1e-8
# New master seed 910024; 2000 draws/cell (spec >= 400), witness 2000, shuffle 1000/eta.
import numpy as np, json
from math import comb
RHO=0.5; FLOOR=1e-8; D_SET=[1,2,3]; N=24; NDRAW=2000; MS=910024
GATE={0.0:1e-10,1e-3:2e-2,1e-2:2e-1}
def mex(d): return [(a,b) for s in range(d+1) for a in range(s+1) for b in [s-a]]
def phi(Z,d): return np.stack([Z[:,0]**a*Z[:,1]**b for (a,b) in mex(d)],axis=1)
def stud_loo(P,Y):
    G=P.T@P; C=np.linalg.lstsq(P,Y,rcond=None)[0]
    h=np.einsum('ij,jk,ik->i',P,np.linalg.inv(G),P)
    E=(Y-P@C)/(1.0-h)[:,None]
    s=float(np.mean(np.linalg.norm(E,axis=1)*np.sqrt(1.0-h)))
    return s,C,float(h.max())
def gen(rng,dstar,eta):
    Z=rng.standard_normal((N,2)); zq=rng.standard_normal((1,2))
    p=comb(dstar+2,2); Cs=rng.uniform(-1,1,(p,6))
    Y=phi(Z,dstar)@Cs+eta*rng.standard_normal((N,6))
    return Z,zq,Y,(phi(zq,dstar)@Cs)[0]
out={'spec':{'N':N,'draws_per_cell':NDRAW,'master_seed':MS,'operator':'studentized_PRESS+floor',
             'tau':'max(1e-8,8eta)','rho':RHO,'floor':FLOOR}}
cells={}
for dstar in D_SET:
    for eta in [0.0,1e-3,1e-2]:
        rng=np.random.default_rng(MS+1000*dstar+int(eta*1e6))
        tau=max(1e-8,8*eta); ab=0; mis=0; gx=0; e_list=[]
        for t in range(NDRAW):
            Z,zq,Y,yq=gen(rng,dstar,eta)
            sv={}; Cs={}
            for d in D_SET:
                sv[d],C,_=stud_loo(phi(Z,d),Y); Cs[d]=C
            smin=min(sv.values())
            if smin>tau: ab+=1; continue
            dhat=next(d for d in D_SET if sv[d]<=(1+RHO)*smin+FLOOR)
            if dhat!=dstar: mis+=1; continue
            yh=(phi(zq,dhat)@Cs[dhat])[0]
            e=float(np.linalg.norm(yh-yq)/max(np.linalg.norm(yq),1e-12))
            e_list.append(e)
            if e>GATE[eta]: gx+=1
        e=np.array(e_list) if e_list else np.array([np.nan])
        fails=ab+mis+gx
        cells['d%d_eta%g'%(dstar,eta)]={
            'false_abstain':ab/NDRAW,'misselect':mis/NDRAW,'gate_exceed':gx/NDRAW,
            'counts':{'abstain':ab,'misselect':mis,'gate_exceed':gx,'n':NDRAW},
            'total_fail_rate':fails/NDRAW,
            'fail_rate_upper95_rule3':(fails+3)/NDRAW if fails==0 else None,
            'e_p99':float(np.nanquantile(e,.99)),'e_max':float(np.nanmax(e)),'gate':GATE[eta]}
out['cells']=cells
# per-fold aggregate and 144-fold pass estimate (each cell appears 16x in development)
import math
logp=0.0; worst=0.0
for k,v in cells.items():
    p_fail=v['total_fail_rate']; worst=max(worst,p_fail)
    logp+=16*math.log(max(1.0-p_fail,1e-300))
out['aggregate']={'per_cell_fail_rates':{k:v['total_fail_rate'] for k,v in cells.items()},
                  'max_cell_fail_rate':worst,
                  'pass_all_144_estimate':math.exp(logp)}
# witness: degree-4 generator, eta=1e-3
rng=np.random.default_rng(MS+77); tau_w=8e-3; wm=[]
for t in range(NDRAW):
    Z,zq,Y,yq=gen(rng,4,1e-3)
    wm.append(min(stud_loo(phi(Z,d),Y)[0] for d in D_SET))
wm=np.array(wm)
out['witness_d4']={'abstain_rate':float(np.mean(wm>tau_w)),'tau':tau_w,
                   'margin_min_x_tau':float(np.min(wm)/tau_w),
                   'margin_med_x_tau':float(np.median(wm)/tau_w)}
# shuffle rejection, 1000/eta
rng=np.random.default_rng(MS+88); sh={}
for eta in [0.0,1e-3,1e-2]:
    tau=max(1e-8,8*eta); rej=0; smin_min=np.inf
    for t in range(1000):
        Z,zq,Y,yq=gen(rng,2,eta)
        Ys=Y[rng.permutation(N)]
        smin=min(stud_loo(phi(Z,d),Ys)[0] for d in D_SET)
        smin_min=min(smin_min,smin)
        if smin>tau: rej+=1
    sh['eta%g'%eta]={'reject_rate':rej/1000,'min_shuffled_s':float(smin_min),'tau':tau}
out['shuffle']=sh
print(json.dumps(out,indent=2,default=float))
