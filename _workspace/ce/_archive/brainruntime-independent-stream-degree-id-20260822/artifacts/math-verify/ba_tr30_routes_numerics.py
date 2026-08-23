# BA-TR30 routes numerics: studentized-PRESS statistic, parsimony floor, shuffle margins
import numpy as np, json
from math import comb
RHO=0.5; D_SET=[1,2,3]; N=14; NDRAW=200; MS=900030
def mex(d): return [(a,b) for s in range(d+1) for a in range(s+1) for b in [s-a]]
def phi(Z,d):
    return np.stack([Z[:,0]**a*Z[:,1]**b for (a,b) in mex(d)],axis=1)
def loo(P,Y):
    G=P.T@P; C=np.linalg.lstsq(P,Y,rcond=None)[0]
    h=np.einsum('ij,jk,ik->i',P,np.linalg.inv(G),P)
    E=(Y-P@C)/(1.0-h)[:,None]
    return np.linalg.norm(E,axis=1),h,C
def gen(rng,dstar,eta):
    Z=rng.standard_normal((N,2)); zq=rng.standard_normal((1,2))
    p=comb(dstar+2,2); Cs=rng.uniform(-1,1,(p,6))
    Y=phi(Z,dstar)@Cs+eta*rng.standard_normal((N,6))
    return Z,zq,Y,(phi(zq,dstar)@Cs)[0]
def stud_stat(P,Y):
    nrm,h,C=loo(P,Y)
    return float(np.mean(nrm*np.sqrt(1.0-h))),C
out={}
# R-C: studentized statistic s'_d = mean ||e_loo_i|| sqrt(1-h_ii); tau kept at max(1e-8,8eta)
for dstar in D_SET:
    for eta in [0.0,1e-3,1e-2]:
        rng=np.random.default_rng(MS+1000*dstar+int(eta*1e6))
        tau=max(1e-8,8*eta); ab=0; selok=0; gateok=0
        GATE={0.0:1e-10,1e-3:2e-2,1e-2:2e-1}[eta]
        for t in range(NDRAW):
            Z,zq,Y,yq=gen(rng,dstar,eta)
            sv={}; Cs={}
            for d in D_SET:
                sv[d],Cs[d]=stud_stat(phi(Z,d),Y)
            smin=min(sv.values())
            if smin>tau: ab+=1; continue
            dhat=next(d for d in D_SET if sv[d]<=(1+RHO)*smin+1e-8)  # R-E floor included
            if dhat==dstar:
                selok+=1
                yh=(phi(zq,dhat)@Cs[dhat])[0]
                e=np.linalg.norm(yh-yq)/max(np.linalg.norm(yq),1e-12)
                if e<=GATE: gateok+=1
        out['RC_RE_d%d_eta%g'%(dstar,eta)]={'false_abstain':ab/NDRAW,
            'sel_ok_of_nonabstain':selok/max(NDRAW-ab,1),'sel_and_gate':gateok/NDRAW}
# witness under studentized statistic
rng=np.random.default_rng(MS+77); wmins=[]
for t in range(NDRAW):
    Z,zq,Y,yq=gen(rng,4,1e-3)
    wmins.append(min(stud_stat(phi(Z,d),Y)[0] for d in D_SET))
out['RC_witness']={'min':float(np.min(wmins)),'units_of_eta':float(np.min(wmins))/1e-3,
                   'abstain_rate':float(np.mean(np.array(wmins)>8e-3))}
# shuffle margin under studentized statistic and under mean stat with tau=60eta
rng=np.random.default_rng(MS+88); res={}
for eta in [1e-3,1e-2]:
    smin_stud=[]; smin_mean=[]
    for t in range(100):
        Z,zq,Y,yq=gen(rng,2,eta)
        Ys=Y[rng.permutation(N)]
        smin_stud.append(min(stud_stat(phi(Z,d),Ys)[0] for d in D_SET))
        smin_mean.append(min(float(np.mean(loo(phi(Z,d),Ys)[0])) for d in D_SET))
    res['eta%g'%eta]={'stud_min':float(np.min(smin_stud)),
                      'mean_min':float(np.min(smin_mean)),
                      'tau_stud_8eta':8*eta,'tau_mean_60eta':60*eta,
                      'stud_reject_all':bool(np.min(smin_stud)>8*eta),
                      'mean60_reject_all':bool(np.min(smin_mean)>60*eta)}
out['shuffle_margins']=res
# R-E alone at eta=0 with original mean statistic: misselect fixed?
rng=np.random.default_rng(MS+1000*1+0); mis=0
for t in range(NDRAW):
    Z,zq,Y,yq=gen(rng,1,0.0)
    sv={d:float(np.mean(loo(phi(Z,d),Y)[0])) for d in D_SET}
    smin=min(sv.values())
    dhat=next(d for d in D_SET if sv[d]<=(1+RHO)*smin+1e-8)
    mis+=(dhat!=1)
out['RE_eta0_d1_misselect_of_200']=mis
print(json.dumps(out,indent=2,default=float))
