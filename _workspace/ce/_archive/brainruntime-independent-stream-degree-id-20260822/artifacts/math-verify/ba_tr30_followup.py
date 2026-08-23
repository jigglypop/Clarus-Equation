# BA-TR30 follow-up: root cause of d3 false abstain + repair-route numerics
import numpy as np, json
from math import comb

RHO = 0.5; D_SET=[1,2,3]; N=14; NDRAW=200; MS=900030

def monomial_exponents(d):
    return [(a,b) for s in range(d+1) for a in range(s+1) for b in [s-a]]
def phi(Z,d):
    ex=monomial_exponents(d)
    return np.stack([Z[:,0]**a*Z[:,1]**b for (a,b) in ex],axis=1)
def loo_norms(Phi,Y):
    G=Phi.T@Phi; C=np.linalg.lstsq(Phi,Y,rcond=None)[0]
    h=np.einsum('ij,jk,ik->i',Phi,np.linalg.inv(G),Phi)
    E=(Y-Phi@C)/(1.0-h)[:,None]
    return np.linalg.norm(E,axis=1), h, C
def gen(rng,dstar,eta,dist='gauss'):
    if dist=='gauss':
        Z=rng.standard_normal((N,2)); zq=rng.standard_normal((1,2))
    else:
        Z=rng.uniform(-1.5,1.5,(N,2)); zq=rng.uniform(-1.5,1.5,(1,2))
    p=comb(dstar+2,2); Cs=rng.uniform(-1,1,(p,6))
    Y=phi(Z,dstar)@Cs+eta*rng.standard_normal((N,6))
    return Z,zq,Y,(phi(zq,dstar)@Cs)[0]

out={}
# 1) s_3/eta distribution and unconditional prediction error, gauss vs uniform cues
for dist in ['gauss','uniform']:
    for eta in [1e-3,1e-2]:
        rng=np.random.default_rng(MS+11)
        r_mean=[]; r_med=[]; e_all=[]; selok=0
        for t in range(NDRAW):
            Z,zq,Y,yq=gen(rng,3,eta,dist)
            svals={}; Cs={}
            for d in D_SET:
                nrm,h,C=loo_norms(phi(Z,d),Y)
                svals[d]=float(np.mean(nrm)); Cs[d]=C
                if d==3:
                    r_mean.append(svals[3]/eta)
                    r_med.append(float(np.median(nrm))/eta)
            smin=min(svals.values())
            dhat=next(d for d in D_SET if svals[d]<=(1+RHO)*smin)
            selok+=(dhat==3)
            yh=(phi(zq,dhat)@Cs[dhat])[0]
            e_all.append(float(np.linalg.norm(yh-yq)/max(np.linalg.norm(yq),1e-12)))
        q=lambda a,p:float(np.quantile(a,p))
        out['%s_eta%g'%(dist,eta)]={
          's3mean_over_eta':{'p50':q(r_mean,.5),'p95':q(r_mean,.95),'p99':q(r_mean,.99),'max':q(r_mean,1)},
          's3median_over_eta':{'p50':q(r_med,.5),'p95':q(r_med,.95),'p99':q(r_med,.99),'max':q(r_med,1)},
          'frac_s3mean_gt_8eta':float(np.mean(np.array(r_mean)>8.0)),
          'frac_s3med_gt_8eta':float(np.mean(np.array(r_med)>8.0)),
          'sel_ok_rate_uncond':selok/NDRAW,
          'e_uncond':{'p50':q(e_all,.5),'p95':q(e_all,.95),'max':q(e_all,1)}}
# 2) witness margin under median statistic and larger tau
rng=np.random.default_rng(MS+77)
w_med=[]; w_mean=[]
for t in range(NDRAW):
    Z,zq,Y,yq=gen(rng,4,1e-3)
    means=[]; meds=[]
    for d in D_SET:
        nrm,_,_=loo_norms(phi(Z,d),Y)
        means.append(float(np.mean(nrm))); meds.append(float(np.median(nrm)))
    w_mean.append(min(means)); w_med.append(min(meds))
out['witness_d4_eta1e-3']={'min_over_draws_of_min_d_mean':float(np.min(w_mean)),
                           'min_over_draws_of_min_d_median':float(np.min(w_med)),
                           'units_of_eta_mean':float(np.min(w_mean))/1e-3,
                           'units_of_eta_median':float(np.min(w_med))/1e-3}
# 3) eta=0 d*=1 misselection detail
rng=np.random.default_rng(MS+1000*1+0)
fails=[]
for t in range(NDRAW):
    Z,zq,Y,yq=gen(rng,1,0.0)
    svals={d:float(np.mean(loo_norms(phi(Z,d),Y)[0])) for d in D_SET}
    smin=min(svals.values())
    dhat=next(d for d in D_SET if svals[d]<=(1+RHO)*smin)
    if dhat!=1: fails.append({'s':svals,'dhat':dhat})
out['eta0_d1_misselect']={'count_of_200':len(fails),'examples':fails[:3]}
# 4) LOO identity reldiff split: noisy cases only
rng=np.random.default_rng(MS+1)
diffs=[]
for t in range(60):
    dstar=int(rng.integers(1,4)); eta=float(rng.choice([1e-3,1e-2]))
    Z,zq,Y,yq=gen(rng,dstar,eta)
    for d in D_SET:
        P=phi(Z,d)
        nrm,_,_=loo_norms(P,Y)
        s_hat=float(np.mean(nrm))
        n=P.shape[0]; refit=[]
        for i in range(n):
            m=np.ones(n,bool); m[i]=False
            Ci=np.linalg.lstsq(P[m],Y[m],rcond=None)[0]
            refit.append(np.linalg.norm(Y[i]-P[i]@Ci))
        s_ref=float(np.mean(refit))
        diffs.append(abs(s_hat-s_ref)/s_ref)
out['loo_identity_reldiff_noisy_max']=float(np.max(diffs))
print(json.dumps(out,indent=2,default=float))
