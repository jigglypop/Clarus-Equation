"""B1: 독립 재구현으로 스펙트럼·중복도 재현 + FD 정밀도(자릿수) 검사."""
import json, math, numpy as np, indep as I

pts = I.regular_points(2.0)
cells = I.refine([tuple(I.BV)], pts)
b0 = np.full(10, math.sqrt(2.0))
hc = I.richardson(I.coarse_action, b0)
cx = I.build(cells, pts, hc)
ev = I.pencil(cx["N"], cx["M"])
evl = I.pencil(cx["Nl"], cx["Ml"])

def clus(v, tol=1e-5):
    out=[]
    for x in np.sort(v.real):
        if out and abs(out[-1][0]-x)<tol*max(1,abs(x)): out[-1].append(float(x))
        else: out.append([float(x)])
    return [{"lambda2":float(np.mean(c)),"mult":len(c)} for c in out]

card = {"triv":1.0,"std":0.4296818,"[3,2]":1.4723737}
cl = clus(ev)
# FD step 민감도: h 다르게
prec={}
for h in (4e-3, 2e-3, 1e-3):
    hess=[I.richardson(lambda v,k=k,c=c: I.cell_action(c,v,k), l, h) for c,l,k in zip(cells,cx["lens"],cx["kap"])]
    hcx = I.richardson(I.coarse_action, b0, h)
    n=np.zeros((50,50))
    for a,(t,hh) in enumerate(zip(cx["T"],hess)): n[10*a:10*a+10,10*a:10*a+10]=t.T@hh@t
    m=cx["L"].T@hcx@cx["L"]
    e=I.pencil(n,m)
    prec[str(h)]=sorted(float(z.real) for z in e)
spread = {}
for k,tgt in (("std",0.4296818),("[3,2]",1.4723737)):
    vals=[c for c in cl if abs(c["lambda2"]-tgt)<1e-3]
    spread[k]=vals
def sigt(H,rel=1e-6):
    w=np.linalg.eigvalsh(H); t=rel*np.max(np.abs(w))
    return [int(np.sum(w>t)),int(np.sum(w<-t)),int(np.sum(np.abs(w)<=t))]
res={
 "clusters":cl,
 "spectrum":[float(z.real) for z in ev],
 "spectrum_length_chart":[float(z.real) for z in evl],
 "max_imag_ratio": float(np.max(np.abs(ev.imag)/np.abs(ev))),
 "card_targets":card,
 "dev_from_card":{"std":float(abs([c for c in cl if c['mult']==4][0]['lambda2']-0.4296818)),
                  "[3,2]":float(abs([c for c in cl if c['mult']==5][0]['lambda2']-1.4723737)),
                  "triv":float(abs([c for c in cl if c['mult']==1][0]['lambda2']-1.0))},
 "fd_step_scan":prec,
 "fd_spread_across_h":{ "std": float(max(prec[k][0] for k in prec)-min(prec[k][0] for k in prec)),
                        "[3,2]": float(max(prec[k][-1] for k in prec)-min(prec[k][-1] for k in prec))},
 "signature_N":sigt(cx["N"]), "signature_Hc":sigt(hc), "signature_cell":sigt(cx["hess"][0]),
}
print(json.dumps(res,ensure_ascii=True,indent=1))
open("b1_indep_spectrum.json","w").write(json.dumps(res,ensure_ascii=True,indent=1))
