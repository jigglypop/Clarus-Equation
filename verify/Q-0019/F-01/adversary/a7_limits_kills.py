"""A7: recovers 재실행(다른 seed/delta), 극한, kill 실행가능성, dof."""
import json, math, subprocess, sys
from itertools import combinations
import numpy as np
HERE=__file__.replace("\\","/").rsplit("/",1)[0]
sys.path.insert(0, HERE.rsplit("/adversary",1)[0])
import predict_lambda as PL

out={}
parent=PL.regular_simplex_vertices()
L=PL.linear_map(parent); H=PL.centering(5)

# --- recovers (a): 공통 궤도, 다른 seed / 다른 alpha
res_orbit={}
for seed in (1, 777, 20260902, 424242):
    for alpha in (0.7, 1.0, 1.3, 2.5):
        rng=np.random.default_rng(seed)
        tet=[]
        for _ in range(5):
            a=rng.normal(size=(4,4)); a=0.2*(a-a.T)
            tet.append(alpha*PL.cayley_rotation(a))
        blk=PL.nonlinear_block(parent,tet)
        res_orbit[f"seed{seed}_alpha{alpha}"]={
            "metric_fine_rms":blk["metric_fine_rms"],"metric_coarse":blk["metric_coarse"],
            "eps12_coarse":blk["eps12_coarse"]}
out["orbit_reruns"]=res_orbit
out["orbit_max_residual"]=max(max(v.values()) for v in res_orbit.values())

# --- recovers (b): 코히런트 항등, 무작위 S 로
rng=np.random.default_rng(31337)
coh=0.0
for _ in range(200):
    s=rng.normal(size=10)
    coh=max(coh, float(np.linalg.norm(L@np.tile(s,5)-s)))
out["coherent_identity_max_residual"]=coh

# --- recovers (c): 게이지 4방향, 여러 q 크기
g=0.0
for scale in (0.01,0.1,0.3):
    cells=PL.sub_cells(parent)
    for k in range(4):
        q=np.zeros(4); q[k]=scale
        tet=[]
        for cell in cells:
            rows=cell[1:]-cell[0]; w=np.linalg.solve(rows,np.ones(4))
            tet.append(np.eye(4)-np.outer(q,w))
        x=np.concatenate([PL.sym_to_vec(0.5*(e+e.T)-np.eye(4)) for e in tet])
        g=max(g,float(np.linalg.norm(L@H@x)))
out["gauge_linear_max_residual"]=g

# --- 극한: 단일 cell 섭동 (다른 넷은 0)
sing={}
for a in range(5):
    best=0.0
    for _ in range(500):
        v=rng.normal(size=10); x=np.zeros(50); x[10*a:10*a+10]=v
        y=H@x
        best=max(best, float(np.linalg.norm(L@y)/(np.linalg.norm(y)/math.sqrt(5))))
    sing[f"cell{a}"]=best
out["single_cell_lambda_max"]=sing

# --- delta -> 0: L 이 delta 에 정확 무관한가 (선형성)
dl={}
for d in (0.2,0.02,0.002,2e-5):
    rng2=np.random.default_rng(9)
    x=H@rng2.normal(size=50); x=x/(np.linalg.norm(x)/math.sqrt(5))
    tet=[np.eye(4)+d*PL.vec_to_sym(x[10*a:10*a+10]) for a in range(5)]
    mets=[e.T@e for e in tet]
    gc=PL.coarse_metric(parent,mets); gbar=sum(mets)/5
    lam_nl=np.linalg.norm(gc-gbar)/ (math.sqrt(np.mean([np.sum((m-gbar)**2) for m in mets])))
    lam_lin=np.linalg.norm(L@x)/(np.linalg.norm(x)/math.sqrt(5))
    dl[str(d)]={"nonlinear":float(lam_nl),"linear":float(lam_lin),"rel_diff":float(abs(lam_nl/lam_lin-1))}
out["delta_limit"]=dl
print(json.dumps(out,indent=2,default=float))
open(HERE+"/a7_limits_kills.json","w").write(json.dumps(out,indent=2,default=float))
