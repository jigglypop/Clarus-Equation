"""B6: content/dof/dimension + L(blocking) 규약 의존 + mu 항등식이 정리인지 항등인지."""
import json, math, itertools, numpy as np, indep as I
pts=I.regular_points(2.0); cells=I.refine([tuple(I.BV)],pts)
b0=np.full(10,math.sqrt(2.0)); hc=I.richardson(I.coarse_action,b0)
cx=I.build(cells,pts,hc)
def clus(v,tol=1e-4):
    out=[]
    for x in np.sort(np.asarray(v).real):
        if out and abs(out[-1][0]-x)<tol*max(1,abs(x)): out[-1].append(float(x))
        else: out.append([float(x)])
    return [{"lambda2":float(np.mean(c)),"mult":len(c)} for c in out]
res={"baseline":clus(I.pencil(cx["N"],cx["M"]))}
# L 규약 대안: (a) 기하평균, (b) 가중 (0.5,0.3,0.2), (c) 제곱길이 평균 후 sqrt (=1차 동일? 확인)
def L_alt(weights):
    Ll=np.zeros((10,50))
    for k,(i,j) in enumerate(I.EDGES):
        own=[]
        for a,c in enumerate(cells):
            if i in c and j in c:
                em=I.edge_map(c); own.append((a,em[tuple(sorted((i,j)))]))
        w=weights(len(own))
        for n,(a,r) in enumerate(own): Ll[k,10*a+r]=w[n]
    return Ll@cx["L"].__class__(np.block([[np.zeros((0,0))]])) if False else Ll
P=np.zeros((50,50))
for a,t in enumerate(cx["T"]): P[10*a:10*a+10,10*a:10*a+10]=t
for name,w in (("weighted_0.5_0.3_0.2", lambda n: np.array([0.5,0.3,0.2][:n])/sum([0.5,0.3,0.2][:n])),
               ("first_cell_only", lambda n: np.array([1.0]+[0.0]*(n-1)))):
    Ll=L_alt(w); Lx=Ll@P; Mx=Lx.T@hc@Lx
    try: c=clus(I.pencil(cx["N"],Mx))
    except Exception as e: c=[{"error":str(e)}]
    res[name]={"clusters":c,"rank_M":int(np.linalg.matrix_rank(Mx,1e-9))}
# 차원 검사: Regge 작용 = 넓이 x 각 -> [길이^2]; Hessian_l = [길이^0]; N,M 둘다 [길이^0]*[mismatch]^2 -> lambda^2 무차원
res["dimension"]={"S":"length^2","H_l=d2S/dl2":"length^0","N,M":"same weight","lambda2":"dimensionless",
                  "scale_invariance_check":None}
# 스케일 불변 실측: 배경 제곱변 2 -> 8 (길이 2배)
p2=I.points_from_sq(np.full(10,8.0)); c2=I.refine([tuple(I.BV)],p2)
b2=np.full(10,math.sqrt(8.0)); h2=I.richardson(I.coarse_action,b2)
cx2=I.build(c2,p2,h2)
res["dimension"]["scale_invariance_check"]={"sq=8_clusters":clus(I.pencil(cx2["N"],cx2["M"]))}
# mu 항등식: N-중심화 펜슬 = lambda^2 - 1 (정리 주장)
coh=np.array([np.tile(np.array([float(np.sum(b*bb)) for b in I.BASIS]),5) for bb in I.BASIS]).T
_,_,vt=np.linalg.svd(coh.T@cx["N"]); Z=vt[10:].T
evc=I.pencil(Z.T@cx["N"]@Z, Z.T@cx["M"]@Z)
ev=I.pencil(cx["N"],cx["M"])
nonunit=ev[np.abs(ev.real-1.0)>1e-5]
res["mu_identity"]={"centered_spectrum":sorted(float(z.real) for z in evc),
  "nonunit_minus_1":sorted(float(z.real-1.0) for z in nonunit),
  "max_dev":float(np.max(np.abs(np.sort(evc.real)-np.sort(nonunit.real-1.0))))}
# dof / 사전등록 숫자 세기
res["dof"]={"free_parameters":0,"prereg_numbers_in_predicts":14,"kill_numbers":6,
            "verdict":"자유 파라미터 0 < 예측 숫자"}
print(json.dumps(res,ensure_ascii=True,indent=1))
open("b6_content.json","w").write(json.dumps(res,ensure_ascii=True,indent=1))
