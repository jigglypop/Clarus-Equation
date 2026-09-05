"""B2: 부호 규약 검사. 음의 섹터에서 'lambda^2>1 = 증폭' 이 규약 의존인가.
검사 A: N->|N| (부호 뒤집기) 펜슬. B: 작용 부호 S->-S (Euclidean Regge 부호 규약).
C: 실제 Rayleigh 해석 - 음의 섹터에서 M/N 비의 의미.
D: 각 고윳값 방향에서 |M| vs |N| 크기 비교 (증폭/수축의 절대 크기 판정)."""
import json, math, numpy as np, indep as I

pts=I.regular_points(2.0); cells=I.refine([tuple(I.BV)],pts)
b0=np.full(10,math.sqrt(2.0)); hc=I.richardson(I.coarse_action,b0)
cx=I.build(cells,pts,hc)
N,M=cx["N"],cx["M"]
ev=I.pencil(N,M)
def clus(v,tol=1e-5):
    out=[]
    for x in np.sort(np.asarray(v).real):
        if out and abs(out[-1][0]-x)<tol*max(1,abs(x)): out[-1].append(float(x))
        else: out.append([float(x)])
    return [{"lambda2":float(np.mean(c)),"mult":len(c)} for c in out]
# A: |N| = 절댓값 스펙트럼 재조립 (음의 고윳값을 양으로)
w,V=np.linalg.eigh(N); absN=V@np.diag(np.abs(w))@V.T
evA=I.pencil(absN,M)
# A2: |M| 도 함께
wm,Vm=np.linalg.eigh(M)
absM=Vm@np.diag(np.abs(wm))@Vm.T
evA2=I.pencil(absN,absM)
# B: 작용 전체 부호 뒤집기 S->-S : N->-N, M->-M => 펜슬 불변
evB=I.pencil(-N,-M)
# C: 음의 섹터에서 Rayleigh: v^T M v / v^T N v. 두 값 다 음수.
#    '증폭' 의 물리적 의미는 |coarse 곡률| vs |unglued 곡률|.
W=np.linalg.solve(N,M); wv,vv=np.linalg.eig(W)
rows=[]
for k in np.argsort(wv.real):
    if abs(wv[k])<1e-6: continue
    v=vv[:,k].real
    nn=float(v@N@v); mm=float(v@M@v)
    rows.append({"lambda2":float(wv[k].real),"vNv":nn,"vMv":mm,
                 "abs_ratio":abs(mm)/abs(nn),
                 "sector":"pos" if nn>0 else "neg"})
# D: 정부호화한 계량 (예: N_+ = V|w|V^T) 로 잰 '노름 증폭' = |M| Rayleigh / |N| Rayleigh
posdef_rayleigh=[]
for k in np.argsort(wv.real):
    if abs(wv[k])<1e-6: continue
    v=vv[:,k].real
    posdef_rayleigh.append(float((v@absM@v)/(v@absN@v)))
# E: conformal(스케일) 모드 Wick: H_c 의 양의 1방향만 부호 뒤집기 -> H_c^wick
wc,Vc=np.linalg.eigh(hc)
sgn=np.where(wc>0,-1.0,1.0)
hc_w=Vc@np.diag(wc*sgn)@Vc.T   # 양 고윳값만 음으로 (conformal Wick)
# 마찬가지로 N: cell 별 (2,8) 의 양의 2방향 뒤집기
Nw=np.zeros_like(N)
for a,h in enumerate(cx["hess"]):
    wa,Va=np.linalg.eigh(h); s=np.where(wa>0,-1.0,1.0)
    hw=Va@np.diag(wa*s)@Va.T
    t=cx["T"][a]
    Nw[10*a:10*a+10,10*a:10*a+10]=t.T@hw@t
Mw=cx["L"].T@hc_w@cx["L"]
evE=I.pencil(Nw,Mw)
res={
 "baseline_clusters":clus(ev),
 "A_absN_vs_M":{"clusters":clus(evA),"note":"N->|N| (부정부호 제거)"},
 "A2_absN_absM":{"clusters":clus(evA2)},
 "B_global_sign_flip":{"clusters":clus(evB),"identical_to_baseline":bool(np.allclose(np.sort(ev.real),np.sort(evB.real)))},
 "C_sectors":rows,
 "D_posdef_rayleigh":posdef_rayleigh,
 "E_conformal_wick":{"clusters":clus(evE),"Hc_wick_signature":[int(np.sum(np.linalg.eigvalsh(hc_w)>1e-9)),int(np.sum(np.linalg.eigvalsh(hc_w)<-1e-9))]},
}
print(json.dumps(res,ensure_ascii=True,indent=1))
open("b2_sign.json","w").write(json.dumps(res,ensure_ascii=True,indent=1))
