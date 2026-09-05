"""B5: recovers 4종 재실행 (다른 seed/delta/h) + 극한(delta->0, 단일 cell, 대칭 깨짐)."""
import json, math, itertools, numpy as np, indep as I
pts=I.regular_points(2.0); cells=I.refine([tuple(I.BV)],pts)
b0=np.full(10,math.sqrt(2.0)); hc=I.richardson(I.coarse_action,b0)
cx=I.build(cells,pts,hc)
N,M,L=cx["N"],cx["M"],cx["L"]
res={}
# R1: so(4) 회전 mismatch = 0 (sym(anti)=0)
devs=[]
for s in (1,777,20260902):
    r=np.random.default_rng(s); a=r.normal(size=(4,4)); a=a-a.T
    v=np.tile(np.array([float(np.sum(b*(0.5*(a+a.T)))) for b in I.BASIS]),5)
    devs.append(float(np.linalg.norm(v)))
res["so4_mismatch_norms"]=devs
# R2: 공통 alpha Rayleigh = 1
al=np.tile(np.array([float(np.sum(b*np.eye(4))) for b in I.BASIS]),5)
res["common_alpha_rayleigh"]=float((al@M@al)/(al@N@al))
# R3: 코히런트 Rayleigh = 1 (10 방향, 여러 h)
coh=np.array([np.tile(np.array([float(np.sum(b*bb)) for b in I.BASIS]),5) for bb in I.BASIS]).T
res["coherent_rayleigh"]=[float((c@M@c)/(c@N@c)) for c in coh.T]
res["coherent_max_dev"]=float(max(abs(x-1) for x in res["coherent_rayleigh"]))
# R4: glued 제한 -> lambda = 1 (perfect action). glued 15차원 부분공간을 mismatch 공간에 embed.
# glued: 15 글로벌 길이 편차 -> cell 길이 편차 J_a ; 그러나 mismatch 공간은 metric 좌표(cell 별 Sym(4)).
# glued 부분공간 = 코히런트(공통 metric 편차) + 내부 정점 이동. 여기서는 길이 좌표에서 직접.
def gidx(i,j):
    i,j=sorted((i,j)); return 10+i if j==5 else list(I.EDGES).index((i,j))
Jg=np.zeros((50,15))
for a,c in enumerate(cells):
    for r,(i,j) in enumerate(itertools.combinations(c,2)):
        Jg[10*a+r, gidx(i,j)]=1.0
Nl,Ml=cx["Nl"],cx["Ml"]
Ng=Jg.T@Nl@Jg; Mg=Jg.T@Ml@Jg
# glued 위에서 내부 5 를 Schur 소거 -> coarse 와 비교
A,B,C=Ng[:10,:10],Ng[10:,10:],None
Abb=Ng[:10,:10]; Bby=Ng[:10,10:]; Cyy=Ng[10:,10:]
Cp=np.linalg.pinv(Cyy)
Heff=Abb-Bby@Cp@Bby.T
res["glued_schur_vs_Hc_rel"]=float(np.linalg.norm(Heff-hc)/np.linalg.norm(hc))
# glued 펜슬: L|glued 는 항등 (경계 변은 3-cell 평균 = 같은 값)
Lg=cx["Ll"]@Jg     # 10x15
Mg2=Lg.T@hc@Lg
w=np.linalg.eigvals(np.linalg.pinv(Ng)@Mg2)
w=w[np.abs(w)>1e-6]
res["glued_pencil_eigs"]=sorted(float(z.real) for z in w)
res["glued_all_near_one"]=bool(np.all(np.abs(np.asarray(res["glued_pencil_eigs"])-1)<1e-5))
# R5: delta->0 선형화 (다른 seed)
lin={}
for s in (1, 20260902):
    r=np.random.default_rng(s); v=r.normal(size=50); v/=np.linalg.norm(v)
    def ft(d):
        return sum(I.cell_action(c, l+d*(t@v[10*a:10*a+10]), k) for a,(c,l,t,k) in enumerate(zip(cells,cx["lens"],cx["T"],cx["kap"])))
    dc=L@v
    e={}
    for d in (0.02,0.01,0.005):
        fd=(ft(d)+ft(-d)-2*ft(0.0))/d**2
        cd=(I.coarse_action(b0+d*dc)+I.coarse_action(b0-d*dc)-2*I.coarse_action(b0))/d**2
        e[str(d)]={"fine_rel":float(abs(fd-v@N@v)/abs(v@N@v)),"coarse_rel":float(abs(cd-v@M@v)/abs(v@M@v))}
    lin[str(s)]=e
res["delta_linearization"]=lin
# R6: 단일 cell 극한 - cells 하나만 (펜슬 정의 가능한가)
res["single_cell_note"]="1->5 가 아니면 L 이 정의되지 않는다(경계 변을 품는 cell 이 하나) -> 펜슬은 N=M 자명 1"
# R7: 대칭 깨짐 (작은 비정규)
def spec(sq):
    p=I.points_from_sq(sq); cs=I.refine([tuple(I.BV)],p)
    bb=np.sqrt(sq); h=I.richardson(I.coarse_action,bb)
    c=I.build(cs,p,h)
    return sorted(float(z.real) for z in I.pencil(c["N"],c["M"]))
rng=np.random.default_rng(20260902); sg=rng.choice([-1.0,1.0],size=10)
res["symmetry_breaking"]={}
for amp in (0.001,0.01):
    res["symmetry_breaking"][str(amp)]=spec(2.0*(1+amp*sg))
print(json.dumps(res,ensure_ascii=True,indent=1))
open("b5_recovers.json","w").write(json.dumps(res,ensure_ascii=True,indent=1))
