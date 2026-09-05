"""B4: unglued 분배 규약(kappa 등분할)이 결과를 결정하는 숨은 입력인가.
대안 분배: (i) 넓이 가중 kappa, (ii) 이면각 비례(=배경 이면각으로 가중, 배경에서 결손 0),
(iii) 전부 한 cell 에 몰아주기, (iv) 무작위 분배. glue 항등식 sum J^T H J = H_f 는 어떤 분배든 성립하나
(선형이므로) N (블록대각) 은 달라진다."""
import json, math, itertools, numpy as np, indep as I
pts=I.regular_points(2.0); cells=I.refine([tuple(I.BV)],pts)
b0=np.full(10,math.sqrt(2.0)); hc=I.richardson(I.coarse_action,b0)
base=I.build(cells,pts,hc)
def clus(v,tol=1e-4):
    out=[]
    for x in np.sort(np.asarray(v).real):
        if out and abs(out[-1][0]-x)<tol*max(1,abs(x)): out[-1].append(float(x))
        else: out.append([float(x)])
    return [{"lambda2":float(np.mean(c)),"mult":len(c)} for c in out]
res={"baseline_equal_split":clus(I.pencil(base["N"],base["M"]))}

def kappas_weighted(cells, weights):
    """weights[(cell_index, tri_key)] -> 가중치; 같은 hinge 의 가중치 합=1."""
    cnt={}
    for a,c in enumerate(cells):
        for t in itertools.combinations(c,3):
            k=tuple(sorted(t)); cnt.setdefault(k,[]).append(a)
    out=[]
    for a,c in enumerate(cells):
        ks=[]
        for t in itertools.combinations(c,3):
            k=tuple(sorted(t))
            tot=math.pi if all(v in I.BV for v in k) else 2*math.pi
            owners=cnt[k]
            w=weights(a,k,owners)
            ks.append(tot*w)
        out.append(np.asarray(ks))
    return out

def build_with_kappa(kap):
    lens=base["lens"]; T=base["T"]
    hess=[I.richardson(lambda v,k=k,c=c: I.cell_action(c,v,k), l) for c,l,k in zip(cells,lens,kap)]
    N=np.zeros((50,50))
    for a,(t,h) in enumerate(zip(T,hess)): N[10*a:10*a+10,10*a:10*a+10]=t.T@h@t
    return N, hess

# (i) 넓이 가중: hinge 를 품는 cell 들의 그 hinge 넓이는 같으므로(공유 삼각형) 등분할과 동일 -> 
#     대신 cell 부피 가중을 쓴다 (regular 에서 모두 같으므로 동일) -> 
#     진짜로 다른 분배: 비대칭 고정 가중
rng=np.random.default_rng(20260902)
def w_asym(a,k,owners):
    # cell index 순서로 (0.5,0.3,0.2) 식 비대칭
    ws=np.array([0.5,0.3,0.2,0.15,0.1][:len(owners)]); ws=ws/ws.sum()
    return float(ws[owners.index(a)])
def w_random(a,k,owners):
    r=np.random.default_rng(abs(hash(k))%(2**31)).uniform(size=len(owners)); r=r/r.sum()
    return float(r[owners.index(a)])
def w_all_first(a,k,owners):
    return 1.0 if a==owners[0] else 0.0
for name,w in (("asymmetric_fixed",w_asym),("random_per_hinge",w_random),("all_to_first_cell",w_all_first)):
    kap=kappas_weighted(cells,w)
    N,hess=build_with_kappa(kap)
    try:
        ev=I.pencil(N,base["M"])
        c=clus(ev)
    except Exception as e:
        c=[{"error":str(e)}]
    def sg(H,rel=1e-6):
        ww=np.linalg.eigvalsh(H); t=rel*np.max(np.abs(ww))
        return [int(np.sum(ww>t)),int(np.sum(ww<-t)),int(np.sum(np.abs(ww)<=t))]
    res[name]={"clusters":c,"signature_N":sg(N)}
# glue 항등식이 분배 무관인지 확인: sum_a J^T H_a J
def gidx(i,j):
    i,j=sorted((i,j))
    return 10+i if j==5 else list(I.EDGES).index((i,j))
def glue(hess):
    hs=np.zeros((15,15))
    for cell,h in zip(cells,hess):
        idx=[gidx(i,j) for i,j in itertools.combinations(cell,2)]
        hs[np.ix_(idx,idx)]+=h
    return hs
# H_f 참조: glued fine action (등분할 kappa 합 = 실제 결손)
base_glue=glue(base["hess"])
for name,w in (("asymmetric_fixed",w_asym),("all_to_first_cell",w_all_first)):
    kap=kappas_weighted(cells,w); N,hess=build_with_kappa(kap)
    res[name]["glue_rel_diff_vs_equal_split"]=float(np.linalg.norm(glue(hess)-base_glue)/np.linalg.norm(base_glue))
print(json.dumps(res,ensure_ascii=True,indent=1))
open("b4_split.json","w").write(json.dumps(res,ensure_ascii=True,indent=1))
