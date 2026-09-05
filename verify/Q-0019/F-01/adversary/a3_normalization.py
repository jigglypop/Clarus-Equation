"""A3: 정규화 민감성. lambda_max>1 이 RMS 규약의 인공물인가."""
import json, math
from itertools import combinations
import numpy as np
HERE = __file__.replace("\\","/").rsplit("/",1)[0]
EDGES = tuple(combinations(range(5),2))
def ip(i,j): return (1.0 if i==j else 0.0)-0.2
FG = np.array([[ip(k,l)-ip(k,4)-ip(4,l)+ip(4,4) for l in range(4)] for k in range(4)])
Lch=np.linalg.cholesky(FG); V=np.zeros((5,4))
for i in range(4):
    c=np.zeros(4); c[i]=1.0; V[i]=Lch.T@c
V-=V.mean(axis=0)
def sym_basis():
    B=[]
    for i in range(4):
        m=np.zeros((4,4)); m[i,i]=1.0; B.append(m)
    for i in range(4):
        for j in range(i+1,4):
            m=np.zeros((4,4)); m[i,j]=m[j,i]=1/math.sqrt(2); B.append(m)
    return B
B=sym_basis()
P=np.array([[(V[i]-V[j])@b@(V[i]-V[j]) for b in B] for i,j in EDGES])
Pinv=np.linalg.inv(P)
W=np.zeros((10,5))
for e,(i,j) in enumerate(EDGES):
    for a in range(5):
        if a not in (i,j): W[e,a]=1/3
L=np.hstack([Pinv@np.diag(W[:,a])@P for a in range(5)])
H=np.kron(np.eye(5)-np.ones((5,5))/5,np.eye(10))
LH=L@H
sv=np.linalg.svd(LH,compute_uv=False)

out={}
# 규약 1 (카드): ||M||_rms = sqrt( (1/5) sum ||M_a||_F^2 ) = ||x||/sqrt5
out["conv_rms_card"] = {"lambda_max": float(math.sqrt(5)*sv[0]),
                        "lambda_iso": float(math.sqrt(5*np.sum(sv**2)/40)),
                        "gt1": bool(math.sqrt(5)*sv[0] > 1)}
# 규약 2: 총합 노름 ||M||_2 = sqrt(sum_a ||M_a||^2) = ||x||  (50차원 Euclid)
out["conv_sum"] = {"lambda_max": float(sv[0]), "lambda_iso": float(math.sqrt(np.sum(sv**2)/40)),
                   "gt1": bool(sv[0] > 1)}
# 규약 3: 최대 cell 노름 max_a ||M_a||  (분모 <= ||x||, 최악은 단일 cell)
# lambda_maxcell = max over x of ||LHx|| / max_a||x_a||.  하한/상한 수치 탐색
rng=np.random.default_rng(20260902)
best=0.0
for _ in range(4000):
    x=H@rng.normal(size=50)
    den=max(np.linalg.norm(x[10*a:10*a+10]) for a in range(5))
    best=max(best, np.linalg.norm(LH@x)/den)
# 단일 cell 만 섭동한 경우도 명시적으로
single=0.0
for a in range(5):
    Ma=L[:,10*a:10*a+10]  # 단일 cell 입력 (중심화 전)
    x=np.zeros(50)
    for _ in range(2000):
        v=rng.normal(size=10); x[:]=0; x[10*a:10*a+10]=v
        y=H@x
        den=max(np.linalg.norm(y[10*b:10*b+10]) for b in range(5))
        single=max(single, np.linalg.norm(LH@y)/den)
out["conv_maxcell"]={"lambda_max_est": float(best), "single_cell_est": float(single), "gt1": bool(best>1)}
# 규약 4: 정보 이론적 자연 규약 — coarse 편차 vs fine 편차 평균 (즉 mean_a ||M_a||^2 그대로 = 카드)
# 규약 5: "cell 부피 가중" — barycentric sub-cell 은 부피 1/5 씩, 계량 편차의 부피가중 L2
#   = 같은 rms (동일). 
# 규약 6: 제곱거리 좌표(계량이 아니라 길이)로 재기: 분자 ||D^G||, 분모 rms ||D^(a)||
Dnum = np.hstack([np.diag(W[:,a])@np.eye(10) for a in range(5)])  # D_a -> D^G, 직접
DH = Dnum @ np.kron(np.eye(5)-np.ones((5,5))/5, np.eye(10))
svd_=np.linalg.svd(DH,compute_uv=False)
out["conv_edge_length"]={"lambda_max": float(math.sqrt(5)*svd_[0]),
                         "lambda_iso": float(math.sqrt(5*np.sum(svd_**2)/40)),
                         "gt1": bool(math.sqrt(5)*svd_[0]>1),
                         "spectrum": (math.sqrt(5)*svd_).tolist()}
# 규약 7: Frobenius on tetrad perturbation sym(eta) 그대로 (= 카드와 동일 좌표, 이미 위)
# 규약 8: 배경계량 기준 지수 좌표 (g = exp(2 delta m)) -> 1차에서 동일
print(json.dumps(out, indent=2))
open(HERE+"/a3_normalization.json","w").write(json.dumps(out, indent=2))
