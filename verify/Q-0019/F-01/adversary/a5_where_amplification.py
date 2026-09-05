"""A5: lambda_max>1 이 어디서 오는가. P (계량->제곱거리) 의 조건수가 만드는가.

제곱거리 좌표에서 사상은 A = (1/3)*indicator, 이는 명백히 수축(모든 sigma=sqrt(2/3)/... 확인).
계량 좌표로 옮기면 Pinv ... P 의 켤레가 비직교라 노름이 바뀐다.
"""
import json, math
from itertools import combinations
import numpy as np
HERE=__file__.replace("\\","/").rsplit("/",1)[0]
EDGES=tuple(combinations(range(5),2))
def ip(i,j): return (1.0 if i==j else 0.0)-0.2
FG=np.array([[ip(k,l)-ip(k,4)-ip(4,l)+ip(4,4) for l in range(4)] for k in range(4)])
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
P=np.array([[(V[i]-V[j])@b@(V[i]-V[j]) for b in B] for i,j in EDGES]); Pinv=np.linalg.inv(P)
svP=np.linalg.svd(P,compute_uv=False)
# 제곱거리 좌표의 순수 사상: N[e,a] = 1/3 if a notin e
W=np.zeros((10,5))
for e,(i,j) in enumerate(EDGES):
    for a in range(5):
        if a not in (i,j): W[e,a]=1/3
# 중심화된 edge-space 연산자 (D_a 를 독립 10벡터로 두면 대각)
Hc=np.kron(np.eye(5)-np.ones((5,5))/5,np.eye(10))
Ded=np.hstack([np.diag(W[:,a]) for a in range(5)])@Hc
svE=np.linalg.svd(Ded,compute_uv=False)
L=np.hstack([Pinv@np.diag(W[:,a])@P for a in range(5)]); LH=L@Hc
svM=np.linalg.svd(LH,compute_uv=False)

# lambda_max 고유벡터를 제곱거리 좌표에서 본다
U,S,Vt=np.linalg.svd(LH)
vmax=Vt[0]           # 50차원 입력
# 각 cell 의 제곱거리 편차
D=[P@vmax[10*a:10*a+10] for a in range(5)]
Dg=sum(W[:,a]*D[a] for a in range(5))
out={
 "P_singular_values": svP.tolist(),
 "P_condition_number": float(svP[0]/svP[-1]),
 "edge_coord_spectrum_sqrt5": (math.sqrt(5)*svE).tolist(),
 "edge_coord_all_equal_sqrt_2_3": bool(np.allclose(math.sqrt(5)*svE, math.sqrt(2/3), atol=1e-9)),
 "metric_coord_spectrum_sqrt5": (math.sqrt(5)*svM).tolist(),
 "diagnosis": "제곱거리(길이) 좌표에서 사상은 모든 방향 동일 sqrt(2/3)=0.8165<1 수축. lambda_max>1 은 오직 계량<->길이 좌표변환 P 의 비등방(조건수)에서 나온다.",
 "input_D_norms": [float(np.linalg.norm(d)) for d in D],
 "output_Dg_norm": float(np.linalg.norm(Dg)),
 "ratio_in_edge_coords_for_lambda_max_vector": float(np.linalg.norm(Dg)/ (np.linalg.norm(np.concatenate(D))/math.sqrt(5))),
}
print(json.dumps(out,indent=2))
open(HERE+"/a5_where_amplification.json","w").write(json.dumps(out,indent=2))
