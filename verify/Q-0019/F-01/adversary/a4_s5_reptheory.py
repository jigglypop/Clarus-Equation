"""A4: S_5 표현론 분해 검증 — 문자표로 차원 세기, 등변성 확인."""
import json, math
from itertools import combinations, permutations
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
W=np.zeros((10,5))
for e,(i,j) in enumerate(EDGES):
    for a in range(5):
        if a not in (i,j): W[e,a]=1/3
L=np.hstack([Pinv@np.diag(W[:,a])@P for a in range(5)])
H=np.kron(np.eye(5)-np.ones((5,5))/5,np.eye(10)); LH=L@H

# S_5 작용: 정점 치환 sigma -> ambient 직교행렬 R(sigma) (V[sigma(i)] = V[i] R^T 꼴)
def rot_of(perm):
    # V_perm[i] = V[perm[i]]. 두 배위 모두 같은 regular simplex이므로 직교 R 존재: V_perm = V @ R^T
    R,_,_ = np.linalg.lstsq(V, V[list(perm)], rcond=None)[0], None, None
    return R
def rot(perm):
    R = np.linalg.lstsq(V, V[list(perm)], rcond=None)[0]  # V @ R = V_perm
    return R  # 열 규약: V_perm = V @ R
# Sym(4) 위 유도 표현: m -> R^T m R  (계량은 pullback)
def sym_rep(R):
    M=np.zeros((10,10))
    for c,b in enumerate(B):
        t = R.T @ b @ R
        M[:,c]=[np.sum(bb*t) for bb in B]
    return M
# R^50 위: cell a 도 함께 치환. cell a 는 "정점 a 를 뺀" sub-cell 이므로 perm 이 a->perm[a]
def big_rep(perm):
    R=rot(perm); S=sym_rep(R)
    Big=np.zeros((50,50))
    for a in range(5):
        Big[10*perm[a]:10*perm[a]+10, 10*a:10*a+10] = S
    return R,S,Big

rng=np.random.default_rng(20260902)
maxerr=0.0; checked=0
allperms=list(permutations(range(5)))
for perm in allperms:
    R,S,Big=big_rep(perm)
    err=np.linalg.norm(LH@Big - S@LH)
    maxerr=max(maxerr,err); checked+=1
# 문자표: 표적 Sym(4) 표현의 문자 (S_5 켤레류)
classes={"1^5":(0,1,2,3,4),"2,1^3":(1,0,2,3,4),"2^2,1":(1,0,3,2,4),"3,1^2":(1,2,0,3,4),
         "3,2":(1,2,0,4,3),"4,1":(1,2,3,0,4),"5":(1,2,3,4,0)}
chars={}
for name,perm in classes.items():
    R,S,Big=big_rep(perm)
    chars[name]=float(np.trace(S))
# S_5 기약문자 (표준): triv, sgn, std(4), std x sgn(4), [3,2](5), [2,2,1](5), [3,1,1](6)
sizes={"1^5":1,"2,1^3":10,"2^2,1":15,"3,1^2":20,"3,2":20,"4,1":30,"5":24}
irr={
 "trivial":{"1^5":1,"2,1^3":1,"2^2,1":1,"3,1^2":1,"3,2":1,"4,1":1,"5":1},
 "sign":{"1^5":1,"2,1^3":-1,"2^2,1":1,"3,1^2":1,"3,2":-1,"4,1":-1,"5":1},
 "standard4":{"1^5":4,"2,1^3":2,"2^2,1":0,"3,1^2":1,"3,2":-1,"4,1":0,"5":-1},
 "std4xsgn":{"1^5":4,"2,1^3":-2,"2^2,1":0,"3,1^2":1,"3,2":1,"4,1":0,"5":-1},
 "[3,2]_5":{"1^5":5,"2,1^3":1,"2^2,1":1,"3,1^2":-1,"3,2":1,"4,1":-1,"5":0},
 "[2,2,1]_5":{"1^5":5,"2,1^3":-1,"2^2,1":1,"3,1^2":-1,"3,2":-1,"4,1":1,"5":0},
 "[3,1,1]_6":{"1^5":6,"2,1^3":0,"2^2,1":-2,"3,1^2":0,"3,2":0,"4,1":0,"5":1},
}
mults={}
for name,ch in irr.items():
    s=sum(sizes[c]*chars[c]*ch[c] for c in sizes)/120.0
    mults[name]=round(s,6)
out={"equivariance_max_err_over_120_perms":float(maxerr),"perms_checked":checked,
     "character_of_Sym4_target":chars,"multiplicities_in_Sym4":mults,
     "card_claims_1_4_5":"trivial=1, standard4=1, [3,2]_5=1"}
print(json.dumps(out,indent=2))
open(HERE+"/a4_s5_reptheory.json","w").write(json.dumps(out,indent=2))
