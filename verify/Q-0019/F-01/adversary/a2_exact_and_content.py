"""A2: L 의 정확 닫힌 꼴 유도 + content 검사 (Schur/flat section 이 장식인가).

주장: 경계 제곱길이 ell2_ij = mean_{a notin {i,j}} u_ij^T g_a u_ij,
      coarse G 는 u_ij^T G u_ij = ell2_ij 로 유일 결정.
      따라서 G = sum_a c_a(g_a) 이고, 대칭성으로 L 은
      G = (1/3) sum_a [ (배경에서 a-빠진 가중) ] 꼴.
정확 형태를 직접 유도: 선형사상 T_a : Sym(4)->Sym(4), g_a |-> G 기여.
"""
import json, math
from fractions import Fraction as Fr
from itertools import combinations
import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=200)
HERE = __file__.replace("\\", "/").rsplit("/", 1)[0]
EDGES = tuple(combinations(range(5), 2))

# R^5 sum-zero 모형에서 모든 것을 유리수로. v_i = e_i - 1/5*1.  <v_i,v_j> = d_ij - 1/5
# 계량 g 는 sum-zero 4차원 위 대칭형식. u_ij = v_i - v_j.
# 5x5 대칭행렬 S 로 g 를 표현 (sum-zero 위 제한). u_ij^T g u_ij = (e_i-e_j)^T S (e_i-e_j)
#   = S_ii + S_jj - 2 S_ij.
# 이는 g 의 자유도 중 sum-zero 위 유효한 10차원과 일치(S 를 doubly-centered 로 고정).
# 따라서 파이프라인은 "S_ii+S_jj-2S_ij" 좌표(=제곱거리 d_ij)로 완전히 표현된다!
# 즉 L 은 제곱거리 공간 R^10 위 사상이다.
#   d^{(a)}_ij := cell a 의 계량이 주는 (i,j) 제곱거리
#   ell2_ij = (1/3) sum_{a notin {i,j}} d^{(a)}_ij
# 그리고 coarse G 는 정확히 d^G_ij = ell2_ij.  --> L 은 완전히 명시적이고 국소적!
print("=== L 의 명시 꼴: d^G_ij = (1/3) sum_{a notin{i,j}} d^{(a)}_ij  (제곱거리 좌표) ===")

# 이제 이 좌표에서 정확 스펙트럼을 유도한다.
# 입력: 다섯 cell 의 편차 계량 m_a in Sym(4)  ->  제곱거리 편차 벡터 D_a in R^10, D_a = P m_a
#   P: Sym(4)->R^10, (P m)_e = u_e^T m u_e.  regular 배경에서 P 는 가역(A 행렬).
# 출력: coarse 편차 G = P^{-1} ( (1/3) sum_{a notin e} D_a ).
# lambda 는 Frobenius(Sym(4)) 노름으로 재므로 P 가 개입한다.

# Frobenius 정규직교 Sym(4) 기저와 ambient 좌표 (Cholesky)
FG = np.array([[ (1.0 if k==l else 0.0) - 1/5 - ( -1/5 if False else 0) for l in range(4)] for k in range(4)])
# 정확히: <f_k,f_l> = ip(k,l)-ip(k,4)-ip(4,l)+ip(4,4), ip(i,j)=d_ij-1/5
def ip(i,j): return (1.0 if i==j else 0.0) - 0.2
FG = np.array([[ip(k,l)-ip(k,4)-ip(4,l)+ip(4,4) for l in range(4)] for k in range(4)])
Lch = np.linalg.cholesky(FG)
V = np.zeros((5,4))
for i in range(4):
    c = np.zeros(4); c[i]=1.0; V[i] = Lch.T @ c
V -= V.mean(axis=0)

def sym_basis():
    B=[]
    for i in range(4):
        m=np.zeros((4,4)); m[i,i]=1.0; B.append(m)
    for i in range(4):
        for j in range(i+1,4):
            m=np.zeros((4,4)); m[i,j]=m[j,i]=1/math.sqrt(2); B.append(m)
    return B
B=sym_basis()
def tv(m): return np.array([np.sum(b*m) for b in B])

P = np.array([[ (V[i]-V[j]) @ b @ (V[i]-V[j]) for b in B] for i,j in EDGES])   # 10x10
Pinv = np.linalg.inv(P)
# 인접 행렬: W[e,a] = 1/3 if a notin e else 0
W = np.zeros((10,5))
for e,(i,j) in enumerate(EDGES):
    for a in range(5):
        if a not in (i,j): W[e,a] = 1/3
# L = Pinv @ (W kron-ish) @ blockdiag(P)  ->  L[:, 10a:10a+10] = Pinv @ diag(W[:,a]) @ P
L = np.hstack([Pinv @ np.diag(W[:,a]) @ P for a in range(5)])
H = np.kron(np.eye(5)-np.ones((5,5))/5, np.eye(10))
LH = L@H
sv = np.linalg.svd(LH, compute_uv=False)
print("closed-form L spectrum sqrt5*sigma:", np.sqrt(5)*sv)

# --- 정확 유리수: 5*LH LH^T = 5 * sum_a Pinv diag(w_a - wbar) P P^T diag(...) Pinv^T ... 
# 대신 유리수 산술로 M = 5 LH LH^T 를 Fraction 으로.
# P 는 sqrt2 를 포함하지만 P P^T 등은 유리수가 될 수 있다. 직접 유리수 Gram 경로:
# 제곱거리 좌표에서: LH 의 작용은 D_a |-> Pinv (1/3) sum_{a notin e}(D_a - Dbar).
# Frobenius 노름 <m,m> = m 좌표. Q := P^{-T} P^{-1} 를 쓰면
#   ||G||_F^2 = || Pinv y ||^2 = y^T Q y,  y = (1/3) sum W (D_a - Dbar)
#   ||m_a||_F^2 = D_a^T Q D_a
# 따라서 스펙트럼은 일반화 고유값 문제 (Q 대칭 양정): 
Q = Pinv.T @ Pinv
Wc = W - W.mean(axis=1, keepdims=True)   # 10x5, 중심화
# 5*LH LH^T 고유값 = 일반화 고유값 of  (5 * Wc-based operator)
# 명시적으로: 입력 x = (D_1..D_5) 중심화, 노름^2 = (1/5) sum D_a^T Q D_a  <- rms
# 출력 노름^2 = y^T Q y, y_e = sum_a Wc[e,a] D_a  (중심화 입력이면 W == Wc 효과)
# lambda^2 = y^T Q y / ((1/5) sum D_a^T Q D_a)
# 최대화 -> 일반화 고유문제. Q 의 Cholesky 로 백색화.
Lq = np.linalg.cholesky(Q)
# z_a = Lq^T D_a  -> D_a^T Q D_a = |z_a|^2 ; y^T Q y = |Lq^T y|^2, Lq^T y = sum_a W[e,a]... 
# 하지만 W 는 대각(edge별)이므로 Lq^T diag 는 섞인다. 연산자 K_a = Lq^T diag(W[:,a]) Lq^{-T}
K = [Lq.T @ np.diag(W[:,a]) @ np.linalg.inv(Lq.T) for a in range(5)]
Kc = [K[a] - sum(K)/5 for a in range(5)]
Op = np.hstack(Kc)   # 10 x 50
sv2 = np.linalg.svd(Op, compute_uv=False)
print("whitened spectrum sqrt5*sigma:", np.sqrt(5)*sv2)

res = {
  "closed_form_reproduces": bool(np.allclose(np.sqrt(5)*sv, np.sqrt(5)*sv2, atol=1e-9)),
  "spectrum_sqrt5": (np.sqrt(5)*sv).tolist(),
  "L_is_local_in_squared_distance": True,
  "note_flat_section": "L 은 제곱거리 좌표에서 d^G_ij=(1/3)sum_{a notin ij} d^(a)_ij 로 완전히 명시적. 내부 길이/Schur/flat section 은 식에 전혀 등장하지 않는다.",
  "sum_of_squares_5trace": float(5*np.sum(sv**2)),
  "iso_sq": float(5*np.sum(sv**2)/40),
}
print(json.dumps(res, indent=2))
open(HERE+"/a2_exact_and_content.json","w").write(json.dumps(res, indent=2))
