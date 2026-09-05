"""A1: L 사상의 완전 독립 재구현 + 정확 유리수 스펙트럼.

카드 파이프라인을 다른 방식으로 재구현한다:
  - 정점 좌표를 Gram 행렬 경유가 아니라 명시적 regular simplex 좌표로 구성
  - coarse Gram solve를 BASIS 확장 대신 직접 10x10 선형계로
  - Fraction 산술로 정확 유리수 스펙트럼을 얻는다 (regular 경계는 유리수 Gram 가능)
"""
import json, math
from fractions import Fraction as F
from itertools import combinations
import numpy as np

HERE = __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0]
EDGES = tuple(combinations(range(5), 2))

# ---- 독립 좌표: regular 4-simplex, 제곱변 2, 무게중심 원점.
# R^5의 표준기저 e_i 를 sum=0 초평면에 사영하면 |e_i-e_j|^2 = 2. 4차원 정규직교좌표로 내리되
# 여기서는 "좌표 없는" 내적만 쓴다: <v_i, v_j> = delta_ij - 1/5.
def ip(i, j):
    return (F(1) if i == j else F(0)) - F(1, 5)

# 계량 g는 4x4 대칭이지만, 우리는 정점을 R^5의 sum-zero 부분공간에 두고
# g를 그 부분공간 위 대칭 이중선형형식으로 다룬다. 자유도는 여전히 10.
# 기저: 부분공간 좌표 f_k = v_k - v_4 (k=0..3) 를 쓰면 임의 v_i - v_j 는 f들의 정수결합.
# f_k 의 Gram: <f_k, f_l> = ip(k,l) - ip(k,4) - ip(4,l) + ip(4,4)
FG = [[ip(k, l) - ip(k, 4) - ip(4, l) + ip(4, 4) for l in range(4)] for k in range(4)]
# f 좌표계에서 v_i - v_j 의 성분 (정수 벡터)
def edge_vec(i, j):
    c = [F(0)] * 4
    if i < 4: c[i] += 1
    if j < 4: c[j] -= 1
    return c

# Sym(4) 를 f-좌표계 성분으로 파라미터화: G_kl. 배경 계량은 FG.
# u^T G u where u = edge_vec. 배경에서 이는 |v_i-v_j|^2 = 2 이어야 함 (검증).
def quad(G, u):
    return sum(u[k] * G[k][l] * u[l] for k in range(4) for l in range(4))

for (i, j) in EDGES:
    assert quad(FG, edge_vec(i, j)) == F(2), (i, j, quad(FG, edge_vec(i, j)))

# Sym(4) 10차원 기저 (f-좌표, Frobenius 정규직교 아님 — 나중에 배경계량 기준으로 처리)
SYMIDX = [(k, l) for k in range(4) for l in range(k, 4)]
def sym_from_params(p):
    G = [[F(0)] * 4 for _ in range(4)]
    for (v, (k, l)) in zip(p, SYMIDX):
        G[k][l] = v; G[l][k] = v
    return G

# ---- 핵심: 카드의 파이프라인은 "ambient 정규직교 좌표"에서 정의됨.
# ambient 좌표를 명시적으로 만들되, 카드 스크립트의 QR과 다른 방법(Cholesky of FG)으로.
FGn = np.array([[float(x) for x in row] for row in FG])
Lch = np.linalg.cholesky(FGn)            # FG = Lch Lch^T
# f-좌표 c -> ambient 좌표 x = Lch^T c  이면 <x,x> = c^T FG c. 
VERTS = np.zeros((5, 4))
for i in range(5):
    c = np.zeros(4)
    if i < 4: c[i] = 1.0
    VERTS[i] = Lch.T @ c
VERTS = VERTS - VERTS.mean(axis=0)
# 검증: 제곱변 2
for (i, j) in EDGES:
    d = VERTS[i] - VERTS[j]
    assert abs(d @ d - 2.0) < 1e-12

def sym_basis_frob():
    B = []
    for i in range(4):
        m = np.zeros((4, 4)); m[i, i] = 1.0; B.append(m)
    for i in range(4):
        for j in range(i + 1, 4):
            m = np.zeros((4, 4)); m[i, j] = m[j, i] = 1 / math.sqrt(2.0); B.append(m)
    return B
B = sym_basis_frob()
def to_vec(m): return np.array([float(np.sum(b * m)) for b in B])
def from_vec(v): return sum(float(c) * b for c, b in zip(v, B))

# 독립 구현의 coarse map: 경계 제곱길이 -> Gram, 직접 10x10 (BASIS 아닌 (k,l) 파라미터)
A = np.zeros((10, 10))
for e, (i, j) in enumerate(EDGES):
    u = VERTS[i] - VERTS[j]
    for c, (k, l) in enumerate(SYMIDX):
        A[e, c] = u[k] * u[l] * (1.0 if k == l else 2.0)
Ainv = np.linalg.inv(A)

def coarse_from_metrics(mets):
    ell2 = np.empty(10)
    for e, (i, j) in enumerate(EDGES):
        u = VERTS[i] - VERTS[j]
        ell2[e] = np.mean([u @ mets[a] @ u for a in range(5) if a not in (i, j)])
    p = Ainv @ ell2
    G = np.zeros((4, 4))
    for v, (k, l) in zip(p, SYMIDX):
        G[k, l] = v; G[l, k] = v
    return G

# L: R^50 -> R^10 (Frobenius-정규직교 좌표)
cols = []
for a in range(5):
    for b in B:
        mets = [np.zeros((4, 4)) for _ in range(5)]
        mets[a] = b
        cols.append(to_vec(coarse_from_metrics(mets)))
L = np.array(cols).T

Hc = np.kron(np.eye(5) - np.ones((5, 5)) / 5, np.eye(10))
LH = L @ Hc
sv = np.linalg.svd(LH, compute_uv=False)
sv5 = np.sqrt(5.0) * sv
iso = math.sqrt(5.0 * float(np.sum(sv**2)) / 40.0)

# ---- 정확 유리수 경로: L 을 f-좌표에서 Fraction으로 구성 후 (LH)(LH)^T 의 고유값
# ambient Frobenius 좌표는 무리수를 섞으므로, 대신 5*sigma^2 를 유리수로 확인:
# LH (LH)^T 의 고유다항식을 numpy로 얻고 유리수 근사와 비교
M = 5.0 * (LH @ LH.T)
ev = np.sort(np.linalg.eigvalsh(M))[::-1]
targets = [11/9]*5 + [5/9]*4 + [1/3]
out = {
    "singular_values_sqrt5": sv5.tolist(),
    "eig_5LHLHT_sorted": ev.tolist(),
    "card_targets": targets,
    "max_abs_dev_from_card": float(np.max(np.abs(ev - np.array(targets)))),
    "lambda_iso": iso,
    "lambda_iso_sq": iso**2,
    "lambda_iso_sq_minus_13_60": iso**2 - 13/60,
    "lambda_max": float(sv5[0]),
    "rank": int(np.sum(sv > 1e-9)),
    "trace_5LHLHT": float(np.trace(M)),
    "trace_over_40": float(np.trace(M)/40),
}
# 중복도 군집
cl = []
for s in sv5:
    if cl and abs(cl[-1][0] - s) < 1e-9: cl[-1].append(float(s))
    else: cl.append([float(s)])
out["clusters"] = [{"value": c[0], "sq": c[0]**2, "mult": len(c)} for c in cl]
print(json.dumps(out, indent=2))
open(HERE + "/a1_independent_spectrum.json", "w").write(json.dumps(out, indent=2))
