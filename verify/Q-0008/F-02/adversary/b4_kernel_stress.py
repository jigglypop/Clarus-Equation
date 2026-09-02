"""Adversary b4: stress the kernel law eps^2 = eps_star^2 ||H kappa H||_F^2 / n^2.

(1) NON-GAUSSIAN labels (Rademacher +-1 entries): the Wick step acquires a fourth-cumulant term
    (E xi^4 - 3) sum_v (H kappa H)_vv^2, which is the SAME order as ||H kappa H||_F^2 for kappa = I
    (both O(n)) but subleading for the heritable kernel (O(n^2) vs O(n^3)).  Prediction: eps_star
    becomes mode-dependent and the her/iid ratio shifts.  -> is the Gaussian scope assumption load-bearing?
(2) ANISOTROPIC labels (support on a single tetrad entry): breaks the internal-isotropy premise,
    E[tl gram] != 0, so a delta^2 FLOOR appears and gamma_iid -> 0.  -> the premise is an assumption
    about P_micro, not a theorem about tetrad perturbations.
(3) OFF-GRID replication of K1 and K2 at n = 12, 24 (NOT the pre-registered grid 8..128, NOT the
    pre-registered seed): does the law already hold before the kill is run?
"""
import math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, simplicity_residual
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

REF = geometric_self_dual_triple(np.eye(4))
DELTA = 0.005
ADV_SEED = 424242            # deliberately NOT the pre-registered 20260902

def cell(lab, d=DELTA):
    return optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + d * lab)).aligned_candidate
def block(labels, d=DELTA):
    return simplicity_residual(sum(cell(l, d) for l in labels))
def rms(v):
    a = np.asarray(v, float); return float(np.sqrt(np.mean(a * a)))
def fit(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])

# ---- own tree sampler + O(n) driver (independent of driver_numbers.py) --------------------------
def cayley(n, rng):
    import heapq
    if n <= 2: return [-1] + [0] * (n - 1)
    seq = rng.integers(0, n, size=n - 2); deg = np.ones(n, dtype=int)
    for s in seq: deg[s] += 1
    adj = [[] for _ in range(n)]; leaves = [i for i in range(n) if deg[i] == 1]; heapq.heapify(leaves)
    for s in seq:
        lf = heapq.heappop(leaves); adj[lf].append(int(s)); adj[int(s)].append(lf); deg[s] -= 1
        if deg[s] == 1: heapq.heappush(leaves, int(s))
    u = heapq.heappop(leaves); v = heapq.heappop(leaves); adj[u].append(v); adj[v].append(u)
    root = int(rng.integers(0, n)); parent = [-2] * n; parent[root] = -1; st = [root]
    while st:
        x = st.pop()
        for y in adj[x]:
            if parent[y] == -2: parent[y] = x; st.append(y)
    return parent

def tree_quant(parent):
    n = len(parent); ch = [[] for _ in range(n)]; root = -1
    for v, p in enumerate(parent):
        if p >= 0: ch[p].append(v)
        else: root = v
    order = [root]; i = 0
    while i < len(order): order.extend(ch[order[i]]); i += 1
    depth = np.zeros(n, dtype=np.int64); sub = np.ones(n, dtype=np.int64); pre = np.zeros(n, dtype=np.int64)
    for v in order[1:]: depth[v] = depth[parent[v]] + 1
    for v in reversed(order):
        if parent[v] >= 0: sub[parent[v]] += sub[v]
    for v in order: pre[v] = sub[v] + (pre[parent[v]] if parent[v] >= 0 else 0)
    s = sub.astype(float)
    w2 = float(np.sum(s * s)); w2p = float(np.sum((2 * depth + 1) * s * s)); srow = float(np.sum(pre.astype(float) ** 2))
    D = w2p - 2 * srow / n + w2 * w2 / (n * n)
    trHk = float(np.sum(s * (1 - s / n)))
    return D, trHk

def her_labels(parent, xi):
    n = len(parent); ch = [[] for _ in range(n)]; root = -1
    for v, p in enumerate(parent):
        if p >= 0: ch[p].append(v)
        else: root = v
    order = [root]; i = 0
    while i < len(order): order.extend(ch[order[i]]); i += 1
    lab = np.zeros_like(xi)
    for v in order:
        lab[v] = xi[v] + (lab[parent[v]] if parent[v] >= 0 else 0.0)
    return lab

print("== exact tree averages at the OFF-GRID sizes (own MC, 120k trees) ==")
rng = np.random.default_rng(9)
EX = {}
for n in (12, 24):
    D = []; T = []
    for _ in range(120000):
        d, t = tree_quant(cayley(n, rng)); D.append(d); T.append(t)
    EX[n] = (float(np.mean(D)), float(np.mean(T)))
    print(f"   n={n}: E[D]={EX[n][0]:.3f}  E[tr(H kappa)]={EX[n][1]:.4f}"
          f"   => her/iid = {math.sqrt(EX[n][0]/(n-1)):.4f}   X(n) = {2*EX[n][1]/math.sqrt((n-1)*EX[n][0]):.4f}")

TR = 500
print(f"\n== (3) off-grid K1: RMS_her/RMS_iid  (seed {ADV_SEED}, {TR} trials, delta={DELTA}) ==")
for n in (12, 24):
    rh = np.random.default_rng(ADV_SEED); ri = np.random.default_rng(ADV_SEED + 1)
    her = rms([block(her_labels(cayley(n, rh), rh.normal(size=(n, 4, 4)))) for _ in range(TR)])
    iid = rms([block(ri.normal(size=(n, 4, 4))) for _ in range(TR)])
    pred = math.sqrt(EX[n][0] / (n - 1))
    print(f"   n={n:<3} observed {her/iid:8.4f}   card law {pred:8.4f}   ratio {her/iid/pred:6.4f}"
          f"   (MC se ~ {1/math.sqrt(TR):.3f} rel)")

print(f"\n== (3b) off-grid K2: mixing excess X(n) (common random numbers) ==")
for n in (12, 24):
    r = np.random.default_rng(ADV_SEED + 7)
    ei, eh, em = [], [], []
    for _ in range(TR * 2):
        p = cayley(n, r); xi = r.normal(size=(n, 4, 4)); ze = r.normal(size=(n, 4, 4))
        h = her_labels(p, ze)
        ei.append(block(xi)); eh.append(block(h)); em.append(block(xi + h))
    ri_, rh_, rm_ = rms(ei), rms(eh), rms(em)
    X = (rm_ ** 2 - ri_ ** 2 - rh_ ** 2) / (ri_ * rh_)
    print(f"   n={n:<3} observed X = {X:7.4f}   card law 2E tr(Hk)/sqrt((n-1)E D) = {2*EX[n][1]/math.sqrt((n-1)*EX[n][0]):7.4f}")

print("\n== (1) NON-GAUSSIAN labels (Rademacher +-1 entries), n = 16 ==")
for n in (16,):
    r = np.random.default_rng(ADV_SEED + 11)
    iid_g = rms([block(r.normal(size=(n, 4, 4))) for _ in range(TR)])
    r = np.random.default_rng(ADV_SEED + 11)
    iid_r = rms([block(r.integers(0, 2, size=(n, 4, 4)) * 2.0 - 1.0) for _ in range(TR)])
    r = np.random.default_rng(ADV_SEED + 12)
    her_g = rms([block(her_labels(cayley(n, r), r.normal(size=(n, 4, 4)))) for _ in range(TR)])
    r = np.random.default_rng(ADV_SEED + 12)
    her_r = rms([block(her_labels(cayley(n, r), r.integers(0, 2, size=(n, 4, 4)) * 2.0 - 1.0)) for _ in range(TR)])
    print(f"   iid   Gaussian {iid_g:.5e}   Rademacher {iid_r:.5e}   ratio {iid_r/iid_g:.4f}")
    print(f"   her   Gaussian {her_g:.5e}   Rademacher {her_r:.5e}   ratio {her_r/her_g:.4f}")
    print(f"   her/iid  Gaussian {her_g/iid_g:.4f}   Rademacher {her_r/iid_r:.4f}"
          f"   (kernel law is blind to the label law; both should equal the same number)")

print("\n== (2) ANISOTROPIC labels: only the (0,0) tetrad entry fluctuates (isotropy premise broken) ==")
sizes = (4, 8, 16, 32)
r = np.random.default_rng(ADV_SEED + 21)
vals = []
for n in sizes:
    acc = []
    for _ in range(TR):
        lab = np.zeros((n, 4, 4)); lab[:, 0, 0] = r.normal(size=n)
        acc.append(block(lab))
    vals.append(rms(acc))
    print(f"   n={n:<3} RMS = {vals[-1]:.5e}   isotropic law would give eps_star sqrt(n-1)/n")
print(f"   measured slope = {fit(sizes, vals):+.4f}   isotropic prediction = {fit(sizes,[math.sqrt(n-1)/n for n in sizes]):+.4f}")
print("   (a nonzero E[tl gram] gives an n-independent floor -> slope -> 0; K5's window [-0.58,-0.38] would fire)")
