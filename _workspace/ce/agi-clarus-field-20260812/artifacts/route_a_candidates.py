# route_a_candidates.py — CF-4 detour routes for p*_A self-convergence
#  R-A1: bootstrap-residual homeostat (threshold adaptation, comparator = canonical B_a residual)
#  R-A2: maxent/Gibbs with A.2.3 energy-cost ratios (over-determination check)
#  R-A3: percolation/branching re-export (a* = non-giant fraction of Poisson(D_eff) graph)
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
rng = np.random.default_rng(20260812)
log = []
def p(s): log.append(s); print(s)

A_d   = 4.0 / (np.e * np.pi) ** (4.0 / 3.0)
D_eff = 3.0 + A_d * (1.0 - A_d)
def a_star(D):
    a = 0.5
    for _ in range(500): a = np.exp(-(1.0 - a) * D)
    return a
AST = a_star(D_eff)
p(f"D_eff={D_eff:.6f}  a*(D_eff)={AST:.6f}")

# ---------------- R-A1: homeostat toy, N=16 field ----------------
p("\n== R-A1: bootstrap-residual homeostat, N=16, w=8 ==")
N, w, T = 16, 8, 200_000
xi = rng.standard_normal(w); xi /= np.linalg.norm(xi)
m  = np.ones(w)
amp, sig_n = 3.0, 1.0

def run(r_sig, D, adapt=True, T=T, eta=0.05, seed=1):
    r = np.random.default_rng(seed)
    theta = 5.0          # initial gate threshold (energy units)
    pi_hat = 0.5         # EMA of open fraction
    ema = 0.005
    occ = []
    for t in range(T):
        is_sig = r.random(N) < r_sig
        z = np.where(r.random(N) < 0.5, 1.0, -1.0)
        x = sig_n * r.standard_normal((N, w))
        x += (is_sig * z * amp)[:, None] * xi[None, :]
        e = np.sum((m[None, :] * x) ** 2, axis=1)
        open_frac = np.mean(e > theta)
        pi_hat = (1 - ema) * pi_hat + ema * open_frac
        if adapt:
            R = pi_hat - np.exp(-(1.0 - pi_hat) * D)   # bootstrap residual, no 0.0487 anywhere
            theta *= np.exp(eta * R)                    # log-space update, dpi/dtheta<0
        if t >= T // 2: occ.append(open_frac)
    return float(np.mean(occ)), theta

for r_sig in (0.049, 0.12, 0.30):
    pi_A, th = run(r_sig, D_eff, adapt=True)
    p(f"  homeostat  r_sig={r_sig:5.3f}: pi_A={pi_A:.5f}  (target a*={AST:.5f}, |err|={abs(pi_A-AST):.5f})  theta_final={th:.3f}")
for r_sig in (0.049, 0.30):
    pi_A, th = run(r_sig, D_eff, adapt=False)
    p(f"  fixed-thr  r_sig={r_sig:5.3f}: pi_A={pi_A:.5f}  (tracks input, no convergence)")
# cross-prediction: comparator dimension changed -> pi_A must move to a*(D')
for Dp in (2.0, 5.0):
    pi_A, _ = run(0.12, Dp, adapt=True)
    p(f"  D'={Dp}: pi_A={pi_A:.5f} vs a*(D')={a_star(Dp):.5f}  |err|={abs(pi_A-a_star(Dp)):.5f}")

# ---------------- R-A2: maxent/Gibbs with A.2.3 costs ----------------
p("\n== R-A2: maxent + linear energy budget, costs C=(1,5.4,14.1) (A.2.3 cond.4) ==")
C = np.array([1.0, 5.4, 14.1])
pstar = np.array([0.0487, 0.2623, 0.6891])
p(f"  note: C ratios vs p* ratios: p_s/p_a={pstar[1]/pstar[0]:.3f} (5.4), p_b/p_a={pstar[2]/pstar[0]:.3f} (14.1) — C is back-computed from p*")
# Gibbs pi_i = exp(-beta C_i)/Z ; fit beta on pi_s/pi_a, predict pi_b/pi_s
beta = -np.log(pstar[1] / pstar[0]) / (C[1] - C[0])
pred_bs = np.exp(-beta * (C[2] - C[1]))
p(f"  beta fitted on s/a ratio: {beta:+.4f} (negative = anti-thermal)")
p(f"  predicted pi_b/pi_s = {pred_bs:.3f}  vs actual p*_b/p*_s = {pstar[2]/pstar[1]:.3f}  -> over-determination FAILS (x{pred_bs/(pstar[2]/pstar[1]):.1f})")

# ---------------- R-A3: percolation re-export ----------------
p("\n== R-A3: a* = extinction/non-giant fraction, ER(Poisson) graph check ==")
def giant_fraction(n, c, r):
    # ER G(n, p=c/n) giant component via union-find
    parent = np.arange(n)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    m_edges = r.poisson(c * n / 2)
    u = r.integers(0, n, m_edges); v = r.integers(0, n, m_edges)
    for a_, b_ in zip(u, v):
        ra, rb = find(a_), find(b_)
        if ra != rb: parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    _, counts = np.unique(roots, return_counts=True)
    return counts.max() / n
for c in (2.0, 2.5, D_eff, 4.0):
    S = np.mean([giant_fraction(20000, c, np.random.default_rng(s)) for s in range(4)])
    p(f"  mean degree c={c:.4f}: non-giant fraction = {1-S:.5f} vs theory a*(c) = {a_star(c):.5f}")

with open("c:/Users/dongh/OneDrive/Desktop/Clarus-Equation/_workspace/ce/agi-clarus-field-20260812/artifacts/route_a_candidates.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log) + "\n")
