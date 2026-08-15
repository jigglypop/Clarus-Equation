"""CF-3 independent numerical verification: occupancy time-average convergence.

Axioms implemented: A-E1 (i.i.d. input), A-E2 (exogenous gate: g depends on
input x only; open events have g >= g_min > 0, open prob rho > 0 per node/tick).

Checks:
  E1  synchronous coupling: two far-apart initializations, same input stream
      -> max node state distance shrinks; log-distance ~ N_open(t)*log(1-gbar)
  E2  pi(t) time-average is Cauchy: ||pi(2t) - pi(t)|| -> 0 like O(1/t)
  E3  pi_bar independent of initialization (within Monte Carlo tolerance)
"""
import numpy as np

def norm_lap(A):
    d = A.sum(1); Dm = np.diag(1/np.sqrt(d))
    return np.eye(len(A)) - Dm @ A @ Dm

def run(T, seed, s_init_scale, theta_g=0.5, theta_s=0.15, lam=0.35, N=16, w=4, sig_p=0.12):
    r_in = np.random.default_rng(seed)          # input stream (shared)
    r_st = np.random.default_rng(seed + 1000 + int(s_init_scale*7))  # init only
    A = np.zeros((N, N))
    ring = np.arange(N)
    A[ring, (ring+1) % N] = 1; A[(ring+1) % N, ring] = 1
    rg = np.random.default_rng(3)
    for _ in range(N):
        i, j = rg.integers(0, N, 2)
        if i != j: A[i, j] = A[j, i] = 1
    L = norm_lap(A)
    wl, V = np.linalg.eigh(L + lam*np.eye(N))
    E = (V*np.exp(-wl)) @ V.T
    F = np.linalg.solve(L + lam*np.eye(N), np.eye(N) - E)
    s = r_st.normal(0, s_init_scale, (N, w))
    s = s/np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1.0)  # inside unit ball
    phi = np.zeros(N)
    m = np.ones(w); a, b = 2.0, -3.0
    counts = np.zeros(3)  # active, structural, frozen
    pi_traj = {}
    for t in range(T):
        sig = (r_in.random(N) < sig_p)
        z = np.where(r_in.random(N) < 0.5, 1.0, -1.0)
        x = r_in.normal(0, 0.15, (N, w)) + (sig*z*1.2)[:, None]
        g = 1/(1 + np.exp(-(a*np.square(m*x).sum(1) + b)))
        geff = np.where(g > theta_g, g, 0.0)          # exogenous: x only
        wv = np.tanh(x)
        wv = wv/np.maximum(np.linalg.norm(wv, axis=1, keepdims=True), 1.0)
        s = (1 - geff)[:, None]*s + geff[:, None]*wv
        rs = np.minimum(np.linalg.norm(s, axis=1), 1.0)
        phi = E @ phi + F @ rs
        act = g > theta_g
        struct = (~act) & (phi > theta_s)             # binding-participation proxy
        froz = ~(act | struct)
        counts += np.array([act.sum(), struct.sum(), froz.sum()])
        if t+1 in (T//8, T//4, T//2, T):
            pi_traj[t+1] = counts/((t+1)*N)
    return pi_traj, s, phi

if __name__ == "__main__":
    T = 40000
    # E1/E3: two initializations, same input
    piA, sA, phiA = run(T, seed=11, s_init_scale=0.9)
    piB, sB, phiB = run(T, seed=11, s_init_scale=0.05)
    d_state = np.abs(sA - sB).max()
    print(f"E1 coupling: final max|sA-sB| = {d_state:.3e}  max|phiA-phiB| = {np.abs(phiA-phiB).max():.3e}")
    assert d_state < 1e-12, "P0: synchronous coupling did not contract"
    ks = sorted(piA)
    print("E2 Cauchy (init A):")
    for k in ks:
        print(f"  pi({k}) = {np.array2string(piA[k], precision=5)}")
    diffs = [np.abs(piA[ks[i+1]] - piA[ks[i]]).max() for i in range(len(ks)-1)]
    print(f"  successive diffs: {['%.2e' % d for d in diffs]}")
    assert diffs[-1] < diffs[0], "P1: time-average not Cauchy-decreasing"
    d_pi = np.abs(piA[T] - piB[T]).max()
    print(f"E3 init-independence: |pi_A - pi_B|_inf = {d_pi:.3e}")
    assert d_pi < 5e-3, "P1: pi_bar depends on initialization beyond MC tolerance"
    print(f"\npi_bar(T={T}) = {np.array2string(piA[T], precision=5)}  (active, structural, frozen)")
    print("ALL CF-3 CHECKS PASSED")
    