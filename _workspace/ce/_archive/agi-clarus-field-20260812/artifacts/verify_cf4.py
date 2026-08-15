"""CF-4 status check: is p* = (0.0487, 0.2623, 0.6891) derivable from field
structure constants?

(1) Recompute the canonical bootstrap scalar a* from first principles:
      A_d = 4/(e*pi)^(4/3),  D_eff = 3 + A_d(1-A_d),
      a* solves a = exp(-(1-a) D_eff)          [12_Equation.md 8.1]
(2) Check the 3-simplex map B of 12_Equation.md A.2.1:
      B(p)_a = exp(-(1-p_a) D_eff), B(p)_b = alpha_s*D_eff, B(p)_s = 1-B_a-B_b
    -> compute its actual fixed point and compare with p*.
(3) Check the Jacobian formula of A.2.2 against the true derivative.
(4) Toy field: occupancy pi_bar as a function of the free thresholds
    (theta_g, theta_s) -> demonstrate pi_bar is threshold-tuned, i.e. no
    parameter-free convergence to p* exists in the CF-1/2/3 dynamics as such.
"""
import numpy as np, math, subprocess, sys

# (1) bootstrap scalar
A_d = 4.0/(math.e**(4/3)*math.pi**(4/3))
D_eff = 3.0 + A_d*(1.0 - A_d)
a = 0.05
for _ in range(200):
    a = math.exp(-(1.0 - a)*D_eff)
resid = a - math.exp(-(1.0 - a)*D_eff)
print(f"(1) A_d={A_d:.10f}  D_eff={D_eff:.10f}  a*={a:.10f}  residual={resid:.2e}")
assert abs(resid) < 1e-12
print(f"    canon ACTIVE_RATIO=0.0487, doc value 0.04865: |a*-0.04865| = {abs(a-0.04865):.2e}")

# (2) the A.2.1 map B and its fixed point
alpha_s = a
p = np.array([0.05, 0.3, 0.65])  # (a, s, b)
for _ in range(500):
    Ba = math.exp(-(1.0 - p[0])*D_eff)
    Bb = alpha_s*D_eff
    Bs = 1.0 - Ba - Bb
    p = np.array([Ba, Bs, Bb])
pstar = np.array([0.0487, 0.2623, 0.6891])
print(f"(2) fixed point of A.2.1 map B: (a,s,b) = ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})")
print(f"    canon p* (a,s,b)           = ({pstar[0]}, {pstar[1]}, {pstar[2]})")
print(f"    |fix(B) - p*|_inf = {np.abs(p - pstar).max():.4f}  <-- map as written does NOT fix p*")

# (3) Jacobian at the a-fixed-point
true_J = D_eff*a                 # d/dp_a exp(-(1-p_a)D_eff) = D_eff * B_a
doc_J = D_eff*a*(1.0 - a)        # formula printed in A.2.2
print(f"(3) true dB_a/dp_a = D_eff*a = {true_J:.6f};  doc formula D_eff*a*(1-a) = {doc_J:.6f}")
print(f"    both < 1 (local contraction in a holds) but the doc formula has a spurious (1-a) factor")

# (4) threshold sensitivity of toy-field occupancy (reuses verify_cf3 dynamics)
sys.path.insert(0, __file__.rsplit("verify_cf4.py", 1)[0].rstrip("/\\") or ".")
from verify_cf3 import run  # noqa: E402
print("(4) pi_bar(T=20000) vs free thresholds (toy field, N=16):")
rows = []
for tg, ts, sp in [(0.5, 0.5, 0.0487), (0.5, 0.5, 0.12), (0.5, 0.5, 0.30),
                   (0.5, 0.9, 0.12), (0.5, 1.2, 0.12), (0.9, 1.2, 0.30)]:
    pi, _, _ = run(20000, seed=11, s_init_scale=0.9, theta_g=tg, theta_s=ts, sig_p=sp)
    v = pi[20000]
    rows.append((tg, ts, sp, v))
    print(f"    theta_g={tg:.2f} theta_s={ts:.2f} sig_p={sp:.3f} -> pi_bar = "
          f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})   |pi-p*|_inf = {np.abs(v-pstar).max():.4f}")
spread = max(np.abs(r1[3]-r2[3]).max() for r1 in rows for r2 in rows)
print(f"    occupancy spread across threshold choices (L_inf) = {spread:.4f}")
print("    -> pi_A tracks the exogenous signal rate sig_p (input statistic), and")
print("       pi_S/pi_F split tracks the free threshold theta_s vs the phi scale; no")
print("       parameter-free self-convergence to p* is present in the CF-1/2/3 field itself.")
