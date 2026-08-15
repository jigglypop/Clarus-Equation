"""CF-5 independent numerical verification: linear-gate interlacing (field A-SAL).

Theorem being checked: for ANY noise law of u^T eta (survival S nonincreasing),
signal x = z*xi + eta with z = +/-1, linear gate opens iff u^T x > c:
    p_+ = S(c - m),  p_- = S(c + m),  q = S(c),  m = u^T xi
    =>  min(p_+, p_-) <= q <= max(p_+, p_-)          (interlacing)
Corollaries:
    (a) p_+ >= 1-d and p_- >= 1-d and q <= d  impossible for d < 1/2 (static c)
    (b) adaptive threshold c(phi): q_bar >= p_bar_+ + p_bar_- - 1  => d >= 1/3
Contrast: even (energy) gate achieves p_+ = p_- ~ 1 with q ~ 0.

Checks:
  I1  exact Gaussian CDF grid over (m, c): interlacing holds identically
  I2  MC with Laplace / uniform / student-t(3) noise, random u, xi (dim 8)
  I3  adaptive-c version: q_bar >= p_bar_+ + p_bar_- - 1 within MC error
  I4  energy gate separation: p_+ , p_- >= 0.999 with q <= 1e-3 at amp/sd = 8
      (existence proof that the even gate escapes the interlacing bound)
"""
import numpy as np
from math import erf, sqrt

Phi = lambda v: 0.5*(1 + erf(v/sqrt(2)))
S_gauss = lambda v: 1 - Phi(v)

print("== I1: exact Gaussian interlacing grid ==")
worst = 0.0
for m in np.linspace(-3, 3, 61):
    for c in np.linspace(-3, 3, 61):
        p_p, p_m, q = S_gauss(c - m), S_gauss(c + m), S_gauss(c)
        worst = max(worst, min(p_p, p_m) - q, q - max(p_p, p_m))
print(f"max violation over grid = {worst:.3e}")
assert worst <= 1e-15
print("PASS I1")

print("\n== I2: MC interlacing, non-Gaussian noise, dim 8 ==")
rng = np.random.default_rng(5)
M = 400_000
for name, sampler in [("laplace", lambda n: rng.laplace(0, 1, (n, 8))),
                      ("uniform", lambda n: rng.uniform(-1, 1, (n, 8))),
                      ("student-t3", lambda n: rng.standard_t(3, (n, 8)))]:
    bad = 0
    for trial in range(20):
        u = rng.normal(0, 1, 8); xi = rng.normal(0, 1, 8)*1.5
        c = rng.normal(0, 1)*2
        eta = sampler(M)
        ue = eta @ u; m = float(u @ xi)
        p_p = (ue > c - m).mean(); p_m = (ue > c + m).mean(); q = (ue > c).mean()
        se = 3/np.sqrt(M)
        if not (min(p_p, p_m) <= q + se and q <= max(p_p, p_m) + se):
            bad += 1
    print(f"  {name}: violations = {bad}/20")
    assert bad == 0
print("PASS I2")

print("\n== I3: adaptive threshold c = c0 + kappa*phi, phi ~ |N(0,1)| ==")
bad = 0
for trial in range(20):
    u = rng.normal(0, 1, 8); xi = rng.normal(0, 1, 8)*1.5
    c0 = rng.normal(0, 1); kappa = rng.normal(0, 0.5)
    phi = np.abs(rng.normal(0, 1, M)); c = c0 + kappa*phi
    eta = rng.normal(0, 1, (M, 8)); ue = eta @ u; m = float(u @ xi)
    p_p = (ue > c - m).mean(); p_m = (ue > c + m).mean(); q = (ue > c).mean()
    if q + 3/np.sqrt(M) < p_p + p_m - 1:
        bad += 1
print(f"violations of q_bar >= p_+ + p_- - 1: {bad}/20")
assert bad == 0
print("PASS I3")

print("\n== I4: even (energy) gate escapes the bound ==")
w = 4; amp = 1.2; sd = 0.15  # V14-scale separation amp/sd = 8
theta = (amp/2)**2 * w       # energy threshold between noise and signal levels
eta = rng.normal(0, sd, (M, w))
xi = np.ones(w)*amp
e_sig_p = np.square(+xi + eta).sum(1); e_sig_m = np.square(-xi + eta).sum(1)
e_noise = np.square(eta).sum(1)
p_p = (e_sig_p > theta).mean(); p_m = (e_sig_m > theta).mean(); q = (e_noise > theta).mean()
print(f"energy gate: p_+ = {p_p:.6f}, p_- = {p_m:.6f}, q = {q:.2e}")
assert p_p > 0.999 and p_m > 0.999 and q < 1e-3
print("PASS I4 (both-sign detection with vanishing false-open rate: impossible for any linear gate)")
print("\nALL CF-5 CHECKS PASSED")
