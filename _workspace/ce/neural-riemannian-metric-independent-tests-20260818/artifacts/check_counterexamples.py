"""Deterministic spot checks for the invalid shortcuts in 11-math.md."""
import numpy as np


# Positive rank-one additions make their own direction longer.
g0 = np.eye(2)
u = np.array([1.0, 0.0])
assert u @ (g0 + 2.0 * np.outer(u, u)) @ u > u @ g0 @ u

# Same distance, distinct drift: distance cannot determine deterministic hitting time.
assert 1.0 / 2.0 < 1.0 / 0.5

# The same directed W admits distinct SPD metric maps.
W = np.array([[0.0, 1.0], [0.0, 0.0]])
S = (W + W.T) / 2
g_a = np.eye(2) + S @ S.T
g_b = np.diag([2.0, 1.0])
assert np.all(np.linalg.eigvalsh(g_a) > 0)
assert np.all(np.linalg.eigvalsh(g_b) > 0)
assert not np.allclose(g_a, g_b)

# A nonconstant coordinate expression need not be intrinsically curved.
# For ds^2 = dr^2 + r^2 dtheta^2, Gamma^r_tt = -r and
# Gamma^theta_rt = 1/r. The two nonzero terms in R^r_{theta r theta}
# cancel even though d_r g_theta_theta is nonzero.
r = 2.3
d_r_g_tt = 2.0 * r
gamma_r_tt = -r
gamma_t_rt = 1.0 / r
riemann_r_trt = -1.0 - gamma_r_tt * gamma_t_rt
assert d_r_g_tt != 0.0
assert np.isclose(riemann_r_trt, 0.0)

# E17 S4 at H=1 algebraically reuses S3's Q, including a common ridge.
Q = np.array([[2.0, 0.2], [0.2, 1.0]])
I = np.eye(Q.shape[0])
C_q1 = I @ Q @ I.T
ridge = 0.37
R = np.array([[1.0, 0.1], [0.1, 1.4]])
assert np.allclose(C_q1, Q)
assert np.allclose(C_q1 + ridge * R, Q + ridge * R)
print("counterexample spot checks: PASS")
