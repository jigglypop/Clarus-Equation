"""adv_k3b.py -- mod-N degeneracy: eps_e - eps_e2 = N with [D,H_S] != 0. Is E_unseen frame-dependent?
Shows the true criterion is [D, exp(2 pi i H_S/N)] != 0, i.e. Z_N-invariance, not [D,H_S] != 0.
Also: |eps|<N/2 convention is what makes the two coincide.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from check_k3 import Model, P_REG, block_diag_part, comm, opnorm, I2, I4, SX, SZ, NQ  # noqa
rng = np.random.default_rng(20260902)
out = {}
# n_Q=1 block: E + g sx + h sz with eigenvalues E +- r. choose r = N/2? N odd -> take eps = (E+r, E-r) = (7, 0) for N=7: E=3.5, r=3.5 -> not (g,h) integer-friendly but spectrum integer is all that matters
N = 7
E, g, h = 3.5, 2.1, 2.8   # r = 3.5 -> eigenvalues 7 and 0
HS = np.kron(E * I2 + g * SX + h * SZ, NQ)
m = Model(N, HS)
states = [m.random_phys(rng) for _ in range(20)]
mx = 0.0
for c in states:
    for tau in m.taus():
        L1 = m.ledger(c, 1, tau); L2 = m.ledger(c, 2, tau)
        mx = max(mx, abs(L1["E_unseen"] - L2["E_unseen"]))
US = np.diag(np.exp(2j * np.pi * np.linalg.eigvalsh(HS) / N))
W = np.linalg.eigh(HS)[1]
US = W @ US @ W.conj().T
out["degenerate_modN_N7_eps_0_7"] = {"eps": m.eps.tolist(), "comm_D_HS": opnorm(comm(block_diag_part(HS), HS)),
                                     "comm_D_US": opnorm(comm(block_diag_part(HS), US)),
                                     "max_frame_diff_E_unseen_20_states": mx,
                                     "note": "[D,H_S] != 0 yet no frame dependence: eps differ by N so k1 = k1p; criterion is Z_N invariance [D,U_S]=0"}
# control: same H_S shape with eps (6,0): N=13 not degenerate
N2 = 13
m2 = Model(N2, HS)
mx2 = 0.0
for c in [m2.random_phys(rng) for _ in range(20)]:
    for tau in m2.taus():
        L1 = m2.ledger(c, 1, tau); L2 = m2.ledger(c, 2, tau)
        mx2 = max(mx2, abs(L1["E_unseen"] - L2["E_unseen"]))
out["control_same_HS_N13"] = {"eps": m2.eps.tolist(), "max_frame_diff_E_unseen_20_states": mx2}
with open(os.path.join(HERE, "adv_result_b.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=float)
print(json.dumps(out, indent=1, default=float))
