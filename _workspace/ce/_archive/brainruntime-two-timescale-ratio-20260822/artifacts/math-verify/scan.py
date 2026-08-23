"""Full dump of the 1200-point LHS scan: all gate metrics, both R2 readings."""
import os
OUTDIR = os.path.dirname(os.path.abspath(__file__))
import csv, time
import numpy as np
import sim
from search import BOX, BANDS, unpack, lhs, band_pen

n = 1200
U = lhs(n, 6, 20260822)
cols = ['eta', 'lam0', 'theta', 'rho_inf', 'kappa', 'T_m', 'R1', 'R2a_new', 'R2b_new',
        'R2a_pop', 'R2b_pop', 'R3a', 'R3b', 'R4_drift100', 'R5', 'R5_rerise', 'R7',
        'Nad', 'wbar', 'q_ad', 'pass_new', 'pass_pop']
t0 = time.time()
with open(os.path.join(OUTDIR,'scan.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(cols)
    npass_new = npass_pop = 0
    both = []
    for i in range(n):
        p = unpack(U[i])
        g = sim.gates(sim.run(N_E=2000, T=700, seed=118001, **p))
        R3b = 1 - p['theta']
        base_ok = (0.02 <= g['R1'] <= 0.08) and (0.10 <= g['R3a'] <= 0.25) and R3b >= 0.15 \
            and abs(g['R4_drift100']) < 0.05 and (1.3 <= g['R5'] <= 1.8) and g['R7'] < 0.005
        pn = base_ok and (0.25 <= g['R2a_new'] <= 0.45) and (0.60 <= g['R2b_new'] <= 0.85)
        pp = base_ok and (0.25 <= g['R2a_pop'] <= 0.45) and (0.60 <= g['R2b_pop'] <= 0.85)
        npass_new += pn; npass_pop += pp
        both.append((g['R1'], g['R3a'], g['q_ad'], p['theta'], p['lam0']))
        wr.writerow([p[k] for k in ['eta', 'lam0', 'theta', 'rho_inf', 'kappa', 'T_m']] +
                    ['%.6g' % g[k] for k in ['R1', 'R2a_new', 'R2b_new', 'R2a_pop', 'R2b_pop', 'R3a']] +
                    ['%.6g' % R3b] + ['%.6g' % g[k] for k in ['R4_drift100', 'R5', 'R5_rerise', 'R7', 'Nad', 'wbar', 'q_ad']] +
                    [int(pn), int(pp)])
b = np.array(both)
print('pass_new=%d pass_pop=%d  time=%.0fs' % (npass_new, npass_pop, time.time() - t0))
m3 = (b[:, 1] >= 0.10) & (b[:, 1] <= 0.25)
print('R3a-in-band candidates: %d' % m3.sum())
for row in b[m3]:
    print('   R1=%.4g R3a=%.4g q_ad=%.4g theta=%.3g lam0=%.3g' % tuple(row))
m1 = (b[:, 0] >= 0.02) & (b[:, 0] <= 0.08)
print('R1-in-band: %d ; their R3a range = [%.4g, %.4g]' % (m1.sum(), b[m1, 1].min(), b[m1, 1].max()))
print('R1-in-band q_ad range = [%.4g, %.4g]' % (b[m1, 2].min(), b[m1, 2].max()))
