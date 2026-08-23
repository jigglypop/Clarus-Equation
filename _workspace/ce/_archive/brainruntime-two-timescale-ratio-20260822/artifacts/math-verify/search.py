"""(b) simultaneous-satisfaction search over the 6 free parameters.
Coarse Latin-hypercube (fixed seed) + local refinement. Reports best margins
and which gate pairs are jointly unreachable."""
import os
OUTDIR = os.path.dirname(os.path.abspath(__file__))
import json, sys, time
import numpy as np
import sim

BANDS = dict(R1=(0.02, 0.08), R2a=(0.25, 0.45), R2b=(0.60, 0.85),
             R3a=(0.10, 0.25), R5=(1.3, 1.8))
# R3b: theta<=0.85 ; R4: |drift100|<0.05 ; R7: <0.005 ; R6: >=1.3 (refine only)

BOX = [('eta', -1.0, 2.5, 'log'), ('lam0', -2.0, -0.02, 'log'),
       ('theta', 0.02, 0.85, 'lin'), ('rho_inf', -4.0, -0.5, 'log'),
       ('kappa', 0.0, 40.0, 'lin'), ('T_m', 0.7, 2.4, 'log')]


def unpack(u):
    p = {}
    for (name, lo, hi, kind), x in zip(BOX, u):
        v = lo + (hi - lo) * x
        p[name] = 10.0 ** v if kind == 'log' else v
    return p


def band_pen(v, lo, hi):
    if not np.isfinite(v):
        return 25.0
    v = max(v, 1e-9)
    if v < lo:
        return (np.log(lo / v)) ** 2
    if v > hi:
        return (np.log(v / hi)) ** 2
    return 0.0


def evaluate(p, N_E=2000, seed=118001, T=700, reading='new'):
    r = sim.run(N_E=N_E, T=T, seed=seed, **p)
    g = sim.gates(r)
    g['R2a'] = g['R2a_' + reading]
    g['R2b'] = g['R2b_' + reading]
    pen = {k: band_pen(g[k], *BANDS[k]) for k in BANDS}
    pen['R3b'] = 0.0 if (1 - p['theta']) >= 0.15 else 25.0
    pen['R4'] = 0.0 if abs(g['R4_drift100']) < 0.05 else (abs(g['R4_drift100']) / 0.05 - 1) ** 2
    pen['R7'] = 0.0 if g['R7'] < 0.005 else (np.log(max(g['R7'], 1e-9) / 0.005)) ** 2
    pen['mono'] = 0.0 if g['R2b'] > g['R2a'] else 1.0
    g['loss'] = float(sum(pen.values()))
    g['pen'] = {k: float(v) for k, v in pen.items()}
    return g


def lhs(n, d, seed):
    rng = np.random.default_rng(seed)
    return (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n


def main():
    reading = sys.argv[1] if len(sys.argv) > 1 else 'new'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    U = lhs(n, 6, 20260822)
    rows = []
    t0 = time.time()
    for i in range(n):
        p = unpack(U[i])
        g = evaluate(p, reading=reading)
        rows.append((g['loss'], p, {k: float(g.get(k, np.nan)) for k in
                     ['R1', 'R2a', 'R2b', 'R3a', 'R4_drift100', 'R5', 'R7', 'Nad', 'wbar', 'q_ad']}, g['pen']))
        if (i + 1) % 200 == 0:
            print('  %d/%d  %.0fs  best=%.4g' % (i + 1, n, time.time() - t0, min(r[0] for r in rows)), flush=True)
    rows.sort(key=lambda r: r[0])
    # local refinement: random-walk descent around top 5
    rng = np.random.default_rng(7)
    refined = []
    for loss0, p0, _, _ in rows[:5]:
        u = np.array([[x for x in U[j]] for j in range(n)])  # placeholder
        best_p, best = dict(p0), loss0
        scale = 0.25
        for it in range(120):
            q = {k: v * float(np.exp(rng.normal(0, scale))) for k, v in best_p.items()}
            q['theta'] = min(max(q['theta'], 0.02), 0.85)
            q['kappa'] = min(q['kappa'], 200.0)
            q['rho_inf'] = min(q['rho_inf'], 0.6)
            g = evaluate(q, reading=reading)
            if g['loss'] < best:
                best, best_p = g['loss'], q
            if it % 40 == 39:
                scale *= 0.5
        g = evaluate(best_p, reading=reading)
        refined.append((best, best_p, {k: float(g.get(k, np.nan)) for k in
                        ['R1', 'R2a', 'R2b', 'R3a', 'R4_drift100', 'R5', 'R7', 'Nad', 'wbar', 'q_ad']}, g['pen']))
    refined.sort(key=lambda r: r[0])
    out = dict(reading=reading, n=n,
               coarse_best=[dict(loss=r[0], params=r[1], metrics=r[2], pen=r[3]) for r in rows[:10]],
               refined=[dict(loss=r[0], params=r[1], metrics=r[2], pen=r[3]) for r in refined],
               gate_pass_counts={k: int(sum(1 for r in rows if r[3].get(k, 1) == 0)) for k in
                                 list(BANDS) + ['R3b', 'R4', 'R7', 'mono']},
               max_R3a=float(max(r[2]['R3a'] for r in rows)),
               min_R1=float(min(r[2]['R1'] for r in rows)),
               n_R1_pass=int(sum(1 for r in rows if r[3]['R1'] == 0)),
               n_R1_and_R3a=int(sum(1 for r in rows if r[3]['R1'] == 0 and r[3]['R3a'] == 0)))
    with open(os.path.join(OUTDIR,'search_%s.json' % reading), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ['gate_pass_counts', 'max_R3a', 'min_R1', 'n_R1_pass', 'n_R1_and_R3a']}, indent=1))
    print('BEST refined loss=%.4g' % refined[0][0])
    print(json.dumps(refined[0][1], indent=1, default=float))
    print(json.dumps(refined[0][2], indent=1, default=float))
    print(json.dumps(refined[0][3], indent=1, default=float))
    print('%.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
