import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jac.py")
s = open(p).read()
s = s.replace("""noise = np.std(""", """bad = ~np.isfinite(J).all(axis=1)
G2 = [g for g, bb in zip(G, bad) if not bb]
J = J[~bad]
noise_all = np.std(""", 1)
s = s.replace("""np.array([vals(P, sd) for sd in [SEED] + SEEDS_NOISE]), axis=0, ddof=1)""",
              """np.array([vals(P, sd) for sd in [SEED] + SEEDS_NOISE]), axis=0, ddof=1)
noise = noise_all[~bad]""", 1)
s = s.replace('"gates": G,', '"gates_used": G2, "gates_dropped_nonfinite": [g for g, bb in zip(G, bad) if bb],', 1)
s = s.replace('{k: float(v) for k, v in zip(G, noise)}', '{k: float(v) for k, v in zip(G2, noise)}', 1)
s = s.replace('for i, g in enumerate(G)}', 'for i, g in enumerate(G2)}', 1)
open(p, "w").write(s)
print("ok")
