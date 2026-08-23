import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S
p = dict(eta=1.5, lam0=0.5, kappa=8.0, rho_inf=2e-3, kappa_m=20.0, T_m=60.0,
         Sstar=200.0, g1g0=4.0)
t = time.time()
m = S.run(p, 119001)
print("%.2f s" % (time.time() - t))
print(json.dumps(m, indent=1, default=float))
print(S.gates_pass(m))
print("loss", S.loss(m))
