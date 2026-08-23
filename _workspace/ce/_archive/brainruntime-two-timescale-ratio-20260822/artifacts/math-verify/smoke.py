import time, numpy as np, sim
for p in [dict(eta=20, lam0=0.4, theta=0.7, rho_inf=0.003, kappa=10, T_m=60),
          dict(eta=5, lam0=0.25, theta=0.8, rho_inf=0.005, kappa=8, T_m=40),
          dict(eta=2, lam0=0.6, theta=0.5, rho_inf=0.01, kappa=5, T_m=80)]:
    t0 = time.time()
    r = sim.run(**p, N_E=2000, T=700, seed=118001)
    g = sim.gates(r)
    print(p, '%.2fs' % (time.time()-t0))
    for k, v in g.items():
        print('   %-16s %.5g' % (k, v))
    print('   M[500],M[699] = %.4g %.4g ; R3a[100],R3a[300],R3a[690] = %.4g %.4g %.4g'
          % (r['M'][500], r['M'][699], r['r3a'][100], r['r3a'][300], r['r3a'][690]))
