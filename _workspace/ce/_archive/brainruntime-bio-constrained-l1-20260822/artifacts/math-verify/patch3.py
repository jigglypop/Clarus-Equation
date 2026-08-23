import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surrogate.py")
s = open(p).read()
assert "sigma_h" not in s
s = s.replace("def run(p, seed, NE=64, T=716, beta=1.0, w0=1.2, tau_e=2, gain_shape=2.0,",
              "def run(p, seed, NE=64, T=716, beta=1.0, w0=1.2, tau_e=2, gain_shape=2.0,\n"
              "        sigma_h=0.0,", 1)
s = s.replace("    low = np.zeros(P, dtype=np.int16)\n",
              "    low = np.zeros(P, dtype=np.int16)\n"
              "    het = np.ones(P)   # quenched per-edge wake-gain heterogeneity\n", 1)
s = s.replace("                w[idx] = w0; alive[idx] = True; birth[idx] = t; low[idx] = 0",
              "                w[idx] = w0; alive[idx] = True; birth[idx] = t; low[idx] = 0\n"
              "                if sigma_h > 0:\n"
              "                    het[idx] = np.exp(rng.normal(-0.5 * sigma_h ** 2,\n"
              "                                                 sigma_h, size=idx.size))", 1)
s = s.replace("        dw = eta * g * e * alive", "        dw = eta * g * e * het * alive", 1)
open(p, "w").write(s)
print("patched")
