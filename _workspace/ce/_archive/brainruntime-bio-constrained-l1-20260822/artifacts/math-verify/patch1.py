import os, re
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surrogate.py")
src = open(p).read()
anchor = '    N_t = L["N_t"]\n'
add = '''    ftop_series = L["ftop_t"][sl]
    gf = np.isfinite(ftop_series)
    o["drift_ftop"] = (float(np.polyfit(days[gf], ftop_series[gf], 1)[0] * 100.0
                             / max(np.nanmean(ftop_series), 1e-12))
                       if gf.sum() > 20 else np.nan)
    Nser = L["N_t"][sl]
    gn = Nser > 0
    o["drift_N"] = (float(np.polyfit(days[gn], Nser[gn], 1)[0] * 100.0 / Nser[gn].mean())
                    if gn.sum() > 20 else np.nan)
'''
assert anchor in src and "drift_ftop" not in src
src = src.replace(anchor, add + anchor, 1)
open(p, "w").write(src)
print("patched")
