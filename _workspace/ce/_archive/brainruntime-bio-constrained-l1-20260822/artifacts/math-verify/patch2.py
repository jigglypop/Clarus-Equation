import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search.py")
src = open(p).read()
old = """        if not all(np.isfinite([s, st, ft, tg, gm])) or ft <= 0: continue"""
new = """        if not all(np.isfinite([s, st, ft, tg, gm])) or ft <= 0: continue
        # identity (I) presupposes cyclic stationarity: require small drift of
        # both total mass and the top-20% mass share over the adult window
        if not (np.isfinite(r.get("drift_ftop", np.nan)) and
                abs(r["drift_ftop"]) < 1.0 and abs(r["R4"]) < 1.0): continue"""
assert old in src and "presupposes" not in src
src = src.replace(old, new, 1)
old2 = '''    stat["sup_R2ad_given_R3a"] = max((r["R2ad_Na"] for r in r3), default=None)'''
new2 = old2 + '''
    stat["R4_stats"] = {"max": float(np.nanmax([r["R4"] for r in rows])),
                        "median": float(np.nanmedian([r["R4"] for r in rows])),
                        "frac_lt_0.5": float(np.nanmean([r["R4"] < 0.5 for r in rows]))}
    stat["E2_cv_stats"] = {"median": float(np.nanmedian([r["E2_cv_rate_proxy"] for r in rows])),
                           "max": float(np.nanmax([r["E2_cv_rate_proxy"] for r in rows])),
                           "frac_skew_gt_0.5": float(np.nanmean([r["E2_skew_rate_proxy"] > 0.5 for r in rows]))}
    stat["E1_stats"] = {"frac_pass": float(np.nanmean([(abs(r["E1_skew_logw"]) < 0.5) and (r["E1_skew_w"] > 1.0) for r in rows]))}'''
assert old2 in src
src = src.replace(old2, new2, 1)
open(p, "w").write(src)
print("patched")
