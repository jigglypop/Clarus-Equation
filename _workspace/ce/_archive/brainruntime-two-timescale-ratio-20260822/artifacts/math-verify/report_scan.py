import csv, os, json
import numpy as np
d = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(d, 'scan.csv'))))
F = lambda k: np.array([float(r[k]) for r in rows])
R1, R3a, R5, R7 = F('R1'), F('R3a'), F('R5'), F('R7')
R4 = np.abs(F('R4_drift100')); R3b = F('R3b')
R2an, R2bn, R2ap, R2bp = F('R2a_new'), F('R2b_new'), F('R2a_pop'), F('R2b_pop')
th, lam = F('theta'), F('lam0')
q = F('q_ad'); Nad = F('Nad')
g = dict(R1=(R1 >= .02) & (R1 <= .08), R3a=(R3a >= .10) & (R3a <= .25),
         R3b=R3b >= .15, R4=R4 < .05, R5=(R5 >= 1.3) & (R5 <= 1.8), R7=R7 < .005,
         R2a_new=(R2an >= .25) & (R2an <= .45), R2b_new=(R2bn >= .60) & (R2bn <= .85),
         R2a_pop=(R2ap >= .25) & (R2ap <= .45), R2b_pop=(R2bp >= .60) & (R2bp <= .85))
out = {k: int(v.sum()) for k, v in g.items()}
core = g['R1'] & g['R3a'] & g['R3b'] & g['R4'] & g['R5'] & g['R7']
out['core(R1,R3a,R3b,R4,R5,R7)'] = int(core.sum())
out['all_new'] = int((core & g['R2a_new'] & g['R2b_new']).sum())
out['all_pop'] = int((core & g['R2a_pop'] & g['R2b_pop']).sum())
out['R1&R3a'] = int((g['R1'] & g['R3a']).sum())
out['R1&R2b_pop'] = int((g['R1'] & g['R2b_pop']).sum())
out['R3a&R2b_new'] = int((g['R3a'] & g['R2b_new']).sum())
out['R2a_new&R2b_new'] = int((g['R2a_new'] & g['R2b_new']).sum())
out['sup_R3a_given_R1band'] = float(R3a[g['R1']].max()) if g['R1'].any() else None
out['sup_R2b_new_given_R3aband'] = float(R2bn[g['R3a']].max()) if g['R3a'].any() else None
out['inf_R2b_pop_given_R1band'] = float(R2bp[g['R1']].min()) if g['R1'].any() else None
out['max_R3a_overall'] = float(R3a.max())
m = g['R3a']
out['R3a_band_points'] = [dict(theta=float(th[i]), lam0=float(lam[i]), R1=float(R1[i]),
                               R3a=float(R3a[i]), q_ad=float(q[i]), Nad=float(Nad[i]))
                          for i in np.flatnonzero(m)]
# monotonicity R2b>R2a
out['n_R2b_new_gt_R2a_new'] = int((R2bn > R2an).sum())
out['n_R2b_pop_gt_R2a_pop'] = int((R2bp > R2ap).sum())
out['median_R2a_new_minus_R2b_new'] = float(np.nanmedian(R2an - R2bn))
# R1 vs S8_pop empirical relation
ok = np.isfinite(R2bp) & (R1 > 0)
out['corr_logR1_S8pop'] = float(np.corrcoef(np.log(R1[ok] + 1e-9), R2bp[ok])[0, 1])
lo = g['R1']
out['R2b_pop_range_when_R1_in_band'] = [float(R2bp[lo].min()), float(R2bp[lo].max())]
json.dump(out, open(os.path.join(d, 'scan_summary.json'), 'w'), indent=1)
print(json.dumps(out, indent=1)[:4000])
