import sys
sys.path.insert(0, "reality_stone/python")
from reality_stone.clarus import cosmology_registry as cr

e = cr.CE_CORE_EXACT_V1
l = cr.LEGACY_DELTA_5DP_V1
r = cr.LEGACY_ROUNDED_RUNTIME_V1
print("exact: sW2=%.17g delta=%.17g D=%.17g q=%.17g" % (e.sin2_theta_w, e.delta, e.d_eff, e.q_ext))
print("5dp:   delta=%.17g D=%.17g q=%.17g" % (l.delta, l.d_eff, l.q_ext))
print("runtime:", r.active_ratio, r.struct_ratio, r.background_ratio, "raw_sum=%.6f" % r.raw_sum)
print("q_exact |diff| vs independent 0.048646719644028206:", abs(e.q_ext - 0.048646719644028206))
print("q_5dp   |diff| vs independent 0.0486466333372140763:", abs(l.q_ext - 0.0486466333372140763))
