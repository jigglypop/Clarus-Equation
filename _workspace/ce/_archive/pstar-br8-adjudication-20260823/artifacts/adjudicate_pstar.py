# -*- coding: utf-8 -*-
"""Independent recomputation for pstar-br8-adjudication-20260823.

Question: is (active,struct,background)=(0.0487,0.2623,0.6891)
 (H-a) the 4dp rounding of the CE core chain pushed through the ledger
       (docs/geomjeung_wonjang/sangsu_ujuron_wonjang.md) section-3 maps, or
 (H-b) the 4dp rounding of an observation baseline recorded in section 5
       (benchmarks/cosmology/observations_v1.json), or
 neither (UNRESOLVED).

Chain is reimplemented from the formulas (no import of cosmology_registry
for the derivation; a cross-check import is done separately).
Tolerance: |diff| <= 5e-5 per component (contract).
"""
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_HALF_EVEN

getcontext().prec = 60

TUPLE = (Decimal("0.0487"), Decimal("0.2623"), Decimal("0.6891"))  # (active, struct, background)
TOL = Decimal("5e-5")

def solve_q(D):
    """Small root of q = exp(-D(1-q)) via Newton in Decimal."""
    q = Decimal("0.05")
    one = Decimal(1)
    for _ in range(200):
        e = (-D * (one - q)).exp()
        f = q - e
        fp = one - D * e
        step = f / fp
        q = q - step
        if abs(step) < Decimal("1e-55"):
            break
    return q

def r4(x, mode=ROUND_HALF_EVEN):
    return Decimal(x).quantize(Decimal("0.0001"), rounding=mode)

def compare(name, trio):
    """trio = (active, struct, background) as Decimals; compare vs TUPLE."""
    print(f"--- {name}")
    verdict = []
    for label, x, t in zip(("active", "struct", "background"), trio, TUPLE):
        d = x - t
        ok = abs(d) <= TOL
        verdict.append(ok)
        print(f"  {label:10s} chain={x!s:.30s}  r4(HE)={r4(x)}  r4(HU)={r4(x, ROUND_HALF_UP)}  tuple={t}  diff={d:+.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  => component-wise: {'ALL PASS' if all(verdict) else 'MISMATCH'}")
    return all(verdict)

# ---------------------------------------------------------------- chains
alpha_s = Decimal("0.11789")
sW2 = 4 * (Decimal(4) / 3 * alpha_s.ln()).exp()
delta = sW2 * (1 - sW2)
D_exact = 3 + delta
q_exact = solve_q(D_exact)
m_exact = D_exact * q_exact

print("== CE_CORE_EXACT_V1 chain (independent recomputation)")
print(f"  alpha_s = {alpha_s}")
print(f"  sW2     = {sW2}")
print(f"  delta   = {delta}")
print(f"  D       = {D_exact}")
print(f"  q_ext   = {q_exact}")
print(f"  1-q     = {1 - q_exact}")
print(f"  m=D*q   = {m_exact}")
print(f"  residual |q - exp(-D(1-q))| = {abs(q_exact - (-D_exact*(1-q_exact)).exp()):.3e}")

# ledger section 2.1 printed digits (cross-check)
LEDGER = {
    "sW2":   Decimal("0.231222068260755116590565382964"),
    "delta": Decimal("0.177758423409973817923268877860"),
    "q":     Decimal("0.048646719644028206426163828128"),
    "m":     Decimal("0.154587523120074076002229959713"),
}
print("  ledger cross-check |diff|:",
      f"sW2={abs(sW2 - LEDGER['sW2']):.1e}",
      f"delta={abs(delta - LEDGER['delta']):.1e}",
      f"q={abs(q_exact - LEDGER['q']):.1e}",
      f"m={abs(m_exact - LEDGER['m']):.1e}")

D_5dp = Decimal("3.17776")
q_5dp = solve_q(D_5dp)
print("== LEGACY_DELTA_5DP_V1 chain: D=3.17776")
print(f"  q_ext   = {q_5dp}   (ledger cites 0.0486466333372140763...)")
print(f"  |q_exact - q_5dp| = {abs(q_exact - q_5dp):.3e}")

# -------------------------------------------------- section-3 maps
print()
print("== Section 3 maps -> (active, struct, background)")
print("   S3.1 LEGACY_DIRECT_READOUT_V1: q -> Omega_b ONLY (no struct/background map).")
print("   S3.2 C-B-COMP-01: f_b(m) = D*q (fraction WITHIN matter; not a 3-tuple map).")
print("   S3.3 C-B-TRANSIENT-ALG-01: Omega_m = 1/D, Omega_b = q  =>")
print("        active=q, struct=1/D - q, background=1 - 1/D  (flat closure; radiation=0).")
print("   S3.3 SUBSYS variant: Omega_m=(1-Om_r)/D, Omega_b=q(1-Om_r); Om_r is an")
print("        EXTERNAL free input (not fixed in the ledger).")
print()

results = {}
for name, D, q in (("S3.3-ALG on CE_CORE_EXACT_V1", D_exact, q_exact),
                   ("S3.3-ALG on LEGACY_DELTA_5DP_V1", D_5dp, q_5dp)):
    om = 1 / D
    trio = (q, om - q, 1 - om)
    results[name] = compare(name, trio)

# radiation sensitivity (free parameter, declared): Om_r = 9.2e-5 (typical)
Om_r = Decimal("9.2e-5")
om = (1 - Om_r) / D_exact
trio = (q_exact * (1 - Om_r), om - q_exact * (1 - Om_r), 1 - om)
results["S3.3-SUBSYS exact, Om_r=9.2e-5 (free param, sensitivity only)"] = compare(
    "S3.3-SUBSYS exact, Om_r=9.2e-5 (free param, sensitivity only)", trio)

# S3.1 direct readout: only active is defined
print("--- S3.1 direct readout (active only)")
for name, q in (("exact", q_exact), ("5dp", q_5dp)):
    d = q - TUPLE[0]
    print(f"  q_{name} vs active: q={q!s:.15s} r4(HE)={r4(q)} r4(HU)={r4(q, ROUND_HALF_UP)} diff={d:+.3e} {'PASS' if abs(d)<=TOL else 'FAIL'}")

# double-rounding probe: q -> 5dp -> 4dp
q5 = q_exact.quantize(Decimal("0.00001"), rounding=ROUND_HALF_EVEN)
print(f"  double-rounding probe: round5(q_exact)={q5}; round4(HALF_UP)={q5.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)}; round4(HALF_EVEN)={q5.quantize(Decimal('0.0001'), rounding=ROUND_HALF_EVEN)}")

# -------------------------------------------------- section-5 baselines
print()
print("== Section 5 observation baselines (benchmarks/cosmology/observations_v1.json)")
OBS = {
    "Planck2018_base":         dict(obh2=Decimal("0.0224"),   och2=Decimal("0.12"),    H0=Decimal("67.4")),
    "Planck_ACT_SPT_combined": dict(obh2=Decimal("0.02228"),  och2=Decimal("0.1195"),  H0=Decimal("68.43")),
    "ACT_DR6_DESI_reported":   dict(obh2=Decimal("0.0226"),   och2=Decimal("0.118"),   H0=Decimal("68.22")),
    "SPT3G_CMBSPA":            dict(obh2=Decimal("0.022398"), och2=Decimal("0.12028"), H0=Decimal("67.19")),
}
for name, v in OBS.items():
    h2 = (v["H0"] / 100) ** 2
    ob = v["obh2"] / h2
    oc = v["och2"] / h2
    ol = 1 - ob - oc  # flat closure, neutrino/radiation ignored (not recorded in manifest)
    results[f"OBS {name}"] = compare(f"OBS {name} (Ob, Oc, 1-Ob-Oc)", (ob, oc, ol))

# DESI DR2 row has only Omega_m / Omega_Lambda
om_desi, ol_desi = Decimal("0.3027"), Decimal("0.6973")
print("--- OBS DESI_DR2 (Omega_m/Omega_Lambda only)")
print(f"  Omega_m=0.3027 vs active+struct={TUPLE[0]+TUPLE[1]} diff={om_desi-(TUPLE[0]+TUPLE[1]):+.4f}")
print(f"  Omega_L=0.6973 vs background={TUPLE[2]} diff={ol_desi-TUPLE[2]:+.4f}  {'PASS' if abs(ol_desi-TUPLE[2])<=TOL else 'FAIL'}")

print()
print("== SUMMARY")
for k, v in results.items():
    print(f"  {'MATCH  ' if v else 'NOMATCH'} {k}")
