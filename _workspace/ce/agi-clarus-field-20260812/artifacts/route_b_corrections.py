# route_b_corrections.py — P1-1 (12_Equation A.2.1-A.2.2) transcription-error correction candidates
# Independent re-implementation from definitions. Deterministic.
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

log = []
def p(s): log.append(s); print(s)

# canonical constants
A_d   = 4.0 / (np.e * np.pi) ** (4.0 / 3.0)
D_eff = 3.0 + A_d * (1.0 - A_d)
# scalar bootstrap fixed point a* (Poisson extinction q*, D=D_eff)
a = 0.5
for _ in range(200):
    a = np.exp(-(1.0 - a) * D_eff)
alpha_s_doc = 0.04865           # doc value used in A.2.1
p_star = np.array([0.0487, 0.2623, 0.6891])  # (a, s, b) canonical constants.py
rho_J = D_eff * a               # multitype Poisson Jacobian radius q*·D (핵심_정리_증명 §14)

p(f"A_d={A_d:.10f}  D_eff={D_eff:.10f}")
p(f"a*={a:.10f}  residual={abs(a-np.exp(-(1-a)*D_eff)):.2e}")
p(f"rho(J)=q*·D = D_eff·a* = {rho_J:.6f}  (doc rho=0.155)")
p(f"alpha_s_doc·D_eff = {alpha_s_doc*D_eff:.6f}  <- the value sitting in B_b slot")

def iterate(Bmap, p0, n=500):
    x = np.array(p0, float)
    for _ in range(n):
        x = Bmap(x)
    return x

def dist(x): return np.max(np.abs(x - p_star))

p("\n== C0: B as written (A.2.1) ==")
def B_doc(q):
    Ba = np.exp(-(1.0 - q[0]) * D_eff)
    Bb = alpha_s_doc * D_eff
    return np.array([Ba, 1.0 - Ba - Bb, Bb])  # (a, s, b)
fp = iterate(B_doc, [1/3, 1/3, 1/3])
p(f"fixed point (a,s,b) = {fp}  d_inf(p*) = {dist(fp):.4f}")

p("\n== C1: component swap s<->b (B_s = alpha_s·D_eff) ==")
def B_swap(q):
    Ba = np.exp(-(1.0 - q[0]) * D_eff)
    Bs = alpha_s_doc * D_eff
    return np.array([Ba, Bs, 1.0 - Ba - Bs])
fp = iterate(B_swap, [1/3, 1/3, 1/3])
p(f"fixed point (a,s,b) = {fp}  d_inf(p*) = {dist(fp):.4f}")

p("\n== C2: linearized map p+ = p* + rho·(p - p*)  [target-aware] ==")
rho = 0.155
x = np.array([1/3, 1/3, 1/3])
for n in range(1, 4):
    x = p_star + rho * (x - p_star)
    p(f"n={n}: (a,s,b)=({x[0]*100:.2f}%, {x[1]*100:.1f}%, {x[2]*100:.1f}%)")
p("doc 7.7 table: n=1 (9.28, 27.3, 63.4) n=2 (5.55, 26.4, 68.1) n=3 (4.98, 26.3, 68.8)")

p("\n== C3: a-only reduction — scalar B_a, contraction |B_a'(a*)| ==")
p(f"B_a'(a*) = D_eff·a* = {rho_J:.6f}; A.2.2 Jacobian slot claims 0.147, B_b slot holds {alpha_s_doc*D_eff:.4f}")
p(f"identity check: 'B_b value' - rho(J) = {alpha_s_doc*D_eff - rho_J:+.6f} (agree to alpha_s rounding)")

p("\n== C4: single clean-constant factor fixes ==")
need_b = p_star[2] / (alpha_s_doc * D_eff)   # factor to turn B_b into 0.6891
need_s = p_star[1] / (alpha_s_doc * D_eff)   # factor to turn slot into 0.2623
p(f"factor needed for b-slot: {need_b:.6f}; for s-slot: {need_s:.6f}")
cands = {
    "e^(3/2)": np.e**1.5, "pi*sqrt2": np.pi*np.sqrt(2), "sqrt(2pi e)": np.sqrt(2*np.pi*np.e),
    "e+pi/2": np.e+np.pi/2, "pi^(4/3)": np.pi**(4/3),
    "e-1": np.e-1, "sqrt(e)": np.sqrt(np.e), "phi_golden": (1+np.sqrt(5))/2,
    "pi/e*sqrt2": np.pi/np.e*np.sqrt(2), "2e/pi": 2*np.e/np.pi, "e*2/3": np.e*2/3,
    "1/(2 a_d)": 1/(2*A_d), "D_eff/2": D_eff/2, "1/(2*rho)": 1/(2*0.155),
}
for k, v in cands.items():
    for tag, need in (("b", need_b), ("s", need_s)):
        rel = abs(v - need) / need
        if rel < 0.02:
            p(f"  {tag}-slot: {k} = {v:.6f} rel.err {rel:.4%}  (NOT exact unless <1e-6)")

p("\n== C5: exponential self-consistency for s,b components? ==")
for tag, ps in (("s", p_star[1]), ("b", p_star[2])):
    D_need = -np.log(ps) / (1.0 - ps)
    p(f"  {tag}* = exp(-(1-{tag}*)·D) requires D = {D_need:.6f}  (no canonical constant equals this)")

p("\n== C6: multitype Poisson (핵심_정리_증명 §14) — degrees of freedom ==")
p("extinction vector q* of 3-type Poisson(A) has 9 free entries for 3 target components;")
p("under-determined: infinitely many A reproduce any q* in (0,1)^3 — not a correction, a new model.")
p("also q* components need not sum to 1, while p* lives on the simplex: object-type mismatch.")

with open("c:/Users/dongh/OneDrive/Desktop/Clarus-Equation/_workspace/ce/agi-clarus-field-20260812/artifacts/route_b_corrections.log", "w", encoding="utf-8") as f:
    f.write("\n".join(log) + "\n")
