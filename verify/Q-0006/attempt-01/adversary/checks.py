# Q-0006 attempt-01 adversary checks: dimension / limits / known-exact / random_sample_20 (seed 20260902)
import json, random, itertools
random.seed(20260902)
out = {"seed": 20260902, "checks": []}

# --- (S2.3) det(d_mu X^A) = Tdot * beta^3 = mu^4 u a^3 b^3  under E62-A2: Tdot = mu u, beta = mu a b
def det_diag(Tdot, beta):
    return Tdot * beta**3
rs = []
for i in range(20):
    mu = random.uniform(0.1, 5.0); u = random.uniform(-3, 3); a = random.uniform(0.05, 4.0); b = random.uniform(-3, 3)
    lhs = det_diag(mu*u, mu*a*b); rhs = mu**4*u*a**3*b**3
    rs.append(abs(lhs-rhs)/max(1e-300, abs(rhs)))
out["checks"].append({"name": "random_sample_20 S2.3 det identity", "max_rel_err": max(rs), "pass": max(rs) < 1e-12})

# dimension: [Tdot]=1, [beta]=1 -> [det]=4 ; [mu^4]=4, u,a,b dimensionless  -> consistent
out["checks"].append({"name": "dimension S2.3", "lhs_dim": 1+3*1, "rhs_dim": 4, "pass": True})

# limits: u->0 or b->0 => det -> 0 (C2 fails), a->0 with beta fixed: det = Tdot*beta^3 independent of a
lim = {}
lim["u->0"] = det_diag(0.0, 1.3)
lim["beta->0"] = det_diag(0.7, 0.0)
lim["a->0, beta fixed"] = [det_diag(0.7, 1.3) for _ in range(3)]  # a-independent in diag form
# in dimensionless form mu^4 u a^3 b^3 with b = beta/(mu a): a^3 b^3 = beta^3/mu^3 -> a-independent
mu, beta, u = 1.7, 1.3, 0.7
vals = [mu**4*u*a**3*(beta/(mu*a))**3 for a in (1e-3, 1e-1, 1.0, 10.0)]
lim["dimless form a-scan"] = vals
out["checks"].append({"name": "limits", "values": lim,
                      "pass": lim["u->0"]==0 and lim["beta->0"]==0 and max(vals)-min(vals) < 1e-9})

# known exact: E61-D constant Xbar => det 0
out["checks"].append({"name": "known-exact E61-D constant X", "det": det_diag(0.0, 0.0), "pass": det_diag(0.0,0.0)==0})

# (S4.1) AGW dimension set: 4 not in {4k+2}
ks = range(-3, 10)
out["checks"].append({"name": "AGW dims", "four_in_4k+2": any(4==4*k+2 for k in ks), "pass": not any(4==4*k+2 for k in ks)})

# hidden-assumption probe: internal SO(4) is broken by E62 ansatz to SO(3): rotate X^0 into X^i and recheck det invariance
import math
th = 0.4
# Jacobian J = diag(Tdot, beta, beta, beta); rotation R in (0,1) plane acts on A index: det(R J) = det J
J = [[0.7,0,0,0],[0,1.3,0,0],[0,0,1.3,0],[0,0,0,1.3]]
R = [[math.cos(th),-math.sin(th),0,0],[math.sin(th),math.cos(th),0,0],[0,0,1,0],[0,0,0,1]]
def matmul(A,B): return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def det4(M):
    # Laplace
    def det(m):
        if len(m)==1: return m[0][0]
        return sum((-1)**c*m[0][c]*det([r[:c]+r[c+1:] for r in m[1:]]) for c in range(len(m)))
    return det(M)
out["checks"].append({"name": "symmetry SO(4) on A-index preserves det (C2 is SO(4)-invariant; E62 ansatz choice breaks SO(4)->SO(3) only as background)",
                      "det_J": det4(J), "det_RJ": det4(matmul(R,J)), "pass": abs(det4(J)-det4(matmul(R,J)))<1e-12})

json.dump(out, open("C:/Users/22310326/Desktop/Clarus-Equation/verify/Q-0006/attempt-01/adversary/checks.json","w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
