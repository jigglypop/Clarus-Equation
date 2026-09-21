"""Preserve the historical Hubble-flow result and invert its dimensionless map.

The old finite-grid result and a converged evaluation of the same toy equations
are reported separately. This is not a recombination/Boltzmann calculation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent


def historical_namespace() -> dict:
    """Evaluate only the archived numeric definitions, never their CLI or imports."""
    scope = {"math": math}
    for name, select in [
        ("hubble_numerics", lambda n: isinstance(n, ast.FunctionDef) and n.name in {"linspace", "simpson"}),
        ("hubble_flow", lambda n: isinstance(n, ast.FunctionDef) or
         (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets))),
    ]:
        path = HERE/"sources"/(name+".txt")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = ast.Module(body=[n for n in tree.body if select(n)], type_ignores=[])
        exec(compile(selected, str(path), "exec"), scope)
    return scope


def refined(delta: float, xi: float, a_min: float = 1e-6, rtol: float = 1e-10) -> dict:
    ef = math.exp(-1)
    e0 = ef-delta/math.pi
    om0, ol0 = (1-e0)/2, (1+e0)/2
    radiation = 9.2e-5
    astar = 1/(1+1089.8)

    # Exact log-odds transformation of epsilon'=r epsilon(ef-epsilon).
    u0 = math.log(e0/(ef-e0))

    def rhs(t, u):
        a = math.exp(t)
        matter = om0*a**-3/(om0*a**-3+ol0+radiation*a**-4)
        return [ef*math.sqrt(2*xi*(12-9*matter))]

    solution = solve_ivp(rhs, (0, math.log(a_min)), [u0], rtol=rtol, atol=rtol/100, dense_output=True)
    assert solution.success, solution.message

    def eps_at(a):
        u = float(solution.sol(math.log(a))[0])
        return ef/(1+math.exp(-u))

    def running_e(a):
        eps = eps_at(a)
        return math.sqrt((1-eps)/2*a**-3+(1+eps)/2+radiation*a**-4)

    def angular_scale(e_fn):
        f = lambda t: math.exp(-t)/e_fn(math.exp(t))
        rs, rs_err = quad(f, math.log(a_min), math.log(astar), epsabs=1e-12, epsrel=rtol, limit=200)
        dm, dm_err = quad(f, math.log(astar), 0, epsabs=1e-12, epsrel=rtol, limit=200)
        return rs/math.sqrt(3)/dm

    target = angular_scale(running_e)

    def standard_angle(kappa):
        om = om0/kappa**2
        ol = 1-om
        if not 0 < om < 1:
            raise ValueError("positive flat matter-vacuum background required")
        return angular_scale(lambda a: math.sqrt(om*a**-3+ol+radiation*a**-4))

    kappa = brentq(lambda k: standard_angle(k)-target, math.sqrt(om0)+1e-6, 1.35, xtol=1e-12)
    return {"delta":delta,"epsilon_today":e0,"xi_flow":xi,"a_min":a_min,"rtol":rtol,
            "omega_m_background":om0,"omega_de_background":ol0,
            "theta_toy":target,"epsilon_at_recombination":eps_at(astar),
            "kappa_Hcmb_over_Hlocal":kappa,"Hcmb_if_Hlocal_73_04":kappa*73.04,
            "delta_H_if_Hlocal_73_04":73.04*(1-kappa),
            "Hlocal_from_Planck_base_67_36":67.36/kappa,
            "Hlocal_from_Planck_BAO_67_66":67.66/kappa,
            "Hlocal_from_DESI_CMB_68_17":68.17/kappa}


def main() -> None:
    old = historical_namespace()
    delta = old["DELTA"]
    xi = math.pi**2/2
    e0 = math.exp(-1)-delta/math.pi
    grid, eps = old["integrate_epsilon"](e0, xi, 1.0)
    theta = old["theta_star"](grid, eps, old["Z_STAR"])
    hcmb = old["extract_h0_cmb"](theta, 73.04, e0, old["Z_STAR"])
    # Historical bisection has a finite angular tolerance; compare its scale property.
    hcmb2 = old["extract_h0_cmb"](theta, 70.0, e0, old["Z_STAR"])
    evaluations = [refined(delta,xi,1e-6,1e-9), refined(delta,xi,1e-6,1e-12),
                   refined(delta,xi,1e-9,1e-11), refined(delta,xi,1e-10,1e-11)]
    assert abs(evaluations[0]["Hcmb_if_Hlocal_73_04"]-evaluations[1]["Hcmb_if_Hlocal_73_04"]) < 1e-5
    assert abs(evaluations[2]["Hcmb_if_Hlocal_73_04"]-evaluations[3]["Hcmb_if_Hlocal_73_04"]) < .001
    result = {
        "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "historical": {"theta_toy":theta,"Hlocal_input":73.04,"Hcmb":hcmb,"delta_H":73.04-hcmb,
            "kappa":hcmb/73.04,"Hlocal_from_67_36":67.36/(hcmb/73.04),
            "second_Hlocal_input":70,"second_Hcmb":hcmb2,"scale_ratio_difference":hcmb/73.04-hcmb2/70},
        "refined_same_equations":evaluations,
        "conditions": ["c_s=c/sqrt(3)","fixed dimensionless radiation fraction",
            "fixed transition parameters","same toy extraction map","no baryon loading or recombination model",
            "historical normalization E(1)^2=1+Omega_r retained; the common factor cancels from the H ratio"],
        "status":"historical branch preserved; conditional inverse map verified; no Hubble-tension solution claim",
    }
    (HERE/"hubble_flow_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
