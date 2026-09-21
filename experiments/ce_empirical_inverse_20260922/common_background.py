"""One conditional background for the retained DM/DE fractions and CE flow clock.

Reconstruct Z(epsilon)>0 and U(epsilon) so the scalar Euler equation, Friedmann
equation, conservation and a curvature-driven logistic clock all hold together.
This changes the old H(epsilon) readout; it does not silently keep its H0 result.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from inverse_predictions import DELTA, forward, q_of

HERE = Path(__file__).resolve().parent


def main() -> None:
    base = forward(DELTA)
    radiation = 9.2e-5
    ob = (1-radiation)*base["omega_b"]
    odm = (1-radiation)*base["omega_dm"]
    de = (1-radiation)*base["omega_de"]
    om = ob+odm
    w0, wa = base["w0_alternative_DE_branch"], base["wa_alternative_DE_branch"]
    ef = math.exp(-1)
    eps0 = ef-DELTA/math.pi
    xi = math.pi**2/2
    nmin = math.log(1e-10)

    def background(n):
        a = math.exp(n)
        w = w0+wa*(1-a)
        w_n = -wa*a
        density = de*a**(-3*(1+w0+wa))*math.exp(3*wa*(a-1))
        e2 = om*a**-3+radiation*a**-4+density
        hn = -.5*(3*om*a**-3+4*radiation*a**-4+3*(1+w)*density)/e2
        omega = density/e2
        ricci = 3*(om*a**-3+(1-3*w)*density)/e2
        rho = 3*density  # / Mbar^2 H0^2
        return {"a":a,"w":w,"w_N":w_n,"E2":e2,"h_N":hn,"Omega_DE":omega,
                "Ricci_over_H2":ricci,"V":(1-w)*rho/2,
                "V_N":-rho/2*(w_n+3*(1-w*w)),
                "phi_N_squared":3*omega*(1+w)}

    def odds_rhs(n, u):
        return [ef*math.sqrt(2*xi*background(n)["Ricci_over_H2"])]

    u0 = math.log(eps0/(ef-eps0))
    odds = solve_ivp(odds_rhs, (0,nmin), [u0], dense_output=True, rtol=2e-12, atol=2e-13)
    assert odds.success, odds.message

    def field(n):
        bg = background(n)
        u = float(odds.sol(n)[0])
        eps = ef/(1+math.exp(-u))
        eps_n = math.sqrt(2*xi*bg["Ricci_over_H2"])*eps*(ef-eps)
        z = bg["phi_N_squared"]/eps_n**2
        return {**bg,"epsilon":eps,"epsilon_N":eps_n,"Z_over_Mbar2":z}

    rows = []
    for a in [.05,.1,1/3,.5,2/3,1.0]:
        n = math.log(a)
        p = field(n)
        step = 2e-5
        minus, plus = field(n-step), field(n+step)
        eps_nn = (plus["epsilon_N"]-minus["epsilon_N"])/(2*step)
        z_n = (plus["Z_over_Mbar2"]-minus["Z_over_Mbar2"])/(2*step)
        terms = [
            p["Z_over_Mbar2"]*(eps_nn+(3+p["h_N"])*p["epsilon_N"]),
            .5*z_n*p["epsilon_N"],
            p["V_N"]/(p["E2"]*p["epsilon_N"]),
        ]
        relative_residual = abs(sum(terms))/sum(abs(t) for t in terms)
        assert relative_residual < 1e-7, (a,relative_residual)
        assert p["Z_over_Mbar2"] > 0
        rows.append({**p,"scalar_EOM_terms":terms,"scalar_EOM_relative_residual":relative_residual})

    # CMB angular proxy with baryon loading and fixed physical matter/radiation.
    # z_star remains supplied, not solved from a recombination calculation.
    a_star = 1/(1+1089.8)
    photon = radiation/(1+.2271*3.046)
    baryon_loading = 3*ob/(4*photon)

    def theta(e_fn):
        sound = lambda n: math.exp(-n)/e_fn(n)/math.sqrt(3*(1+baryon_loading*math.exp(n)))
        distance = lambda n: math.exp(-n)/e_fn(n)
        rs = quad(sound,nmin,math.log(a_star),epsabs=1e-12,epsrel=1e-10,limit=150)[0]
        dm = quad(distance,math.log(a_star),0,epsabs=1e-12,epsrel=1e-10,limit=150)[0]
        return rs/dm

    target = theta(lambda n:math.sqrt(background(n)["E2"]))

    def theta_standard(k):
        matter, rad = om/k**2,radiation/k**2
        vacuum = 1-matter-rad
        return theta(lambda n:math.sqrt(matter*math.exp(-3*n)+rad*math.exp(-4*n)+vacuum))

    kappa = brentq(lambda k:theta_standard(k)-target,math.sqrt(om+radiation)+1e-5,1.5,xtol=1e-12)
    required_kappa = 67.36/73.04
    required_theta = theta_standard(required_kappa)

    def cpl_angle(candidate_wa):
        def e(n):
            a = math.exp(n)
            candidate_density = de*a**(-3*(1+w0+candidate_wa))*math.exp(3*candidate_wa*(a-1))
            return math.sqrt(om*a**-3+radiation*a**-4+candidate_density)
        return theta(e)

    required_wa = brentq(lambda candidate:cpl_angle(candidate)-required_theta,-8,0,xtol=1e-11)
    inverse_residual = cpl_angle(required_wa)-required_theta
    assert abs(inverse_residual) < 1e-11
    current = rows[-1]
    result = {
        "implementation_sha256":{p:hashlib.sha256((HERE/p).read_bytes()).hexdigest()
                                 for p in ["common_background.py","inverse_predictions.py"]},
        "inputs":{"omega_b":ob,"omega_dm":odm,"omega_de":de,"omega_r":radiation,
                  "w0":w0,"wa":wa,"epsilon_today":eps0,"xi_flow":xi,"z_star":1089.8,"Neff_for_photon_split":3.046},
        "normalization":{"E_today_squared":background(0)["E2"],
                         "nonradiation_fractions_preserved":True,
                         "absolute_fraction_change_from_legacy":{k:-radiation*base[k] for k in ["omega_b","omega_dm","omega_de"]}},
        "old_readout_constraints":{
            "epsilon_required_by_static_split":2*base["omega_de"]-1,
            "epsilon_from_old_offset":eps0,
            "static_split_epsilon_above_logistic_fixed_point":2*base["omega_de"]-1>ef,
            "w0_for_any_smooth_old_H_readout_with_conserved_present_matter":-1,
            "reason":"d/dln(a)[(f(a)-f(1))*(a^-3-1)] vanishes at a=1"},
        "current_reconstructed_action":{"Z_over_Mbar2":current["Z_over_Mbar2"],
            "U_over_Mbar2_H02":current["V"],
            "dU_depsilon_over_Mbar2_H02":current["V_N"]/current["epsilon_N"],
            "epsilon_N":current["epsilon_N"],
            "Ricci_over_H2":current["Ricci_over_H2"]},
        "trajectory":rows,
        "max_relative_scalar_EOM_residual":max(p["scalar_EOM_relative_residual"] for p in rows),
        "epsilon_early":field(nmin)["epsilon"],
        "acoustic_proxy":{"theta":target,"kappa_H_LCDM_inferred_over_H_actual":kappa,
            "H_LCDM_inferred_if_actual_73_04":73.04*kappa,
            "H_LCDM_inferred_if_actual_Hphase":base["H_phase"]*kappa,
            "H_actual_inverse_from_Planck_67_36":67.36/kappa,
            "baryon_loading_coefficient":baryon_loading,
            "scope":"conditional fixed-z_star proxy, not full CMB likelihood; physical densities held fixed"},
        "inverse_common_Hubble_requirements":{
            "input_H_actual":73.04,"input_H_LCDM_inferred":67.36,
            "required_kappa":required_kappa,"required_wa_at_frozen_w0":required_wa,
            "required_CPL_early_w":w0+required_wa,
            "canonical_scalar_nonphantom_condition_satisfied":w0+required_wa>=-1,
            "original_wa":wa,
            "angular_equation_residual":inverse_residual,
            "status":"retrospective inverse requirement; not adopted in reconstructed action"},
        "baryon_Hubble_domain":{
            "q_min_at_delta_one_quarter":q_of(.25),"q_max_limit_at_delta_zero":q_of(0),
            "omega_b_h2_reference":.02242,
            "H_range_with_legacy_omega_b_equals_q":[100*math.sqrt(.02242/q_of(0)),100*math.sqrt(.02242/q_of(.25))],
            "required_q_at_H_73_04":.02242/.7304**2,
            "radiation_normalized_H_range":[100*math.sqrt(.02242/((1-radiation)*q_of(0))),100*math.sqrt(.02242/((1-radiation)*q_of(.25)))],
            "scope":"conditional on the chosen baryon-density reference and q-to-energy readout; not a model-independent bound on H0"},
        "status":"one reconstructed background and clock action exists, but the old Hubble-tension readout/result is not preserved by this candidate",
        "remaining":["independent origin of Z(epsilon), U(epsilon), initial data and the common delta",
                     "recombination and perturbations, not only fixed acoustic epoch",
                     "muon interactions and radiative consistency in the same microscopic theory"],
    }
    (HERE/"common_background_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["old_readout_constraints","current_reconstructed_action","max_relative_scalar_EOM_residual","acoustic_proxy"]},indent=2))


if __name__ == "__main__":
    main()
