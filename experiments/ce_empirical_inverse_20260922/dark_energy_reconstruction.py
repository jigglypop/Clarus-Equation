"""Construct a canonical scalar potential from the retained CE DE readout.

The supplied present fractions, H0 and CPL history determine a parametric
potential. The proof is conservation + Friedmann + monotonic field inversion;
finite-difference KG checks are an independent numerical check, not the proof.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

from scipy.integrate import quad

from inverse_predictions import DELTA, forward

HERE = Path(__file__).resolve().parent


def main() -> None:
    base = forward(DELTA)
    om, de = base["omega_m"], base["omega_de"]
    w0, wa = base["w0_alternative_DE_branch"], base["wa_alternative_DE_branch"]

    def state(n):
        a = math.exp(n)
        w = w0+wa*(1-a)
        rho = 3*de*a**(-3*(1+w0+wa))*math.exp(3*wa*(a-1))
        e2 = om*a**-3+rho/3
        omega = rho/(3*e2)
        u = math.sqrt(3*omega*(1+w))  # d(phi/Mbar)/d ln a
        v = (1-w)*rho/2             # V/(Mbar^2 H0^2)
        en = -.5*(3*om*a**-3+(1+w)*rho)/e2  # d ln H/d ln a
        return {"a":a,"w":w,"rho":rho,"E2":e2,"Omega_DE":omega,"u":u,"V":v,"E_log_derivative":en}

    rows = []
    for a in [.05,.1,.2,1/3,.4,.5,.6,2/3,.8,1.0]:
        n = math.log(a)
        s = state(n)
        phi, err = quad(lambda t: state(t)["u"], 0, n, epsabs=1e-12, epsrel=1e-12)
        h = 2e-5
        lp, lm = state(n+h), state(n-h)
        un = (lp["u"]-lm["u"])/(2*h)
        vn = (lp["V"]-lm["V"])/(2*h)
        kg = un+(3+s["E_log_derivative"])*s["u"]+vn/(s["u"]*s["E2"])
        rows.append({**s,"phi_over_reduced_M_Pl":phi,"KG_dimensionless_residual":kg,
                     "redshift":1/a-1,"H_km_s_Mpc":base["H_phase"]*math.sqrt(s["E2"]),
                     "H_static_Lambda_comparison":base["H_phase"]*math.sqrt(om*a**-3+de),
                     "Omega_DM":base["omega_dm"]*a**-3/s["E2"],
                     "DM_over_DE":3*base["omega_dm"]*a**-3/s["rho"]})
    maximum = max(abs(r["KG_dimensionless_residual"]) for r in rows)
    assert maximum < 1e-7, maximum
    xi = base["alpha_s"]**(1/3)
    current = state(0)
    assert abs(.5*current["u"]**2-xi**2) < 1e-14
    slope = (-3*de/2*(-wa+3*(1-w0*w0)))/current["u"]
    result = {
        "implementation_sha256":{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                  for name in ["dark_energy_reconstruction.py","inverse_predictions.py"]},
        "inputs":{"omega_m":om,"omega_de":de,"H0":base["H_phase"],"w0":w0,"wa":wa,"xi":xi},
        "domain":{"a_min":.05,"a_max":1.0,"w_early_limit":w0+wa,"nonphantom_all_0_a_1":w0+wa>-1 and w0>-1},
        "current":{"phi_dot_over_Mbar_H0":current["u"],"kinetic_over_Mbar2_H02":xi**2,
                    "potential_over_Mbar2_H02":current["V"],"potential_slope_over_Mbar_H02":slope},
        "reconstruction":rows,"max_KG_numerical_residual":maximum,
        "CPL_pair_inverse":{"equation":"Omega_DE=1+wa/(3*(1+w0))",
            "historical_pair_recovers_DE":1+wa/(3*(1+w0)),
            "DESI_DR2_CMB_DESY5_central_implied_DE":1-.86/(3*(1-.752)),
            "status":"central-value compatibility constraint, not a confidence exclusion"},
        "DM_inverse_dynamics":{
            "preserved_present_DM_over_DE":base["R_3layer"],
            "conserved_dust_ratio_history":"R(a)=R0*a^(3*(w0+wa))*exp(-3*wa*(a-1))",
            "if_constant_ratio_is_required_Q_over_H_rho_DM_today":-3*w0/(1+base["R_3layer"]),
            "constant_ratio_transfer_law":"Q=-3*H*w*rho_DM/(1+R); positive Q flows DE to DM",
            "status":"constant-ratio alternative is an inverse requirement, not used in reconstructed conserved-dust evolution"},
        "status":"new conditional scalar-potential construction; CPL history and normalization supplied, no CE microscopic or unique-potential claim",
        "limitations":["dust + scalar late-time background; no radiation/neutrino or recombination completion",
            "no independent derivation of the prescribed CPL history or present scalar initial data",
            "not the 29.65 MeV muon mediator; two fields are not a unified action derivation",
            "flat-LCDM compressed posterior scores do not validate this evolving-DE background"],
    }
    (HERE/"dark_energy_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["inputs","domain","current","max_KG_numerical_residual","CPL_pair_inverse"]},indent=2))


if __name__ == "__main__":
    main()
