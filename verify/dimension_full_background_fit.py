"""Direct nuisance fits of the homogeneous nonlinear scalar background.

Both models fit one common BAO scale and matter density. Spectrum/initial
preparation/s are fixed for each case. This is retrospective, not a holdout.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from dimension_bao_comparison import score
from dimension_bao_identifiability import combined_data, profile_leading_model
from dimension_frequency_quadrature import FrequencyProbe
from dimension_homogeneous_background import HomogeneousBackground
from dimension_growth_bridge import cassini_band


def evaluate_background(data, *, omega_local, s, eta, cells=100):
    probe = FrequencyProbe(eta=eta,coupling=0.,cells=cells)
    background = HomogeneousBackground(s=s,mass_over_h0=100.,probe=probe,
                                       omega_local=omega_local,n_u=32,n_x=8).calibrate()
    shape = background.observed_distances(data["z"],data["kind"])
    result = score(data["y"],data["cov"],shape)
    result.update(omega_local=float(omega_local),eta=eta,s=s,cells=cells,
                  background_diagnostics=background.diagnostics())
    return result


def fit_background(data, *, s, eta, effective_bounds=(.05,.6)):
    cache = {}
    def objective(omega):
        if omega not in cache:
            cache[omega] = evaluate_background(data,omega_local=omega,s=s,eta=eta)
        return cache[omega]["chi2"]
    bounds = tuple(x*(1+s) for x in effective_bounds)
    optimum = minimize_scalar(objective,bounds=bounds,method="bounded",options={"xatol":2e-9})
    if not optimum.success:
        raise RuntimeError(optimum.message)
    objective(optimum.x)
    result = dict(cache[optimum.x])
    fine = evaluate_background(data,omega_local=optimum.x,s=s,eta=eta,cells=200)
    # Local bracketing is a minimum check, not a proof of global convexity.
    offsets = [-1e-4,1e-4]
    neighbor_values = [objective(optimum.x+d) for d in offsets]
    result.update(fitted_nuisance_count=2,optimizer_evaluations=len(cache),
                  evaluation_history=[{"omega_local":float(x),"chi2":r["chi2"]} for x,r in cache.items()],
                  neighbor_density_offsets=offsets,neighbor_chi2=neighbor_values,
                  refinement_delta_chi2=fine["chi2"]-result["chi2"],
                  refinement_max_relative_prediction_change=float(np.max(np.abs(
                      np.array(fine["prediction"])/np.array(result["prediction"])-1))),
                  optimization_status="bounded numerical minimization with local bracketing; no global theorem",
                  scientific_success=False)
    return result


def report():
    data = combined_data()
    baseline = profile_leading_model(data,0.)
    s = cassini_band()["maximum_s"]
    candidates = [fit_background(data,s=s,eta=eta) for eta in (0.,.1,1.)]
    cassini = json.loads(Path(__file__).with_name("cassini2003_signed_summary.json").read_text())
    mean,sigma = cassini["value"],cassini["quoted_uncertainty"]
    penalty = ((mean+2*s/(1+s))/sigma)**2-(mean/sigma)**2
    for r in candidates:
        r["delta_BAO_chi2_to_equally_fitted_GR"] = r["chi2"]-baseline["chi2"]
        r["delta_BAO_plus_Cassini_chi2_independence_assumed"] = r["delta_BAO_chi2_to_equally_fitted_GR"]+penalty
    return {"baseline_GR_same_two_nuisances":baseline,"candidates":candidates,
            "Cassini_marginal_penalty_at_fixed_s":penalty,
            "fixed_inputs":{"mass_over_H0":100.,"s":s,"a_E_initial":.25,
                            "beta":1.,"b":1.,"m":1.,"internal_a":1.},
            "data_sources":data["sources"],
            "limitations":["same BAO block-diagonal covariance approximation and seen data",
                           "A=c/(H0 rd) profiled; sound horizon not predicted",
                           "s and eta not optimized; these are three fixed candidate cases",
                           "initial preparation and potential remain supplied assumptions",
                           "no quantum likelihood, early-universe radiation or full perturbations",
                           "Cassini contribution assumes the same unscreened long-range regime"],
            "actual_quantum_rmse":None,"all_domain_rmse":None,"scientific_success":False}


if __name__ == "__main__":
    result = report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for r in result["candidates"]:
        print(json.dumps({k:r[k] for k in ("eta","omega_local","whitened_rmse",
              "delta_BAO_chi2_to_equally_fitted_GR","delta_BAO_plus_Cassini_chi2_independence_assumed",
              "refinement_delta_chi2","refinement_max_relative_prediction_change")},indent=2))
