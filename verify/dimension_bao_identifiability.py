"""Fair nuisance profiling and local spectral-signal identifiability.

Retrospective compressed data, block-diagonal covariance assumption. A profile
fit is not a predictive RMSE gain; no quantum or all-domain fit is computed.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar

from dimension_bao_comparison import load_blocks, distance_shape, score
from dimension_growth_bridge import cassini_band


def combined_data():
    blocks, sources = load_blocks()
    return {"z": np.concatenate([b["z"] for b in blocks]),
            "y": np.concatenate([b["y"] for b in blocks]),
            "kind": sum([b["kind"] for b in blocks],[]),
            "cov": block_diag(*[b["cov"] for b in blocks]),
            "block_names": [b["name"] for b in blocks], "sources": sources}


def profile_leading_model(data, s, *, effective_bounds=(.05,.6)):
    """Profile A and Omega_local with a common domain in Omega_effective.

    Omega_effective=Omega_local/(1+s); no independent density/H0/rd prior.
    Same effective domain prevents boundary restrictions from mimicking evidence.
    """
    if not math.isfinite(s) or s < 0:
        raise ValueError("finite s>=0 required")
    lo,hi = effective_bounds
    if not 0 < lo < hi < 1 or hi*(1+s) >= 1:
        raise ValueError("valid shared effective domain and local density<1 required")
    def objective(omega_local):
        v = distance_shape(data["z"],data["kind"],omega_local/(1+s))
        return score(data["y"],data["cov"],v)["chi2"]
    optimum = minimize_scalar(objective,bounds=(lo*(1+s),hi*(1+s)),
                              method="bounded",options={"xatol":1e-12})
    if not optimum.success:
        raise RuntimeError(optimum.message)
    effective = optimum.x/(1+s)
    result = score(data["y"],data["cov"],distance_shape(data["z"],data["kind"],effective))
    result.update(s=s,omega_local_fitted=float(optimum.x),omega_effective_fitted=float(effective),
                  fitted_nuisance_count=2,s_identifiable_from_leading_BAO=False)
    return result


def distance_derivative(z,kinds,omega):
    """Analytic derivative w.r.t. effective matter fraction under the integral."""
    def derivative(redshift):
        a = (1+redshift)**3-1
        return -.5*a/(1+omega*a)**1.5
    return np.array([quad(derivative,0,x,epsabs=1e-12,epsrel=1e-12)[0]
                     if kind == "DM_over_rs" else derivative(x)
                     for x,kind in zip(z,kinds,strict=True)])


def nuisance_orthogonal_signal(delta_prediction,covariance,jacobian):
    """Local linear signal after removing specified nuisance tangent directions.

    This is NOT a nonlinear refit or a significance measured from residuals.
    """
    d,c,j = map(lambda x: np.asarray(x,dtype=float),(delta_prediction,covariance,jacobian))
    if d.ndim != 1 or c.shape != (len(d),len(d)) or j.ndim != 2 or j.shape[0] != len(d):
        raise ValueError("matching signal, covariance and Jacobian required")
    if not all(np.isfinite(x).all() for x in (d,c,j)) or not np.allclose(c,c.T):
        raise ValueError("finite inputs and symmetric covariance required")
    l = np.linalg.cholesky(c)
    wd,wj = np.linalg.solve(l,d),np.linalg.solve(l,j)
    shift,_,rank,_ = np.linalg.lstsq(wj,wd,rcond=None)
    remaining = wd-wj@shift
    return {"raw_whitened_norm": float(np.linalg.norm(wd)),
            "nuisance_orthogonal_whitened_norm": float(np.linalg.norm(remaining)),
            "nuisance_parameter_shift_local": shift.tolist(), "nuisance_rank": int(rank),
            "orthogonality_residual": float(np.linalg.norm(wj.T@remaining)),
            "remaining_whitened_vector": remaining.tolist(),
            "role": "local detectability diagnostic, not observed improvement or a nonlinear fit"}


def report():
    data = combined_data()
    s_values = (0.,cassini_band()["maximum_s"],.02)
    fits = [profile_leading_model(data,s) for s in s_values]
    cassini = json.loads(Path(__file__).with_name("cassini2003_signed_summary.json").read_text())
    for fit in fits:
        gamma = -2*fit["s"]/(1+fit["s"])
        marginal = ((cassini["value"]-gamma)/cassini["quoted_uncertainty"])**2
        fit["Cassini_quoted_Gaussian_chi2"] = marginal
        fit["BAO_plus_Cassini_independence_assumed_chi2"] = fit["chi2"]+marginal
        fit["role"] = "retrospective diagnostic; .02 violates the declared long-range Cassini band" if fit["s"] == .02 else "retrospective diagnostic"
    reference = score(data["y"],data["cov"],distance_shape(data["z"],data["kind"],.315),299792.458/(67.4*147.09))
    source = Path(__file__).with_name("dimension_frequency_quadrature.json")
    payload = source.read_bytes()
    dynamic = json.loads(payload)
    if not dynamic["one_percent_shape_resolution_criterion_passed"]:
        raise ValueError("archived dynamic shape comparison is not resolved")
    branches = [dynamic["shape_runs"][0],dynamic["runs"][-1],dynamic["shape_runs"][1]]
    predictions = []
    for eta,branch in zip((0.,.1,1.),branches,strict=True):
        if (branch["eta"] != eta or branch["mass_over_Href"] != 100.
                or not math.isclose(branch["s"],s_values[1],rel_tol=1e-12)
                or not math.isclose(branch["diagnostics"]["omega_local_reconstructed"],.315,abs_tol=1e-10)):
            raise ValueError("archived dynamic parameters differ from declared contrast")
        if [b["sample"] for b in branch["BAO_blocks"]] != data["block_names"]:
            raise ValueError("archived BAO order mismatch")
        predictions.append(np.concatenate([b["candidate"]["prediction"] for b in branch["BAO_blocks"]]))
    omega = .315/(1+s_values[1])
    amplitude = 299792.458/(67.4*147.09)
    jacobian = np.column_stack([distance_shape(data["z"],data["kind"],omega),
                               amplitude*distance_derivative(data["z"],data["kind"],omega)])
    shape_signal = nuisance_orthogonal_signal(predictions[2]-predictions[0],data["cov"],jacobian)
    return {"role": "seen-data identifiability audit, not all-domain optimization",
            "fixed_reference": reference, "equally_profiled_leading_models": fits,
            "leading_family_identity": "f(A,Omega_local,s)=f_GR(A,Omega_local/(1+s))",
            "shape_eta_one_minus_zero_local_diagnostic": shape_signal,
            "dynamic_prediction_source_sha256": hashlib.sha256(payload).hexdigest(),
            "dynamic_inputs": {"mass_over_H0":100., "s":s_values[1], "omega_local":.315,
                               "eta_contrast":[0.,1.],"nuisances":["A","Omega_effective"]},
            "limitations": ["BAO cross-sample covariance approximated as block diagonal",
                            "sound horizon not derived; no external matter-density prior used",
                            "profiled comparison covers leading background only",
                            "dynamic spectral projection is local, not a full nonlinear parameter refit",
                            "Cassini uses the stated unscreened long-range Gaussian-summary approximation",
                            "shared independent quantum/cosmology likelihood remains missing"],
            "actual_quantum_rmse": None, "all_domain_rmse": None, "scientific_success":False}


if __name__ == "__main__":
    result = report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"fixed_RMSE":result["fixed_reference"]["whitened_rmse"],
                      "profiles":[{k:r[k] for k in ("s","omega_effective_fitted","whitened_rmse","BAO_plus_Cassini_independence_assumed_chi2")}
                                  for r in result["equally_profiled_leading_models"]],
                      "shape":result["shape_eta_one_minus_zero_local_diagnostic"]},indent=2))
