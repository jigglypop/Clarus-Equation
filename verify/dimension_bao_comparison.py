"""Retrospective BAO comparison of the declared late-time background.

Uses published Gaussian BAO blocks, not growth/RSD measurements. The sound
horizon is supplied or profiled, never claimed as derived by this model.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag

from dimension_growth_bridge import cassini_band

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"benchmarks/cosmology/snkc_quench_bg_holdout_v1"
ASSETS = [
    ("DR12_LRG", "sdss_DR12_LRG_BAO_DMDH.dat", "sdss_DR12_LRG_BAO_DMDH_covtot.txt",
     "ccdbe5ad44016ea09e10e30f0178eb75b756417730b7b53c611ac896623c5f81",
     "fd2a67856f0ffa7267cff5245b579dcdb4b5cd461849377a2f8e9582a7679544"),
    ("DR16_LRG", "sdss_DR16_LRG_BAO_DMDH.dat", "sdss_DR16_LRG_BAO_DMDH_covtot.txt",
     "b3317e7590799fad71a9a707023d0743c14d87399d6bb4129965d6a5732d91be",
     "1a45e106f8e2bbf8742a6c3d4a9c11bdc288801fc6824e0db8cfbab4290f6160"),
    ("DR16_QSO", "sdss_DR16_QSO_BAO_DMDH.txt", "sdss_DR16_QSO_BAO_DMDH_covtot.txt",
     "9d3a43515d009d5c836728d4af1f1d02887fcdd874aba098c597f1f47693bbe6",
     "c0d8bab47132045139c5bbd0ebfd8464434e1354371ceeeca70bb90ecbcee383")]


def load_blocks(directory=DATA):
    blocks, provenance = [], []
    for name, filename, covfile, filehash, covhash in ASSETS:
        for file, expected in ((filename, filehash), (covfile, covhash)):
            actual = hashlib.sha256((directory/file).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"source hash mismatch: {file}")
            provenance.append({"file": file, "sha256": actual,
                               "upstream": f"https://raw.githubusercontent.com/CobayaSampler/bao_data/bb0c1c9/{file}"})
        rows = [line.split() for line in (directory/filename).read_text().splitlines() if line.strip()]
        z, y = np.array([[float(row[0]), float(row[1])] for row in rows]).T
        kind = [row[2] for row in rows]
        if not all(k in ("DM_over_rs", "DH_over_rs") for k in kind):
            raise ValueError("unknown distance observable")
        if len(set(zip(z, kind))) != len(z) or not np.isfinite([z, y]).all() or (z <= 0).any():
            raise ValueError("invalid or duplicate data rows")
        cov = np.loadtxt(directory/covfile)
        if cov.shape != (len(z), len(z)) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T):
            raise ValueError("invalid covariance")
        np.linalg.cholesky(cov)
        blocks.append({"name": name, "z": z, "y": y, "kind": kind, "cov": cov})
    return blocks, provenance


def distance_shape(z, kind, omega):
    if not math.isfinite(omega) or not 0 < omega <= 1:
        raise ValueError("0<omega<=1 required")
    def inverse_e(x):
        return 1/math.sqrt(omega*(1+x)**3+1-omega)
    output = []
    for redshift, observable in zip(z, kind, strict=True):
        if not math.isfinite(redshift) or redshift < 0:
            raise ValueError("finite nonnegative redshift required")
        if observable == "DM_over_rs":
            output.append(quad(inverse_e, 0, redshift, epsabs=1e-12, epsrel=1e-12)[0])
        elif observable == "DH_over_rs":
            output.append(inverse_e(redshift))
        else:
            raise ValueError("unknown observable")
    return np.array(output)


def score(y, cov, shape, amplitude=None):
    """GLS for y=A*shape, with one shared A=c/(H0*rd) if profiled."""
    chol = np.linalg.cholesky(cov)
    wy, wv = np.linalg.solve(chol, y), np.linalg.solve(chol, shape)
    fitted = amplitude is None
    if fitted:
        amplitude = float(wv@wy/(wv@wv))
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("positive finite BAO amplitude required")
    residual = wy-amplitude*wv
    chi2 = float(residual@residual)
    return {"amplitude_c_over_H0_rd": amplitude, "chi2": chi2,
            "whitened_rmse": math.sqrt(chi2/len(y)), "n": len(y),
            "fitted_nuisance_count": int(fitted), "prediction": (amplitude*shape).tolist()}


def report():
    blocks, sources = load_blocks()
    # Cross-sample blocks are set to zero as an explicit approximation, not
    # inferred from the absence of a cross-covariance file.
    combined = {"name": "eight_points_block_diagonal_approximation",
                "z": np.concatenate([b["z"] for b in blocks]),
                "y": np.concatenate([b["y"] for b in blocks]),
                "kind": sum([b["kind"] for b in blocks], []),
                "cov": block_diag(*[b["cov"] for b in blocks])}
    band_s = cassini_band()["maximum_s"]
    fixed_amplitude = 299792.458/(67.4*147.09)
    results = []
    for block in blocks+[combined]:
        for mode in ("fixed_H0_rd", "profile_one_H0_rd"):
            entries = []
            for s in (0., band_s, .02):
                omega = .315/(1+s)  # same local G, long-range calibration limit
                shape = distance_shape(block["z"], block["kind"], omega)
                item = score(block["y"], block["cov"], shape,
                             fixed_amplitude if mode == "fixed_H0_rd" else None)
                item.update(s=s, omega_E0=omega,
                            cassini_nominal_band_compatible=s <= band_s)
                item["delta_chi2_to_GR"] = item["chi2"]-entries[0]["chi2"] if entries else 0.
                entries.append(item)
            results.append({"sample": block["name"], "mode": mode, "comparisons": entries})
    return {"role": "retrospective conditional late-time BAO check, not independent holdout",
            "sources": sources, "results": results,
            "quality": {"rows": 8, "unique_sample_z_observable_keys": 8, "missing_values": 0,
                        "verified_file_hashes": 6, "positive_definite_covariance_blocks": 3,
                        "redshift_range": [.38, 1.48]},
            "supplied_inputs": {"omega_local": .315, "H0": 67.4, "rd_mpc_fixed_mode": 147.09,
                                "f_cal": 1., "mass_over_H0": 1000.},
            "limitations": ["rd is not predicted; radiation and early-time scalar background are absent",
                            "profiled A is one nuisance per fit; per-block fits are diagnostic only",
                            "joint eight-point covariance assumes zero cross-sample blocks",
                            "compressed BAO template validity under this modified model remains unchecked",
                            "Lambda boundary condition varies with s to retain flat H0",
                            ".02 is an incompatible control; no s is optimized using these observations"],
            "quantum_rmse": None, "all_domain_rmse": None, "scientific_success": False}


if __name__ == "__main__":
    result = report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for block in result["results"]:
        print(block["sample"], block["mode"],
              [(r["s"], r["whitened_rmse"], r["delta_chi2_to_GR"]) for r in block["comparisons"]])
