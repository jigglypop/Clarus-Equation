"""Convert a public TDCOSMO emcee HDF5 chain into the H0 covariance JSON.

The default target is the public TDCOSMO+SLACS hierarchical chain from
TDCOSMO/hierarchy_analysis_2020_public. The parameter order is inferred from
the public JointAnalysis notebook comments:

    h0, omega_m, lambda_mst, lambda_mst_sigma,
    alpha_lambda, a_ani, a_ani_sigma, sigma_sigmaP

The resulting JSON is intentionally marked experimental because the local vs
global node assignment is our readout-topology hypothesis, not a claim made by
the TDCOSMO collaboration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np

from h0_tdcosmo_notebook_factor_extract_gate import (
    CHAIN_TO_NOTEBOOK,
    cell_for_chain,
    code_cells,
    extract_sampler_terms,
    factor_graph_from_terms,
    notebook_path,
)


COMMIT = "6c293af582c398a5c9de60a51cb0c44432a3c598"
CHAIN_PATH = "JointAnalysis/tdcosmo_slacs_chain_slope_log_scatter.h5"
RAW_URL = (
    "https://raw.githubusercontent.com/TDCOSMO/"
    f"hierarchy_analysis_2020_public/{COMMIT}/{CHAIN_PATH}"
)

PARAMETER_NAMES_BY_DIM = {
    7: [
        "h0",
        "omega_m",
        "lambda_mst",
        "lambda_mst_sigma",
        "alpha_lambda",
        "a_ani",
        "a_ani_sigma",
    ],
    8: [
        "h0",
        "omega_m",
        "lambda_mst",
        "lambda_mst_sigma",
        "alpha_lambda",
        "a_ani",
        "a_ani_sigma",
        "sigma_sigmaP",
    ],
    9: [
        "h0",
        "omega_m",
        "lambda_mst",
        "lambda_mst_sigma",
        "lambda_mst_ifu",
        "lambda_mst_ifu_sigma",
        "alpha_lambda",
        "a_ani",
        "a_ani_sigma",
    ],
    10: [
        "h0",
        "omega_m",
        "lambda_mst",
        "lambda_mst_sigma",
        "lambda_mst_ifu",
        "lambda_mst_ifu_sigma",
        "alpha_lambda",
        "a_ani",
        "a_ani_sigma",
        "sigma_sigmaP",
    ],
}


def load_chain(path: Path, download: bool) -> np.ndarray:
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} does not exist; pass --download to fetch it")
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(RAW_URL, path)

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["mcmc/chain"], dtype=float)


def parameter_names(chain: np.ndarray) -> list[str]:
    try:
        return PARAMETER_NAMES_BY_DIM[chain.shape[2]]
    except KeyError as exc:
        raise ValueError(f"unsupported TDCOSMO chain parameter dimension: {chain.shape[2]}") from exc


def covariance_from_chain(chain: np.ndarray, burnin: int) -> np.ndarray:
    if chain.ndim != 3:
        raise ValueError("expected emcee chain shape (steps, walkers, parameters)")
    parameter_names(chain)
    if burnin < 0 or burnin >= chain.shape[0]:
        raise ValueError("burnin must be non-negative and smaller than the number of steps")
    flat = chain[burnin:].reshape(-1, chain.shape[2])
    return np.cov(flat.T)


def percentile_summary(chain: np.ndarray, burnin: int) -> dict[str, dict[str, float]]:
    flat = chain[burnin:].reshape(-1, chain.shape[2])
    summary = {}
    for index, name in enumerate(parameter_names(chain)):
        p16, p50, p84 = np.percentile(flat[:, index], [16, 50, 84])
        summary[name] = {
            "p16": float(p16),
            "p50": float(p50),
            "p84": float(p84),
            "std": float(np.std(flat[:, index], ddof=1)),
        }
    return summary


def default_local_nodes(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if name.startswith("lambda_mst") or name == "alpha_lambda"
    ]


def default_global_nodes(names: list[str]) -> list[str]:
    return [
        name
        for name in names
        if name not in {"h0"} and name not in set(default_local_nodes(names))
    ]


def infer_likelihood_factors(names: list[str], source_chain_path: str) -> list[dict[str, object]]:
    node_set = set(names) - {"h0"}
    lambda_family = sorted(node for node in node_set if node.startswith("lambda_mst"))
    anisotropy = sorted(node for node in node_set if node.startswith("a_ani"))
    factors: list[dict[str, object]] = [
        {
            "name": "time_delay_lens",
            "closure_scope": "local_endpoint",
            "nodes": ["h0", *lambda_family, *sorted({"alpha_lambda"} & node_set)],
        }
    ]
    if anisotropy:
        factors.append(
            {
                "name": "stellar_kinematics",
                "closure_scope": "global_nuisance",
                "nodes": anisotropy,
            }
        )
    if "omega_m" in node_set:
        factors.append(
            {
                "name": "cosmology_prior",
                "closure_scope": "global_closure",
                "nodes": ["omega_m"],
            }
        )
    if "sigma_sigmaP" in node_set:
        factors.append(
            {
                "name": "velocity_dispersion_systematics",
                "closure_scope": "global_nuisance",
                "nodes": ["sigma_sigmaP"],
            }
        )
    if "slacs" in source_chain_path.lower():
        factors.append(
            {
                "name": "slacs_population_hierarchy",
                "closure_scope": "population_global",
                "nodes": lambda_family,
            }
        )
    return factors


def factor_graph_has_global_mst(factors: list[dict[str, object]]) -> bool:
    for factor in factors:
        if factor.get("closure_scope") != "population_global":
            continue
        nodes = [str(node) for node in factor.get("nodes", [])]
        if any(node.startswith("lambda_mst") for node in nodes):
            return True
    return False


def closure_roles(
    names: list[str],
    source_chain_path: str,
    closure_model: str,
    factors: list[dict[str, object]] | None = None,
) -> tuple[list[str], list[str], str]:
    factors = factors if factors is not None else infer_likelihood_factors(names, source_chain_path)
    model = closure_model
    if model == "auto":
        model = "slacs_global_mst" if factor_graph_has_global_mst(factors) else "lens_local_mst"

    node_set = set(names) - {"h0"}
    if model == "lens_local_mst":
        local = set(default_local_nodes(names))
        mode = "direct"
    elif model == "slacs_global_mst":
        local = {"alpha_lambda"} & node_set
        mode = "direct"
    elif model == "legacy_path":
        local = set(default_local_nodes(names))
        mode = "path"
    else:
        raise ValueError("closure_model must be auto, lens_local_mst, slacs_global_mst, or legacy_path")
    return sorted(local), sorted(node_set - local), mode


def build_payload(
    covariance: np.ndarray,
    chain: np.ndarray,
    burnin: int,
    name: str,
    h0_obs: float | None,
    h0_sigma: float | None,
    chain_path: str,
    closure_model: str,
    factor_source: str,
    output_name: str,
) -> dict:
    names = parameter_names(chain)
    summary = percentile_summary(chain, burnin)
    if factor_source == "ast":
        likelihood_factors = ast_likelihood_factors(names, output_name)
    elif factor_source == "path":
        likelihood_factors = infer_likelihood_factors(names, chain_path)
    else:
        raise ValueError("factor_source must be 'ast' or 'path'")
    local_nodes, global_nodes, conductance_mode = closure_roles(
        names,
        chain_path,
        closure_model,
        likelihood_factors,
    )
    return {
        "name": name,
        "nodes": names,
        "observable": "h0",
        "local_nodes": local_nodes,
        "global_nodes": global_nodes,
        "conductance_mode": conductance_mode,
        "likelihood_factors": likelihood_factors,
        "matrix_type": "covariance",
        "matrix": covariance.tolist(),
        "h0_obs": float(h0_obs) if h0_obs is not None else summary["h0"]["p50"],
        "h0_sigma": float(h0_sigma) if h0_sigma is not None else summary["h0"]["std"],
        "source": {
            "repo": "https://github.com/TDCOSMO/hierarchy_analysis_2020_public",
            "commit": COMMIT,
            "chain_path": chain_path,
            "chain_shape": list(chain.shape),
            "burnin_steps": burnin,
            "parameter_order_basis": "JointAnalysis/joint_inference.ipynb comments and printed table order",
            "mapping_status": "experimental CE readout topology assignment",
            "closure_model": closure_model,
            "factor_source": factor_source,
            "role_basis": "AST-generated likelihood_factors" if factor_source == "ast" else "path-inferred likelihood_factors",
        },
        "posterior_summary": summary,
    }


def ast_likelihood_factors(names: list[str], output_name: str) -> list[dict[str, object]]:
    if output_name not in CHAIN_TO_NOTEBOOK:
        raise ValueError(f"no notebook provenance rule for output file: {output_name}")
    spec = CHAIN_TO_NOTEBOOK[output_name]
    cells = code_cells(notebook_path(spec["notebook"]))
    cell = cell_for_chain(cells, spec["pattern"])
    terms, _ = extract_sampler_terms(cell)
    return factor_graph_from_terms(names, terms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(Path(__file__).with_name("h0_real_data") / "tdcosmo_slacs_chain_slope_log_scatter.h5"),
        help="local HDF5 chain path",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("h0_fisher_io_examples") / "tdcosmo_slacs_covariance.json"),
        help="output covariance JSON path",
    )
    parser.add_argument(
        "--name",
        default="TDCOSMO+SLACS public hierarchical covariance",
        help="channel name stored in JSON",
    )
    parser.add_argument(
        "--source-chain-path",
        default=CHAIN_PATH,
        help="source path stored in JSON metadata",
    )
    parser.add_argument("--h0-obs", type=float, default=None, help="optional H0 observation value")
    parser.add_argument("--h0-sigma", type=float, default=None, help="optional H0 observation sigma")
    parser.add_argument(
        "--closure-model",
        default="auto",
        choices=["auto", "lens_local_mst", "slacs_global_mst", "legacy_path"],
        help="local/global role model for CE readout topology",
    )
    parser.add_argument(
        "--factor-source",
        default="ast",
        choices=["ast", "path"],
        help="build likelihood_factors from notebook AST provenance or path heuristic",
    )
    parser.add_argument("--burnin", type=int, default=200, help="discard this many initial emcee steps")
    parser.add_argument("--download", action="store_true", help="download the public chain if missing")
    args = parser.parse_args()

    chain_path = Path(args.input)
    output_path = Path(args.output)
    chain = load_chain(chain_path, args.download)
    covariance = covariance_from_chain(chain, args.burnin)
    payload = build_payload(
        covariance,
        chain,
        args.burnin,
        args.name,
        args.h0_obs,
        args.h0_sigma,
        args.source_chain_path,
        args.closure_model,
        args.factor_source,
        output_path.name,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("# TDCOSMO HDF5 to H0 Covariance JSON")
    print()
    print(f"input = {chain_path}")
    print(f"shape = {tuple(chain.shape)}")
    print(f"burnin = {args.burnin}")
    print(f"output = {output_path}")
    print(f"h0 median = {payload['posterior_summary']['h0']['p50']:.6f}")
    print("Verdict: covariance JSON written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
