"""Extract TDCOSMO likelihood factors from the public JointAnalysis notebook."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from urllib.request import urlretrieve


COMMIT = "6c293af582c398a5c9de60a51cb0c44432a3c598"
NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/TDCOSMO/"
    f"hierarchy_analysis_2020_public/{COMMIT}/JointAnalysis/joint_inference.ipynb"
)
TDCOSMO_SAMPLE_NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/TDCOSMO/"
    f"hierarchy_analysis_2020_public/{COMMIT}/TDCOSMO_sample/tdcosmo_sample.ipynb"
)

CHAIN_TO_NOTEBOOK = {
    "tdcosmo_only_alpha_free_om_covariance.json": {
        "notebook": "tdcosmo_sample",
        "pattern": "tdcosmo_chain_alpha_free_om.h5",
    },
    "tdcosmo_ifu_covariance.json": {
        "notebook": "joint_inference",
        "pattern": "tdcosmo_ifu_chain",
    },
    "tdcosmo_slacs_covariance.json": {
        "notebook": "joint_inference",
        "pattern": "tdcosmo_slacs_chain",
    },
    "tdcosmo_slacs_ifu_covariance.json": {
        "notebook": "joint_inference",
        "pattern": "tdcosmo_slacs_ifu_chain",
    },
}


def notebook_path(name: str) -> Path:
    if name == "joint_inference":
        candidates = [
            Path(__file__).with_name("h0_real_data") / "joint_inference.ipynb",
            Path(
                r"C:/Users/22310326/AppData/Local/Temp/"
                r"tdcosmo_hierarchy_analysis_2020_public/JointAnalysis/joint_inference.ipynb"
            ),
        ]
        url = NOTEBOOK_URL
    elif name == "tdcosmo_sample":
        candidates = [
            Path(__file__).with_name("h0_real_data") / "tdcosmo_sample.ipynb",
            Path(
                r"C:/Users/22310326/AppData/Local/Temp/"
                r"tdcosmo_hierarchy_analysis_2020_public/TDCOSMO_sample/tdcosmo_sample.ipynb"
            ),
        ]
        url = TDCOSMO_SAMPLE_NOTEBOOK_URL
    else:
        raise ValueError(f"unknown notebook: {name}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    target = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target)
    return target


def code_cells(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in data.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def cell_for_chain(cells: list[str], filename_pattern: str) -> str:
    for source in cells:
        if filename_pattern in source:
            return source
    raise ValueError(f"could not find notebook cell containing {filename_pattern}")


def names_in_expr(node: ast.AST) -> set[str]:
    terms: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            name = child.id
            if name == "tdcosmo_posterior_list" or (name.startswith("kwargs_") and name.endswith("_list")):
                terms.add(name)
    return terms


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def extract_sampler_terms_ast(source: str) -> tuple[set[str], bool]:
    """Extract lens-list terms feeding MCMCSampler from one code cell.

    Returns (terms, used_ast). If parsing fails, the caller can fall back to a
    simple text heuristic; public notebooks sometimes contain exploratory code.
    """
    tree = ast.parse(source)
    assignments: dict[str, set[str]] = {}
    sampler_terms: set[str] = set()
    found_sampler = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = names_in_expr(node.value)
        elif isinstance(node, ast.Call) and call_name(node.func) == "MCMCSampler":
            found_sampler = True
            if node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Name) and first_arg.id in assignments:
                    sampler_terms.update(assignments[first_arg.id])
                else:
                    sampler_terms.update(names_in_expr(first_arg))
    return sampler_terms, found_sampler


def extract_sampler_terms_text(source: str) -> set[str]:
    terms: set[str] = set()
    for line in source.splitlines():
        if "lens_list" not in line or "=" not in line:
            continue
        for token in ["tdcosmo_posterior_list", "kwargs_ifu_all_list", "kwargs_ifu_quality_list", "kwargs_sdss_all_list", "kwargs_sdss_quality_list"]:
            if token in line:
                terms.add(token)
    return terms


def extract_sampler_terms(source: str) -> tuple[set[str], str]:
    try:
        terms, found_sampler = extract_sampler_terms_ast(source)
        if found_sampler:
            return terms, "ast"
    except SyntaxError:
        pass
    return extract_sampler_terms_text(source), "text"


def expected_factor_presence(terms: set[str]) -> dict[str, bool]:
    return {
        "time_delay_lens": "tdcosmo_posterior_list" in terms,
        "slacs_population_hierarchy": any("sdss" in term for term in terms),
        "stellar_kinematics": any(
            term == "tdcosmo_posterior_list" or "ifu" in term or "sdss" in term
            for term in terms
        ),
    }


def factor_graph_from_terms(nodes: list[str], terms: set[str]) -> list[dict[str, object]]:
    node_set = set(nodes) - {"h0"}
    lambda_family = sorted(node for node in node_set if node.startswith("lambda_mst"))
    anisotropy = sorted(node for node in node_set if node.startswith("a_ani"))
    factors: list[dict[str, object]] = []
    if "tdcosmo_posterior_list" in terms:
        factors.append(
            {
                "name": "time_delay_lens",
                "closure_scope": "local_endpoint",
                "nodes": ["h0", *lambda_family, *sorted({"alpha_lambda"} & node_set)],
            }
        )
    if any(term == "tdcosmo_posterior_list" or "ifu" in term or "sdss" in term for term in terms):
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
    if any("sdss" in term for term in terms):
        factors.append(
            {
                "name": "slacs_population_hierarchy",
                "closure_scope": "population_global",
                "nodes": lambda_family,
            }
        )
    return factors


def normalized_factors(factors: list[dict[str, object]]) -> list[tuple[str, str, tuple[str, ...]]]:
    return sorted(
        (
            str(factor.get("name")),
            str(factor.get("closure_scope")),
            tuple(sorted(str(node) for node in factor.get("nodes", []))),
        )
        for factor in factors
    )


def payload_factor_presence(payload: dict) -> dict[str, bool]:
    names = {str(factor.get("name")) for factor in payload.get("likelihood_factors", [])}
    return {
        "time_delay_lens": "time_delay_lens" in names,
        "slacs_population_hierarchy": "slacs_population_hierarchy" in names,
        "stellar_kinematics": "stellar_kinematics" in names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("h0_fisher_io_examples")),
        help="directory containing generated TDCOSMO covariance JSON files",
    )
    args = parser.parse_args()
    root = Path(args.path)
    cells_by_notebook: dict[str, list[str]] = {}

    print("# H0 TDCOSMO Notebook Factor Extract Gate")
    print()
    print("| file | notebook | parser | notebook terms | generated factors | status |")
    print("|---|---|---|---|---|---|")

    failed = 0
    for file_name, spec in CHAIN_TO_NOTEBOOK.items():
        notebook_name = spec["notebook"]
        pattern = spec["pattern"]
        if notebook_name not in cells_by_notebook:
            cells_by_notebook[notebook_name] = code_cells(notebook_path(notebook_name))
        cells = cells_by_notebook[notebook_name]
        cell = cell_for_chain(cells, pattern)
        terms, parser_name = extract_sampler_terms(cell)
        payload = json.loads((root / file_name).read_text(encoding="utf-8"))
        generated_factors = factor_graph_from_terms([str(node) for node in payload["nodes"]], terms)
        generated = normalized_factors(generated_factors)
        declared = normalized_factors(payload.get("likelihood_factors", []))
        ok = generated == declared
        if not ok:
            failed += 1
        generated_names = ", ".join(name for name, _, _ in generated)
        print(
            f"| {file_name} | {notebook_name} | {parser_name} | {', '.join(sorted(terms))} | "
            f"{generated_names} | {'PASS' if ok else 'FAIL'} |"
        )

    print()
    if failed:
        raise SystemExit(1)
    print("Verdict: likelihood factor metadata matches the public notebook sampler composition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
