"""Test whether the induced local one-loop action preserves length gluing.

This is a conditional off-glue continuation, not a derived CE microscopic action:
U_prod(x)=sum_cells[beta*S_cell(x_cell,kappa_cell)+Phi_centroid(x_cell)].
The five cells are the actual 1->5 refinement, with 50 independent edge copies.
The local boundary kappas are the previous equal-split convention. Phi follows
from integrating one further barycenter in each cell in the supplied quotient
measure. The kinetic term W is a supplied constant positive diagonal matrix.

For W*xddot=-grad U, preservation of ALL glued positions and velocities requires
C*W^-1*grad U(AQ)=0. A necessary derivative condition is C*W^-1*H*A=0.
The regular initial force satisfies gluing for equal owner masses, but this
derivative condition fails. This does not exclude smaller invariant glued subsets,
special trajectories, a stationary common-metric sector, nonlocal kinetic terms,
or a different microscopic interaction. No Lorentzian Regge dynamics is derived.

An explicit entry uses cells a=(5,1,2,3,4), b=(5,0,2,3,4),
shared internal edge e=(4,5), and f=(0,2), which is absent from a.
The local mixed Hessian in b is
  H_b(e,f)=-2*sqrt(2)*beta-1449*sqrt(10)/722 < 0.
Thus the corresponding entry of C*W^-1*H*A is -H_b(e,f)/w_(b,e),
nonzero for every beta>0 and every finite positive diagonal W.
The one-loop coefficient follows from G=I-J/5, G^-1=I+J,
B=sum_local lengths^2=76/5, and
  H_Phi(e,f)=b_e*b_f*(-4/B^2+tr(G^-1*G_e2*G^-1*G_f2)),
where tr(... )=-5, b_e=2/sqrt(5), b_f=sqrt(2).
G_e2 is the derivative with respect to the squared length.
The disjoint-edge classical entry has zero area second derivative, so its
-2*sqrt(2) value is independent of the local kappa allocation.

Retaining the whole previous flat-branch action does not remove this entry.
For a boundary-only completion B(bar_b), bar_b is the average root-boundary
length. On the flat branch choose B=Phi_root-log|det R_global|. Then the
completed product value equals beta*S_coarse+Phi_depth2. Every such B has
zero derivatives in the witness's INTERNAL edge, so it cannot cancel the defect.
This is a chosen continuation away from the flat branch, not uniqueness of one.

A finite quadratic penalty kappa*||Cx||^2/2 has C^T*C*A=0 and cannot
repair the derivative condition. The same C-coupled bath self-energy also
annihilates A. Arbitrary bath noise can itself drive mismatch; no exact noisy
glued trajectory is claimed. The short-time demonstration uses a frozen
quadratic variational equation, not a full nonlinear or physical time evolution.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

import boundary_one_loop_action as loop


HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=4)
def geometry(shear=0.):
    shear = float(shear)
    if not math.isfinite(shear) or abs(shear) > .3:
        raise ValueError("Finite shear with absolute value at most 0.3 is required")
    data = loop.fixed.gluing.length_gluing(1)
    r = loop.fixed.gluing.reference()
    transform = np.eye(4)
    transform[0, 1] = shear
    points = {v: transform @ p for v, p in data["points"].items()}
    maps = [{tuple(sorted(e)): 10*i+j for j, e in enumerate(combinations(cell, 2))}
            for i, cell in enumerate(data["cells"])]
    global_lengths = np.array([np.linalg.norm(points[u]-points[v]) for u, v in data["global_edges"]])
    a, c = data["assembly"], data["constraint"]
    boundary = [j for j, e in enumerate(data["global_edges"]) if max(e) < 5]
    readout = (a.T/a.sum(axis=0)[:, None])[boundary]
    return {
        **data, "points": points, "maps": maps, "global_lengths": global_lengths,
        "copies": a @ global_lengths, "boundary_readout": readout,
        "kappas": r.equal_split_kappas(data["cells"], tuple(range(5)), np.full(10, math.pi)),
        "boundary_lengths": global_lengths[boundary],
    }


def product_action(copies, data, beta=1.):
    beta = float(beta)
    x = np.asarray(copies, dtype=float)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if x.shape != (50,) or not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("Fifty finite positive copy lengths are required")
    classical, phi = 0., 0.
    gs, gp = np.zeros(50), np.zeros(50)
    hs, hp = np.zeros((50, 50)), np.zeros((50, 50))
    for i, kappa in enumerate(data["kappas"]):
        indices = slice(10*i, 10*i+10)
        value, gradient, hessian, _ = loop.fixed.simplex_derivatives(x[indices], kappa)
        correction = loop.centroid_one_loop(x[indices], beta)
        classical += value
        phi += correction["phi"]
        gs[indices], gp[indices] = gradient, correction["gradient"]
        hs[indices, indices], hp[indices, indices] = hessian, correction["hessian"]
    return {
        "value": beta*classical+phi, "classical_action": classical, "phi": phi,
        "gradient": beta*gs+gp, "hessian": beta*hs+hp,
        "classical_gradient": gs, "loop_gradient": gp,
        "classical_hessian": hs, "loop_hessian": hp,
    }


def witness(data):
    left, right = data["maps"][0][(4, 5)], data["maps"][1][(4, 5)]
    exclusive = data["maps"][1][(0, 2)]
    if (0, 2) in data["maps"][0]:
        raise ArithmeticError("The witness requires an edge absent from the first cell")
    rows = np.flatnonzero((data["constraint"][:, left] == 1)
                         & (data["constraint"][:, right] == -1))
    if len(rows) != 1:
        raise ArithmeticError("The expected oriented shared-face constraint is missing")
    return int(rows[0]), left, right, exclusive, data["global_edges"].index((0, 2))


def flow_case(beta=1., copy_masses=None, penalty=0., shear=0.):
    data = geometry(shear)
    row = product_action(data["copies"], data, beta)
    w = np.ones(50) if copy_masses is None else np.asarray(copy_masses, dtype=float)
    penalty = float(penalty)
    if w.shape != (50,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Fifty finite positive diagonal masses are required")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("The penalty must be finite and nonnegative")
    a, c = data["assembly"], data["constraint"]
    hessian = row["hessian"]+penalty*c.T @ c
    mixed = (c/w) @ hessian @ a
    constraint_row, left, right, exclusive, global_f = witness(data)
    u = a[:, global_f]
    epsilon = 1e-5
    plus = product_action(data["copies"]+epsilon*u, data, beta)["gradient"]
    minus = product_action(data["copies"]-epsilon*u, data, beta)["gradient"]
    measured = ((c/w) @ ((plus-minus)/(2*epsilon)))[constraint_row]
    classical = row["classical_hessian"][right, exclusive]
    correction = row["loop_hessian"][right, exclusive]
    return {
        "beta": beta, "penalty": penalty, "shear": shear,
        "glued_position_residual": float(np.linalg.norm(c @ data["copies"])),
        "glued_force_residual": float(np.linalg.norm((c/w) @ row["gradient"])),
        "linearized_gluing_defect": float(np.linalg.norm(mixed)),
        "witness": {
            "constraint_row": constraint_row, "left_copy": left, "right_copy": right,
            "exclusive_copy": exclusive, "shared_internal_edge": [4, 5], "exclusive_edge": [0, 2],
            "classical_mixed_entry": float(classical), "loop_mixed_entry": float(correction),
            "right_copy_mass": float(w[right]),
            "force_difference_derivative": float(mixed[constraint_row, global_f]),
            "predicted_derivative": float(-(beta*classical+correction)/w[right]),
            "independent_gradient_difference": float(measured),
        },
    }


def projected_control(copies, data, copy_masses, penalty=1., beta=1.):
    """A supplied cross-cell control that preserves the product action on glue.

    P=A*(A^T W A)^-1*A^T W, U_control(x)=U_prod(Px)+penalty*||Cx||^2/2.
    Its gluing preservation is imposed by this projection, not derived dynamically.
    The arbitrary normal penalty demonstrates missing transverse physical input.
    """
    w = np.asarray(copy_masses, dtype=float)
    x = np.asarray(copies, dtype=float)
    penalty = float(penalty)
    if w.shape != (50,) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Fifty finite positive diagonal masses are required")
    if x.shape != (50,) or not np.isfinite(x).all():
        raise ValueError("Fifty finite copy coordinates are required")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("The penalty must be finite and nonnegative")
    a, c = data["assembly"], data["constraint"]
    projector = a @ np.linalg.solve(a.T @ (w[:, None]*a), a.T*w)
    product = product_action(projector @ x, data, beta)
    laplacian = c.T @ c
    return {
        "value": product["value"]+penalty*np.linalg.norm(c @ x)**2/2,
        "gradient": projector.T @ product["gradient"]+penalty*laplacian @ x,
        "hessian": projector.T @ product["hessian"] @ projector+penalty*laplacian,
        "projector": projector,
    }


def full_flat_action_check(shear=0., beta=1.):
    data = geometry(shear)
    product = product_action(data["copies"], data, beta)
    first = loop.evaluate(data["boundary_lengths"], 1, beta)
    second = loop.evaluate(data["boundary_lengths"], 2, beta)
    completion = first["phi"]+second["minus_log_jacobian"]
    return {
        "shear": shear, "beta": beta, "product_value": product["value"],
        "root_and_global_jacobian_completion": completion,
        "completed_value": product["value"]+completion, "previous_full_gamma": second["gamma"],
        "full_action_value_residual": abs(product["value"]+completion-second["gamma"]),
        "completion_depends_only_on_root_boundary": True,
    }


def short_time_variation(times=(.0004, .0002, .0001), penalty=0.):
    """Exact frozen quadratic equation, tested against its initial acceleration."""
    data = geometry()
    row = product_action(data["copies"], data)
    a, c = data["assembly"], data["constraint"]
    penalty = float(penalty)
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("The penalty must be finite and nonnegative")
    h = row["hessian"]+penalty*c.T @ c
    constraint_row, _, _, _, global_f = witness(data)
    initial = a[:, global_f]
    values, vectors = np.linalg.eigh(h)
    coefficients = vectors.T @ initial
    limit = -float((c @ h @ initial)[constraint_row])/2
    out = []
    for time in times:
        time = float(time)
        if not math.isfinite(time) or not 0 < time <= .01:
            raise ValueError("Positive short times at most 0.01 are required")
        # delta(t)-delta(0), using stable cos(z)-1 and cosh(z)-1 formulas.
        argument = np.sqrt(abs(values))*time/2
        difference = np.where(values >= 0, -2*np.sin(argument)**2, 2*np.sinh(argument)**2)
        change = vectors @ (difference*coefficients)
        scaled = float((c @ change)[constraint_row]/time**2)
        out.append({"time": time, "mismatch_over_time_squared": scaled,
                    "half_initial_acceleration": limit, "residual": abs(scaled-limit)})
    return out


def run():
    data = geometry()
    masses = np.linspace(.7, 1.3, 50)
    cases = [flow_case(), flow_case(beta=.1), flow_case(beta=2.),
             flow_case(copy_masses=masses), flow_case(penalty=1000.), flow_case(shear=.2)]
    control = projected_control(data["copies"], data, masses)
    product = product_action(data["copies"], data)
    files = ["one_loop_gluing_flow.py", "boundary_one_loop_action.py", "fixed_boundary_composition.py",
             "length_gluing_bath.py", "local_refinement_bath.py", "continuum_bath.py",
             "F-01/predict_fold_budget.py", "F-01/regge_one_to_five_boundary_hessian.py",
             "F-01/regge_one_to_five_refinement.py"]
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_sha256": {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files},
        "copy_count": len(data["copies"]), "global_length_count": data["assembly"].shape[1],
        "cases": cases, "full_flat_action_checks": [full_flat_action_check(), full_flat_action_check(.2)],
        "short_time_variation": short_time_variation(),
        "short_time_variation_with_penalty": short_time_variation(penalty=1000.),
        "exact_regular_witness_coefficients": [-2*math.sqrt(2), -1449*math.sqrt(10)/722],
        "supplied_cross_cell_control": {
            "same_glued_action_value_residual": abs(control["value"]-product["value"]),
            "force_gluing_residual": float(np.linalg.norm((data["constraint"]/masses) @ control["gradient"])),
            "linearized_gluing_residual": float(np.linalg.norm(
                (data["constraint"]/masses) @ control["hessian"] @ data["assembly"])),
            "cross_cell_projection_and_normal_penalty_are_inputs": True,
        },
        "scope": {
            "local_product_off_glue_continuation_supplied": True,
            "diagonal_kinetic_term_supplied": True, "quotient_measure_supplied": True,
            "complete_prior_flat_action_value_retained": True,
            "all_glued_configurations_preserved_by_tested_flow": False,
            "all_positive_diagonal_masses_repair_regular_witness": False,
            "smaller_common_metric_invariant_sectors_excluded": False,
            "all_possible_cross_cell_interactions_excluded": False,
            "full_nonlinear_trajectory_computed": False, "quantum_noise_removed": False,
            "microscopic_ce_action_derived": False, "continuum_einstein_limit_derived": False,
        },
    }


if __name__ == "__main__":
    result = run()
    (HERE/"one_loop_gluing_flow.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
