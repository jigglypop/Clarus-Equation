"""Positive mass-frequency quadrature for time-dependent continuum response.

Collapses the prepared Gamma mixture to one density before resolving phases.
Finite numerical integration limits have probability bounds; not physical cutoffs.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaincc, gammaln, log_ndtr

from dimension_boundary_probe import BoundaryProbe
from dimension_homogeneous_background import HomogeneousBackground
from dimension_bao_comparison import distance_shape, load_blocks, score
from dimension_growth_bridge import cassini_band


class FrequencyProbe(BoundaryProbe):
    def __init__(self, *, cells=100, x_max=80., **kwargs):
        super().__init__(**kwargs)
        if not isinstance(cells, int) or cells < 2 or not math.isfinite(x_max) or x_max <= 0:
            raise ValueError("integer cells>=2 and positive finite x_max required")
        self.cells, self.x_max = cells, x_max

    def mixture(self, n_u):
        if not isinstance(n_u, int) or n_u < 2:
            raise ValueError("integer mixture order>=2 required")
        q = self.kernel(self.beta)["continuum_weight"]
        shape, weight = [2.], [1-q]
        tail, upper = 0., 0.
        if q:
            center = -math.log(self.beta)/(2*self.b)
            sd = 1/math.sqrt(2*self.b)
            upper = max(center, 0.)+10*sd
            log_mass = float(log_ndtr(center/sd))
            tail = q*math.exp(float(log_ndtr((center-upper)/sd))-log_mass)
            nodes, weights = np.polynomial.legendre.leggauss(n_u)
            u = (nodes+1)*upper/2
            density = np.exp(-.5*((u-center)/sd)**2-math.log(sd*math.sqrt(2*math.pi))-log_mass)
            shape.extend(2+u)
            weight.extend(q*weights*upper/2*density)
        return np.array(shape), np.array(weight), tail, 2+upper

    def positive_quadrature(self, n_u=32, n_x=8):
        """n_x is Gauss order per frequency cell, not Gamma Laguerre order.

        First-cell squared coordinate improves the fractional-power endpoint.
        No weights are renormalized. Phase convergence must be checked separately.
        """
        if not isinstance(n_x, int) or n_x < 2:
            raise ValueError("integer cell order>=2 required")
        shape, mixture_weights, u_tail, maximum_shape = self.mixture(n_u)
        lo, hi = self.m, math.sqrt(self.m**2+self.a*self.x_max)
        edges = np.linspace(lo, hi, self.cells+1)
        frequencies, frequency_weights = [], []
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            order = max(32, 2*n_x) if index == 0 else n_x
            nodes, weights = np.polynomial.legendre.leggauss(order)
            t, wt = (nodes+1)/2, weights/2
            if index == 0:
                frequencies.extend(left+(right-left)*t*t)
                frequency_weights.extend(wt*2*(right-left)*t)
            else:
                frequencies.extend(left+(right-left)*t)
                frequency_weights.extend(wt*(right-left))
        frequency = np.array(frequencies)
        x = (frequency**2-self.m**2)/self.a
        if not np.isfinite(x).all() or (x <= 0).any():
            raise ValueError("frequency transformation exceeds numerical resolution")
        log_density = ((shape[:, None]-1)*np.log(x)[None, :]-self.beta*x[None, :]
                       +shape[:, None]*math.log(self.beta)-gammaln(shape)[:, None])
        density = mixture_weights@np.exp(log_density)
        weights = np.array(frequency_weights)*2*frequency/self.a*density
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("nonfinite or negative integration weights")
        # For each retained shape k<=k_max, Gamma(k,beta) is stochastically
        # bounded by Gamma(k_max,beta). Add omitted u mass separately.
        x_tail_bound = float(gammaincc(maximum_shape, self.beta*self.x_max))
        return x, weights, {"method": "composite mass-frequency Gauss with squared first cell",
                             "weight_sum": float(weights.sum()), "nodes": len(x),
                             "n_u": n_u, "cell_order": n_x, "cells": self.cells,
                             "x_max_numerical": self.x_max,
                             "u_tail_probability_bound": u_tail,
                             "x_tail_probability_upper": x_tail_bound,
                             "omitted_probability_upper": min(1., u_tail+x_tail_bound),
                             "maximum_frequency_cell_width": (hi-lo)/self.cells}


def run_branch(eta, *, cells, cell_order=8, n_u=32, x_max=80., rtol=3e-10, atol=3e-11):
    s = cassini_band()["maximum_s"]
    probe = FrequencyProbe(eta=eta, coupling=0., cells=cells, x_max=x_max)
    background = HomogeneousBackground(s=s, mass_over_h0=100., probe=probe,
                                       n_u=n_u, n_x=cell_order, rtol=rtol, atol=atol).calibrate()
    z, kinds = [.38, .698, 1.48, 1.48], ["DM_over_rs", "DH_over_rs", "DM_over_rs", "DH_over_rs"]
    values = background.observed_distances(z, kinds)
    leading = distance_shape(z, kinds, .315/(1+s))
    block_scores = []
    for b in load_blocks()[0]:
        physical = background.observed_distances(b["z"], b["kind"])
        reference = distance_shape(b["z"], b["kind"], .315)
        amplitude = 299792.458/(67.4*147.09)
        baseline = score(b["y"], b["cov"], reference, amplitude)
        candidate = score(b["y"], b["cov"], physical, amplitude)
        block_scores.append({"sample": b["name"], "baseline": baseline, "candidate": candidate,
                             "delta_chi2": candidate["chi2"]-baseline["chi2"]})
    history = []
    for n in np.linspace(background.n_initial, 0., 161):
        state = background.solution.sol(n)
        rho, kinetic, potential, h, conformal, log_a_dot = background.quantities(
            n, state, background.rho_bar, background.cosmological_constant)
        history.append([math.log(conformal), log_a_dot, (.5*kinetic+potential)/rho])
    # Exact expanding positive-energy Raychaudhuri evolution makes h>=h_today.
    # This upper-bounds the elapsed time and hence the mass-frequency phase.
    h_today = background.quantities(0., background.solution.y[:,-1],
                                     background.rho_bar, background.cosmological_constant)[3]
    phase_bound = 100.*(-background.n_initial)/h_today
    return {"eta": eta, "mass_over_Href": 100., "s": s, "diagnostics": background.diagnostics(),
            "collective_history_columns": ["ln_A", "d_ln_A_d_tau", "scalar_energy_over_dust"],
            "collective_history": history, "phase_coefficient_upper_from_raychaudhuri": phase_bound,
            "redshifts": z, "observables": kinds, "distances": values.tolist(),
            "relative_distance_change_from_leading": (values/leading-1).tolist(),
            "BAO_blocks": block_scores,
            "block_diagonal_BAO_delta_chi2": sum(b["delta_chi2"] for b in block_scores)}


def report():
    runs = [run_branch(.1, cells=n) for n in (100, 200, 400)]
    reference = np.array(runs[-1]["distances"])
    refinements = [(np.array(r["distances"])/reference-1).tolist() for r in runs[:-1]]
    shapes = [run_branch(eta, cells=400) for eta in (0., 1.)]
    tail_check = run_branch(.1, cells=200, x_max=100.)
    ode_check = run_branch(.1, cells=200, rtol=3e-12, atol=3e-13)
    cell_order_check = run_branch(.1, cells=200, cell_order=12)
    mixture_order_check = run_branch(.1, cells=200, n_u=48)
    contrast = max(np.max(np.abs(np.array(r["distances"])/reference-1)) for r in shapes)
    checks = {"tail_limit": tail_check, "ODE_tolerance": ode_check,
              "cell_order": cell_order_check, "mixture_order": mixture_order_check}
    check_differences = {name: (np.array(branch["distances"])/reference-1).tolist()
                         for name, branch in checks.items()}
    error = max(np.max(np.abs(row)) for row in refinements+list(check_differences.values()))
    history_reference = np.array(runs[-1]["collective_history"])
    history_errors = {name: np.max(np.abs(np.array(branch["collective_history"])-history_reference), axis=0).tolist()
                      for name, branch in {"100_cells": runs[0], "200_cells": runs[1], **checks}.items()}
    return {"method": "frequency resolved positive quadrature of the same measure and initial history",
            "runs": runs, "relative_changes_from_finest": refinements, "shape_runs": shapes,
            "additional_refinement_relative_changes": check_differences,
            "collective_history_max_absolute_changes": history_errors,
            "empirical_refinement_max": float(error), "shape_contrast_max": float(contrast),
            "one_percent_shape_resolution_criterion_passed": bool(error < .01*contrast),
            "status": "numerical sensitivity, not an observational prediction or a rigorous continuum bound",
            "all_domain_rmse": None, "scientific_success": False}


if __name__ == "__main__":
    result = report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("relative_changes_from_finest", "empirical_refinement_max",
                                            "shape_contrast_max", "one_percent_shape_resolution_criterion_passed")}, indent=2))
