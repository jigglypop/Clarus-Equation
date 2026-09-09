"""Classical homogeneous continuum-scalar/dust background via positive quadrature.

Retains matter exchange, scalar stress and Jordan-frame observables. Initial
preparation, quadratic potential, Lambda, GR and universal coupling are inputs.
No radiation, quantum stress, early sound horizon or independent prediction.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import lambertw

from dimension_boundary_probe import BoundaryProbe
from dimension_bao_comparison import load_blocks, distance_shape, score
from dimension_growth_bridge import cassini_band


class HomogeneousBackground:
    def __init__(self, *, s, mass_over_h0=1000., probe=None, omega_local=.315,
                 a_initial=.25, n_u=16, n_x=24, rtol=3e-10, atol=3e-11):
        if not (math.isfinite(s) and s >= 0 and math.isfinite(mass_over_h0) and mass_over_h0 > 0
                and 0 < omega_local < 1 and 0 < a_initial < 1):
            raise ValueError("finite nonnegative coupling, positive mass and valid density/initial scale required")
        self.probe = probe if probe is not None else BoundaryProbe(coupling=0.)
        if self.probe.coupling != 0:
            raise ValueError("background rank-one potential coupling must be zero")
        x, weights, self.quadrature = self.probe.positive_quadrature(n_u, n_x)
        self.v = np.sqrt(weights)
        self.d = self.probe.m**2+self.probe.a*x
        self.n = len(x)
        self.g0 = float(np.sum(weights/self.d))
        self.s, self.mass2 = s, mass_over_h0**2
        self.epsilon = s/(2*self.mass2)
        self.kinetic_factor = s/(2*self.mass2**2)
        self.omega, self.n_initial = omega_local, math.log(a_initial)
        self.rtol, self.atol = rtol, atol
        # Finite quadrature has a measured norm, not silently normalized to one.
        self.local_factor = 1+s*float(np.sum(weights))

    def ambient_resolvent(self, z, rho_e):
        """Static fluctuation Hessian at fixed conserved dust density.

        In units M_*^2: D_eff=D0+(s*rho_e/(2*Mratio^2))*|v><v|.
        rho_e is in Mpl^2 Href^2 units, not kg/m^3. This is the exact
        rank-one resolvent of the stated local Hessian, not a device solution.
        """
        if not math.isfinite(rho_e) or rho_e < 0:
            raise ValueError("finite nonnegative Einstein-frame density required")
        g = self.probe.bare_resolvent(z)[0]
        return g/(1+self.epsilon*rho_e*g)

    def quantities(self, n, state, rho_bar, cosmological_constant):
        u, w = state[:self.n], state[self.n:2*self.n]
        log_a = self.epsilon*float(self.v@u)
        a_conformal = math.exp(log_a)
        rho = rho_bar*math.exp(-3*n)*a_conformal
        kinetic_twice = self.kinetic_factor*float(w@w)
        potential = .5*self.epsilon*float((self.d*u)@u)
        h2 = (rho+.5*kinetic_twice+potential+cosmological_constant)/3
        if h2 <= 0:
            raise ValueError("non-expanding branch")
        h = math.sqrt(h2)
        log_a_dot = self.epsilon*float(self.v@w)
        return rho, kinetic_twice, potential, h, a_conformal, log_a_dot

    def initial(self, rho_bar, cosmological_constant):
        unperturbed_rho = rho_bar*math.exp(-3*self.n_initial)
        coefficient = self.epsilon*self.g0
        rho = (float(lambertw(coefficient*unperturbed_rho).real)/coefficient
               if coefficient else unperturbed_rho)
        u = -self.v*rho/self.d
        w_over_h = 3*self.v*rho/(self.d*(1+coefficient*rho))
        potential = .5*self.epsilon*float((self.d*u)@u)
        denominator = 3-.5*self.kinetic_factor*float(w_over_h@w_over_h)
        if denominator <= 0:
            raise ValueError("supplied tracking initial velocity is outside expanding branch")
        h = math.sqrt((rho+potential+cosmological_constant)/denominator)
        # The last two components integrate conformal distance and an independent
        # Raychaudhuri H for the Friedmann constraint check.
        return np.r_[u, h*w_over_h, 0., h]

    def integrate(self, rho_bar, cosmological_constant, *, dense=False):
        def rhs(n, state):
            rho, kinetic, _, h, _, _ = self.quantities(n, state, rho_bar, cosmological_constant)
            u, w = state[:self.n], state[self.n:2*self.n]
            return np.r_[w/h, -3*w-self.mass2*(self.d*u+self.v*rho)/h,
                         math.exp(-n)/h, -.5*(rho+kinetic)/h]
        solution = solve_ivp(rhs, (self.n_initial, 0.), self.initial(rho_bar, cosmological_constant),
                             method="DOP853", rtol=self.rtol, atol=self.atol, dense_output=dense,
                             t_eval=None if dense else [0.])
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution

    def calibrate(self):
        """Match measured H_J0/H_ref=1, Omega_local; Lambda is a fitted boundary input."""
        rho_bar = 3*self.omega/self.local_factor
        cosmological_constant = 3-rho_bar
        for iteration in range(8):
            sol = self.integrate(rho_bar, cosmological_constant)
            final = sol.y[:, -1]
            rho, kinetic, potential, h, a0, log_a_dot = self.quantities(0., final, rho_bar, cosmological_constant)
            hj = (h+log_a_dot)/a0
            omega = self.local_factor*rho/(3*a0**2*hj**2)
            if max(abs(hj-1), abs(omega-self.omega)) < 2e-12:
                break
            rho_bar = 3*self.omega*a0/self.local_factor
            cosmological_constant = 3*(a0-log_a_dot)**2-rho_bar*a0-.5*kinetic-potential
        else:
            raise RuntimeError("local calibration did not converge")
        self.rho_bar, self.cosmological_constant = rho_bar, cosmological_constant
        self.solution = self.integrate(rho_bar, cosmological_constant, dense=True)
        self.a0 = self.quantities(0., self.solution.y[:, -1], rho_bar, cosmological_constant)[4]
        self.calibration_iterations = iteration+1
        return self

    def at(self, n):
        state = self.solution.sol(n)
        q = self.quantities(n, state, self.rho_bar, self.cosmological_constant)
        rho, kinetic, potential, h, conformal, log_a_dot = q
        return {"z_J": self.a0*math.exp(-n)/conformal-1,
                "H_J_over_Href": (h+log_a_dot)/conformal,
                "DM_Href_over_c": self.a0*(self.solution.y[-2,-1]-state[-2]),
                "rho_E": rho, "A": conformal, "scalar_energy": .5*kinetic+potential,
                "relative_raychaudhuri_constraint_error": float((state[-1]-h)/h)}

    def observed_distances(self, redshifts, kinds):
        result = []
        zmax = self.at(self.n_initial)["z_J"]
        for z, kind in zip(redshifts, kinds, strict=True):
            if not 0 <= z <= zmax:
                raise ValueError("observed redshift outside supplied initial history")
            n = 0. if z == 0 else brentq(lambda n: self.at(n)["z_J"]-z,
                                        self.n_initial, 0., xtol=2e-14)
            point = self.at(n)
            if point["H_J_over_Href"] <= 0:
                raise ValueError("Jordan expansion branch required")
            if kind == "DM_over_rs":
                result.append(point["DM_Href_over_c"])
            elif kind == "DH_over_rs":
                result.append(1/point["H_J_over_Href"])
            else:
                raise ValueError("unknown observable")
        return np.asarray(result)

    def diagnostics(self):
        points = [self.at(n) for n in np.linspace(self.n_initial, 0, 81)]
        today = points[-1]
        return {"quadrature": self.quadrature, "g0_quadrature": self.g0,
                "g0_exact_heat_integral": self.probe.bare_resolvent(0.)[0],
                "calibration_iterations": self.calibration_iterations,
                "H_J0_over_Href": today["H_J_over_Href"],
                "omega_local_reconstructed": self.local_factor*today["rho_E"]/(3*today["A"]**2*today["H_J_over_Href"]**2),
                "rho_bar_boundary_input": self.rho_bar, "Lambda_boundary_input": self.cosmological_constant,
                "maximum_constraint_error": max(abs(p["relative_raychaudhuri_constraint_error"]) for p in points),
                "maximum_abs_ln_A": max(abs(math.log(p["A"])) for p in points),
                "maximum_scalar_energy_to_dust": max(p["scalar_energy"]/p["rho_E"] for p in points),
                "a_E_initial_input": math.exp(self.n_initial),
                "initial_state": "instantaneous matter-dependent minimum and its tracking velocity",
                "local_G_condition": "unscreened effectively massless laboratory response; ambient-density check is separate",
                "physical_status": "classical homogeneous truncation, finite quadrature approximation",
                "scientific_success": False}


def report():
    s = cassini_band()["maximum_s"]
    model = HomogeneousBackground(s=s).calibrate()
    comparisons = []
    amplitude = 299792.458/(67.4*147.09)
    for block in load_blocks()[0]:
        corrected = model.observed_distances(block["z"], block["kind"])
        leading = distance_shape(block["z"], block["kind"], .315/(1+s))
        baseline = distance_shape(block["z"], block["kind"], .315)
        comparisons.append({"sample": block["name"],
                            "relative_distance_change_from_leading": (corrected/leading-1).tolist(),
                            "baseline": score(block["y"], block["cov"], baseline, amplitude),
                            "leading": score(block["y"], block["cov"], leading, amplitude),
                            "homogeneous": score(block["y"], block["cov"], corrected, amplitude)})
    z = [.38, .698, 1.48, 1.48]
    kinds = ["DM_over_rs", "DH_over_rs", "DM_over_rs", "DH_over_rs"]
    original = model.observed_distances(z, kinds)
    fine = HomogeneousBackground(s=s, n_u=24, n_x=40).calibrate()
    fine_distances = fine.observed_distances(z, kinds)
    tight = HomogeneousBackground(s=s, rtol=3e-12, atol=3e-13).calibrate()
    convergence = {"redshifts": z, "observables": kinds,
                   "fine_quadrature": fine.diagnostics(),
                   "relative_fine_minus_default": (fine_distances/original-1).tolist(),
                   "relative_tighter_ODE_minus_default": (tight.observed_distances(z, kinds)/original-1).tolist(),
                   "status": "empirical quadrature/ODE refinement, not a rigorous continuum error bound"}
    # Fixed sensitivity cases, not an optimization or fitted dimension law.
    sensitivity = []
    for eta in (0., .1, 1.):
        branch = HomogeneousBackground(s=s, mass_over_h0=100.,
                                       probe=BoundaryProbe(eta=eta, coupling=0.), n_u=24, n_x=40).calibrate()
        values = branch.observed_distances(z, kinds)
        leading = distance_shape(z, kinds, .315/(1+s))
        sensitivity.append({"eta": eta, "mass_over_Href": 100.,
                            "relative_distance_change_from_leading": (values/leading-1).tolist(),
                            "diagnostics": branch.diagnostics()})
    shape_fine = HomogeneousBackground(s=s, mass_over_h0=100., n_u=32, n_x=64).calibrate()
    reference_shape = np.array(sensitivity[1]["relative_distance_change_from_leading"])+1
    convergence["mass100_eta_point1_relative_refinement_32_by_64"] = (
        shape_fine.observed_distances(z, kinds)/distance_shape(z, kinds, .315/(1+s))/reference_shape-1).tolist()
    shape_change = np.max(np.abs(np.array(sensitivity[-1]["relative_distance_change_from_leading"])
                                 -np.array(sensitivity[0]["relative_distance_change_from_leading"])))
    refinement_change = np.max(np.abs(convergence["mass100_eta_point1_relative_refinement_32_by_64"]))
    convergence["mass100_shape_comparison_status"] = (
        "unresolved: dynamic quadrature refinement is not small relative to shape contrast"
        if refinement_change > .01*shape_change else "empirically resolved at one-percent refinement criterion")
    for branch in sensitivity:
        branch["interpretation"] = "finite quadrature diagnostic; dynamic continuum convergence not established"
    return {"s": s, "mass_over_Href": 1000., "eta": .1, "diagnostics": model.diagnostics(),
            "comparisons": comparisons, "convergence": convergence, "shape_sensitivity": sensitivity,
            "rd_status": "external 147.09 Mpc input; no early-universe prediction",
            "initial_history_status": "supplied late-time preparation; initial-state sensitivity not eliminated",
            "all_domain_rmse": None, "scientific_success": False}


if __name__ == "__main__":
    result = report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"diagnostics": result["diagnostics"], "convergence": result["convergence"],
                      "block_diagonal_BAO_delta_chi2": sum(b["homogeneous"]["chi2"]-b["baseline"]["chi2"] for b in result["comparisons"])}, indent=2))
