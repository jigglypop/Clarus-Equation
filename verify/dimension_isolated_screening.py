"""Nonlinear isolated-source bound for the existing exponential matter coupling.

Positive spectral Green function, quadratic scalar potential, zero boundary
field, fixed conserved density. Not a chameleon potential or full lab model.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp

from dimension_growth_bridge import cassini_band

C = 299792458.
G_UPPER = 6.67430e-11


def environment_response_bound(*, beta_environment, beta_source,
                               gradient_environment, gradient_source,
                               T=.0155, k_eff=4*math.pi/852.35e-9):
    """Source-on minus environment-only response, compared to linear source.

    beta_i bounds ||T_i||_infinity. gradient_i bounds c^2||grad T_i||.
    Fixed conserved densities, common bare G and zero field at infinity.
    Neither sign nor cancellation of the source-induced field is assumed.
    """
    inputs = (beta_environment,beta_source,gradient_environment,gradient_source)
    if not all(math.isfinite(v) and v >= 0 for v in inputs):
        raise ValueError("finite nonnegative operator/gradient bounds required")
    if not math.isfinite(T) or not math.isfinite(k_eff) or T <= 0 or k_eff <= 0:
        raise ValueError("positive finite pulse parameters required")
    total = beta_environment+beta_source
    if total >= 1:
        raise ValueError("environment plus source contraction bound must be below one")
    field_response = beta_source/(1-beta_environment)
    direct = gradient_source*(-math.expm1(-total))
    feedback = gradient_environment*field_response
    return {"source_induced_field_sup_upper": field_response,
            "direct_nonlinear_source_acceleration_upper_m_s2": direct,
            "induced_environment_acceleration_upper_m_s2": feedback,
            "one_position_response_minus_linear_source_upper_m_s2": direct+feedback,
            "near_minus_far_response_minus_linear_upper_m_s2": 2*(direct+feedback),
            "near_minus_far_phase_upper_rad": 2*k_eff*T*T*(direct+feedback),
            "comparison": "fixed conserved densities and bare G; source-induced nonlinear response minus linear source",
            "includes_environment_feedback": True,
            "full_calibrated_observational_bound": False}


def bounded_environment_example(s):
    """Declared density/support envelope, NOT a measured geophysical profile.

    Any nonnegative environment density <=15000 kg/m^3 supported inside a
    radius 7e6 m ball obeys these Newton-kernel operator bounds. Real material
    density, frame conversion and calibration are not reconstructed here.
    """
    density_max, radius = 15000., 7e6
    isolated = isolated_source_bound(s=s)
    beta_e = 2*math.pi*s*G_UPPER*density_max*radius**2/C**2
    gradient_e = 4*math.pi*s*G_UPPER*density_max*radius
    beta_s = isolated["beta_Newton_potential_upper"]
    gradient_s = G_UPPER*s*isolated["conserved_source_mass_kg_input"]/.003**2
    return {"environment_density_upper_kg_m3_input": density_max,
            "environment_support_radius_m_input": radius,
            "envelope_role": "conditional supplied bounds; no factual Earth/chamber reconstruction",
            "beta_environment_upper": beta_e, "beta_source_upper": beta_s,
            "gradient_environment_upper_m_s2": gradient_e,
            "gradient_source_upper_m_s2": gradient_s,
            "result": environment_response_bound(beta_environment=beta_e,beta_source=beta_s,
                       gradient_environment=gradient_e,gradient_source=gradient_s)}


def isolated_source_bound(*, radius=.0095, bore=.0015, density=2700., s,
                          minimum_distance=.003, T=.0155, k_eff=4*math.pi/852.35e-9):
    if not all(math.isfinite(x) and x >= 0 for x in (radius,bore,density,s)):
        raise ValueError("finite nonnegative source inputs required")
    if not (radius > bore and density > 0 and minimum_distance > 0 and T > 0 and k_eff > 0):
        raise ValueError("valid radius/density/distance/pulse inputs required")
    # Fill the bore for the Newton-potential sup bound; retain its removal in M.
    beta = 2*math.pi*s*G_UPPER*density*radius**2/C**2
    mass = 4*math.pi*density/3*(radius**2-bore**2)**1.5
    fraction = -math.expm1(-beta)  # preserve a tiny nonzero bound
    acceleration = G_UPPER*s*mass*fraction/minimum_distance**2
    return {"beta_Newton_potential_upper": beta,
            "contraction_certified_under_assumptions": beta < 1,
            "fractional_source_density_suppression_upper": fraction,
            "conserved_source_mass_kg_input": mass,
            "one_position_nonlinear_minus_linear_acceleration_upper_m_s2": acceleration,
            "near_minus_far_acceleration_upper_m_s2": 2*acceleration,
            "near_minus_far_phase_upper_rad": 2*k_eff*T*T*acceleration,
            "comparison": "same conserved source density and same bare G; nonlinear minus linear scalar force",
            "G_condition": "bare G_E does not exceed supplied G_upper",
            "boundary": "isolated compact source, scalar field tends to zero at infinity",
            "full_environment_or_calibrated_mass_bound": False}


def radial_fixed_point(beta, *, inverse_range=0., order=192, tolerance=2e-13):
    """Dimensionless uniform spherical source, optional single Yukawa mass.

    beta=2 pi s G rho R^2/c^2. A one-mass kernel checks the nonlinear bound;
    the proof applies to any positive normalized mixture of these kernels.
    Residual certificate concerns this discrete operator, not continuum error.
    """
    if not (0 < beta < 1 and inverse_range >= 0 and order >= 4 and tolerance > 0):
        raise ValueError("0<beta<1, nonnegative range and positive precision required")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    r, weights = (nodes+1)/2, weights/2
    larger = np.maximum(r[:,None],r[None,:])
    smaller = np.minimum(r[:,None],r[None,:])
    if inverse_range == 0:
        angular = 1/larger
    else:
        angular = (np.exp(-inverse_range*(larger-smaller))
                   *(-np.expm1(-2*inverse_range*smaller))/(2*inverse_range*smaller*larger))
    operator = 2*beta*angular*(weights*r*r)[None,:]
    linear = operator@np.ones(order)
    lipschitz = float(np.max(linear))
    if lipschitz >= 1:
        raise ValueError("discrete operator is not a contraction")
    u = np.zeros(order)
    for iteration in range(10000):
        new = operator@np.exp(-u)
        if np.max(np.abs(new-u))/(1-lipschitz) < tolerance:
            u = new
            break
        u = new
    else:
        raise RuntimeError("fixed-point iteration did not converge")
    residual = float(np.max(np.abs(operator@np.exp(-u)-u)))
    return {"radius": r, "u": u, "linear_u": linear, "iterations": iteration+1,
            "discrete_operator_norm": lipschitz,
            "discrete_solution_error_upper_from_residual": residual/(1-lipschitz),
            "relative_total_charge": float(3*np.sum(weights*r*r*np.exp(-u)))}


def independent_massless_bvp(beta, radius):
    """w=r*u, w''=-2 beta r exp(-w/r), w(0)=0, w'(1)=0."""
    mesh = np.linspace(0.,1.,301)
    initial = np.array([beta*mesh*(1-mesh**2/3), beta*(1-mesh**2)])
    def rhs(r,y):
        u = np.divide(y[0],r,out=y[1].copy(),where=r!=0)
        return np.array([y[1], -2*beta*r*np.exp(-u)])
    solution = solve_bvp(rhs,lambda left,right: np.array([left[0],right[1]]),
                         mesh, initial,tol=1e-10,max_nodes=10000)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.sol(radius)[0]/radius


def report():
    physical = isolated_source_bound(s=cassini_band()["maximum_s"])
    checks = []
    for beta in (.1,.8):
        coarse = radial_fixed_point(beta,order=96)
        fine = radial_fixed_point(beta,order=192)
        reference = independent_massless_bvp(beta,fine["radius"])
        coarse_reference = independent_massless_bvp(beta,coarse["radius"])
        checks.append({"beta_artificial_numerical_stress_case": beta,
                       "coarse_error_vs_independent_BVP": float(np.max(np.abs(coarse["u"]-coarse_reference))),
                       "fine_error_vs_independent_BVP": float(np.max(np.abs(fine["u"]-reference))),
                       "discrete_certificate": fine["discrete_solution_error_upper_from_residual"],
                       "relative_total_charge": fine["relative_total_charge"],
                       "iterations": fine["iterations"]})
    return {"physical_isolated_source": physical, "independent_checks": checks,
            "bounded_environment": bounded_environment_example(cassini_band()["maximum_s"]),
            "role": "fixed-source nonlinear bound and independent mathematical checks",
            "actual_experiment_rmse": None, "all_domain_rmse": None, "scientific_success": False}


if __name__ == "__main__":
    result = report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
