"""Finite source and ideal light-pulse readout of the shared scalar force.

Published dimensions constrain a representative geometry, not a reconstructed
Hamilton2015 run. No measured initial trajectory or raw fringe likelihood is used.
"""
import json
import math
from pathlib import Path

import numpy as np

from dimension_boundary_probe import BoundaryProbe
from dimension_growth_bridge import cassini_band

HBAR = 1.054571817e-34
G_N = 6.67430e-11


def drilled_sphere(radius=.0095, bore=.0015, density=2700., *, order=20, azimuths=32):
    """Uniform sphere with a cylindrical through-hole along z; SI units.

    Density is a supplied nominal value, not a measured source mass.
    """
    if not (0 <= bore < radius and density > 0 and order >= 2 and azimuths >= 4):
        raise ValueError("valid source dimensions and integration orders required")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    zmax = math.sqrt(radius**2-bore**2)
    points, masses = [], []
    angles = 2*math.pi*np.arange(azimuths)/azimuths
    for z, wz in zip(nodes*zmax, weights*zmax):
        rmax = math.sqrt(radius**2-z*z)
        radial = bore+(nodes+1)*(rmax-bore)/2
        wr = weights*(rmax-bore)/2
        for r, weight in zip(radial, wr):
            points.extend(np.column_stack((r*np.cos(angles), r*np.sin(angles), np.full(azimuths,z))))
            masses.extend(np.full(azimuths, density*r*weight*wz*2*math.pi/azimuths))
    return np.asarray(points), np.asarray(masses)


def source_acceleration(position, source_points, source_masses, *, g=G_N):
    displacement = source_points-np.asarray(position)
    radius2 = np.sum(displacement**2, axis=1)
    if (radius2 == 0).any():
        raise ValueError("observer on a point quadrature node")
    return g*np.sum(displacement*(source_masses/radius2**1.5)[:, None], axis=0)


def pulse_phase(acceleration, mean_trajectory, *, T=.0155, k_eff=4*math.pi/852.35e-9,
                atom_mass=132.90545196*1.66053906660e-27, direction=(0.,0.,-1.),
                time_order=16, arm_order=4):
    """First-order perturbing phase about closed unperturbed pointlike paths.

    Exact arm average at this perturbative order; not midpoint acceleration.
    Ideal instantaneous pulses; no cloud, finite pulses, recoil-systematics fit.
    """
    if not all(math.isfinite(v) and v > 0 for v in (T, k_eff, atom_mass)):
        raise ValueError("finite positive interferometer parameters required")
    e = np.asarray(direction, dtype=float)
    if e.shape != (3,) or not np.isfinite(e).all() or not np.isclose(np.linalg.norm(e), 1.):
        raise ValueError("unit readout direction required")
    times, wt = np.polynomial.legendre.leggauss(time_order)
    arm_nodes, arm_weights = np.polynomial.legendre.leggauss(arm_order)
    recoil_velocity = HBAR*k_eff/atom_mass
    phase = 0.
    for left in (0., T):
        for t, time_weight in zip(left+(times+1)*T/2, wt*T/2):
            f = min(t, 2*T-t)
            average = 0.
            center = np.asarray(mean_trajectory(t))
            for u, weight in zip(arm_nodes/2, arm_weights/2):
                average += weight*float(e@acceleration(center+u*recoil_velocity*f*e))
            phase += k_eff*time_weight*f*average
    return phase


def calibrated_linear_signal_bound(*, source_mass, inverse_length, s, minimum_distance,
                                    calibration_distance=.1, probe=None, T=.0155,
                                    k_eff=4*math.pi/852.35e-9):
    """Conservative linear-vacuum force/phase bound for any positive source.

    Applies if EVERY sampled arm/material separation exceeds minimum_distance.
    Calibration is ideal point-source force at calibration_distance. It does
    not bound nonlinear density response, environmental changes or laser shifts.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (source_mass,inverse_length,s,calibration_distance)):
        raise ValueError("finite nonnegative source/scale/coupling required")
    if not all(math.isfinite(v) and v > 0 for v in (minimum_distance,T,k_eff)):
        raise ValueError("positive distance/time/wavenumber required")
    p = probe if probe is not None else BoundaryProbe(coupling=0.)
    moment = p.m**2+p.a*p.kernel(p.beta)["dimension"]/(2*p.beta)
    one_position = .5*G_N*s*source_mass*inverse_length**2*moment*(1+(calibration_distance/minimum_distance)**2)
    return {"mean_internal_mass_squared": moment,
            "one_source_acceleration_upper_m_s2": one_position,
            "near_minus_far_acceleration_upper_m_s2": 2*one_position,
            "near_minus_far_phase_upper_rad": k_eff*T*T*2*one_position,
            "assumptions": "linear vacuum propagation; same ideal G calibration; prescribed closed arms",
            "complete_device_bound": False}


def representative_geometry(order=20, azimuths=32):
    radius, bore, density = .0095, .0015, 2700.
    points, masses = drilled_sphere(radius, bore, density, order=order, azimuths=azimuths)
    T, k = .0155, 4*math.pi/852.35e-9
    # Effective paper distance is NOT an initial coordinate. This deliberately
    # supplied example places the mean-path apex at that height at t=T.
    mean = lambda t: np.array([0., 0., radius+.0088-.5*9.81*(t-T)**2])
    phases = []
    for shift in (np.zeros(3), np.array([.03,0.,0.])):
        acceleration = lambda position: source_acceleration(position, points+shift, masses)
        phases.append(pulse_phase(acceleration, mean, T=T, k_eff=k))
    mass_atom = 132.90545196*1.66053906660e-27
    recoil = HBAR*k/mass_atom
    # For this trajectory every segment point is above z>=R+this value;
    # translating the source horizontally cannot reduce that vertical gap.
    gap_bound = .0088-.5*9.81*T*T-.5*recoil*T
    return {"source_mass_kg_nominal": float(masses.sum()), "near_phase_rad_Newton": phases[0],
            "far_phase_rad_Newton": phases[1], "near_minus_far_phase_rad_Newton": phases[0]-phases[1],
            "effective_Newton_acceleration_m_s2": (phases[0]-phases[1])/(k*T*T),
            "source_nodes": len(masses), "source_order": order, "azimuths": azimuths,
            "arm_to_material_distance_lower_bound_m": gap_bound,
            "role": "representative unmeasured apex trajectory; not a reconstructed experimental run"}


def report():
    coarse = representative_geometry()
    fine = representative_geometry(28,48)
    if fine["arm_to_material_distance_lower_bound_m"] < .003:
        raise ValueError("representative arms violate the distance bound")
    s = cassini_band()["maximum_s"]
    inverse_length = 1000.*67.4*1000/3.085677581491367e22/299792458.
    bound = calibrated_linear_signal_bound(source_mass=fine["source_mass_kg_nominal"],
                                           inverse_length=inverse_length, s=s, minimum_distance=.003)
    return {"representative": fine, "source_refinement_phase_difference_rad":
            fine["near_minus_far_phase_rad_Newton"]-coarse["near_minus_far_phase_rad_Newton"],
            "same_cosmological_mass_over_H0": 1000., "s": s, "linear_bound": bound,
            "reported_acceleration_sigma_m_s2": 3.7e-6,
            "linear_bound_over_reported_sigma": bound["near_minus_far_acceleration_upper_m_s2"]/3.7e-6,
            "actual_experiment_rmse": None, "all_domain_rmse": None,
            "limitations": ["published effective distance is not an initial-state distribution",
                            "near/far positions are approximate; source mass and density are not independently measured here",
                            "finite pulses, wavepacket distribution, reversal and nuisance covariance omitted",
                            "linear-vacuum cancellation bound is not a bound on full nonlinear/environmental corrections"],
            "scientific_success": False}


if __name__ == "__main__":
    result = report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
