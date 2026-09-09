"""Conditional linear growth with the same prepared scalar spectrum.

Leading weak-background Einstein-frame approximation, not a Boltzmann solver,
survey f*sigma8, or an observational fit. Local G calibration is explicit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import hyp2f1

from dimension_boundary_probe import BoundaryProbe


def cassini_band(mean=2.1e-5, sigma=2.3e-5, gaussian_z=1.959963984540054):
    """Invert a quoted Gaussian band, NOT a reconstructed confidence analysis.

    Only unscreened spectrum-averaged massless solar response is applicable.
    gamma-1=-2*s/(1+s), s=2*alpha**2 >= 0.
    """
    if not all(map(math.isfinite, (mean, sigma, gaussian_z))) or sigma <= 0 or gaussian_z <= 0:
        raise ValueError("finite mean, positive sigma and z required")
    lo, hi = mean-gaussian_z*sigma, mean+gaussian_z*sigma
    if lo <= -2 or hi < 0 or lo > 0:
        raise ValueError("this helper requires a finite band crossing GR zero")
    return {"lower_gamma_minus_one": lo, "upper_gamma_minus_one": hi,
            "maximum_s": -lo/(2+lo), "role": "retrospective quoted Gaussian band"}


def gr_initial(a, omega):
    """Growing dust+Lambda mode normalized by D/a -> 1 in early matter era."""
    c = (1-omega)/omega
    z = -c*a**3
    f = hyp2f1(1/3, 1, 11/6, z)
    return np.array([a*f, a*(f+(6/11)*z*hyp2f1(4/3, 2, 17/6, z))])


class GrowthBridge:
    def __init__(self, *, probe=None, mass_over_h0=1000., k_over_h0=300., omega_local=.315):
        self.probe = probe if probe is not None else BoundaryProbe(coupling=0.)
        if self.probe.coupling != 0:
            raise ValueError("this branch requires background rank-one coupling zero")
        if not (math.isfinite(mass_over_h0) and math.isfinite(k_over_h0)
                and mass_over_h0 > 0 and k_over_h0 > 0 and 0 < omega_local < 1):
            raise ValueError("positive finite scales and 0<omega_local<1 required")
        self.mass_over_h0, self.k_over_h0 = mass_over_h0, k_over_h0
        self.omega_local = omega_local

    @property
    def mean_mass_squared(self):
        p = self.probe
        return p.m**2+p.a*p.kernel(p.beta)["dimension"]/(2*p.beta)

    def solar_and_lab_bounds(self, *, s, h0_km_s_mpc=67.4, solar_radius_au=10.,
                             lab_radius_m=.01, calibration_radius_m=.1):
        """Point-source calibration diagnostic; rigorous spectral moment bounds.

        Bounds avoid subtracting two floating point values indistinguishable
        from one. Actual Cavendish geometry requires its own response integral.
        """
        if not all(math.isfinite(x) and x >= 0 for x in
                   (s, solar_radius_au, lab_radius_m, calibration_radius_m)):
            raise ValueError("finite nonnegative inputs required")
        if not math.isfinite(h0_km_s_mpc) or h0_km_s_mpc <= 0:
            raise ValueError("positive H0 required")
        inverse_m = self.mass_over_h0*h0_km_s_mpc*1000/3.085677581491367e22/299792458.
        solar_r = inverse_m*solar_radius_au*149597870700.
        lab_r, cal_r = inverse_m*lab_radius_m, inverse_m*calibration_radius_m
        moment = self.mean_mass_squared
        return {"inverse_length_m": inverse_m,
                "solar_1_minus_B_upper": min(1., solar_r*math.sqrt(moment)),
                "solar_1_minus_F_upper": min(1., solar_r**2*moment/2),
                "calibration_1_minus_F_upper": min(1., cal_r**2*moment/2),
                "absolute_calibrated_lab_force_fraction_upper":
                    s*moment*abs(lab_r**2-cal_r**2)/2,
                "calibration_model": "ideal point-source force; not apparatus likelihood"}

    def solve(self, s, *, convention="local_G", f_cal=1., a_initial=.05, samples=61):
        """Identical initial D,D' in both conventions, no normalization at a=1.

        local_G holds measured G_N, physical matter density, H0 fixed to leading
        order A~1; flat Lambda is chosen separately at each parameter point.
        fixed_Einstein_G holds the supplied background fixed for a comparison
        theorem only. Neither includes scalar background stress or frame lag.
        f_cal=1 is the explicitly declared long-range calibration limit.
        """
        if not math.isfinite(s) or s < 0 or not math.isfinite(f_cal) or not 0 <= f_cal <= 1:
            raise ValueError("s>=0 and 0<=f_cal<=1 required")
        if not 0 < a_initial < 1 or not isinstance(samples, int) or samples < 2:
            raise ValueError("0<a_initial<1 and integer samples>=2 required")
        if convention not in ("local_G", "fixed_Einstein_G"):
            raise ValueError("unknown G/background convention")
        omega = self.omega_local/(1+s*f_cal) if convention == "local_G" else self.omega_local
        ns = np.linspace(math.log(a_initial), 0., samples)
        initial = gr_initial(a_initial, self.omega_local)

        def rhs(n, y):
            a_cos = math.exp(n)
            om = omega*a_cos**-3/(omega*a_cos**-3+1-omega)
            z = (self.k_over_h0/(a_cos*self.mass_over_h0))**2
            response = z*self.probe.bare_resolvent(z)[0] if s else 0.
            return [y[1], -(2-1.5*om)*y[1]+1.5*om*(1+s*response)*y[0]]

        solution = solve_ivp(rhs, (ns[0], 0.), initial, t_eval=ns,
                             method="DOP853", rtol=2e-10, atol=2e-12)
        if not solution.success:
            raise RuntimeError(solution.message)
        scale = np.exp(ns)
        baseline = np.array([gr_initial(a, self.omega_local) for a in scale]).T
        e2 = omega*scale**-3+1-omega
        g0 = self.probe.bare_resolvent(0.)[0]
        ln_a_bound = 1.5*s*omega*scale**-3/self.mass_over_h0**2*g0
        ratios = solution.y/baseline
        return {"convention": convention, "s": s, "f_cal": f_cal,
                "omega_E0": omega, "lambda_density_fraction_input": 1-omega,
                "scale_factor": scale.tolist(), "D": solution.y[0].tolist(),
                "D_prime_log_a": solution.y[1].tolist(), "initial_D_Dprime": initial.tolist(),
                "D_ratio_to_GR": ratios[0].tolist(), "Dprime_ratio_to_GR": ratios[1].tolist(),
                "today_D_ratio": float(ratios[0,-1]), "today_Dprime_ratio": float(ratios[1,-1]),
                "today_normalization_multiplier_if_refitted": float(1/ratios[0,-1]),
                "minimum_mass_over_H": float(np.min(self.probe.m*self.mass_over_h0/np.sqrt(e2))),
                "minimum_k_physical_over_H": float(np.min(self.k_over_h0/scale/np.sqrt(e2))),
                "maximum_tracking_abs_ln_A_estimate": float(np.max(ln_a_bound)),
                "background_status": "leading A~1 dust+Lambda truncation; scalar stress omitted",
                "survey_fsigma8": None, "observational_rmse": None, "scientific_success": False}


def report():
    bridge = GrowthBridge()
    band = cassini_band()
    values = [0., band["maximum_s"], .02]
    return {"status": "conditional shared-spectrum growth, not full cosmology",
            "parameters": {"beta": 1., "eta": .1, "b": 1., "m": 1., "internal_a": 1.,
                           "mass_over_h0": 1000., "k_over_h0": 300., "omega_local": .315,
                           "a_initial": .05, "s_definition": "2 alpha^2"},
            "cassini_band": band,
            "long_range_bounds_at_band_limit": bridge.solar_and_lab_bounds(s=values[1]),
            "oversized_s_role": ".02 is a counterexample, incompatible with this Cassini limit",
            "solutions": [bridge.solve(s, convention=c) for c in ("fixed_Einstein_G", "local_G")
                          for s in values], "scientific_success": False}


if __name__ == "__main__":
    result = report()
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"cassini": result["cassini_band"],
                      "bounds": result["long_range_bounds_at_band_limit"],
                      "growth": [{k: r[k] for k in ("convention", "s", "today_D_ratio", "today_Dprime_ratio",
                                                     "maximum_tracking_abs_ln_A_estimate")}
                                 for r in result["solutions"]]}, indent=2))
