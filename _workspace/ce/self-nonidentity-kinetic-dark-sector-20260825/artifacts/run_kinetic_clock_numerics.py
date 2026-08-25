"""Reproducible, explicitly post-hoc kinetic-clock background/BAO scan."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

# This artifact is run from the staging mirror before canonical promotion.
# Resolve it from this file, not from pytest's conftest/PYTHONPATH injection.
_WORKSPACE = Path(__file__).resolve().parents[4]
_STAGING_SRC = _WORKSPACE / ".tmp" / "ce-cosmo-dso-20260825" / "src"
if not (_STAGING_SRC / "ce_cosmo").is_dir():
    raise RuntimeError(f"staging ce_cosmo source is unavailable: {_STAGING_SRC}")
sys.path.insert(0, str(_STAGING_SRC))

from ce_cosmo.gates import self_nonidentity_kinetic_dark_sector as kc

KineticClockConfig, profile_desi_bao, solve_background = kc.KineticClockConfig, kc.profile_desi_bao, kc.solve_background

grid=(3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0,8.0,9.0,10.0,12.0,15.0,20.0,30.0)


def _quad(vector, inverse):
    return sum(vector[i]*inverse[i][j]*vector[j] for i in range(len(vector)) for j in range(len(vector)))


def _lcdm_e(z):
    return math.sqrt(kc.OMEGA_B0*(1+z)**3 + kc.OMEGA_DM0*(1+z)**3 + kc.OMEGA_R0*(1+z)**4 + kc.OMEGA_DE0)


def _lcdm_shape(z, intervals=256):
    if intervals % 2: intervals += 1
    dz=z/intervals; total=1/_lcdm_e(0)+1/_lcdm_e(z)
    for i in range(1,intervals): total += (4 if i % 2 else 2)/_lcdm_e(i*dz)
    dm=total*dz/3; dh=1/_lcdm_e(z)
    return dh,dm,(z*dm*dm*dh)**(1/3)


def lcdm_baseline_profile():
    """Pinned-DESI analytic flat LCDM control at exactly the same fractions."""
    data,cov=kc._desi(); inverse=kc._inverse(cov); shapes=[]
    for point in data:
        dh,dm,dv=_lcdm_shape(point.z); shapes.append({"dh":dh,"dm":dm,"dv":dv}[point.kind])
    gg=_quad(shapes,inverse)
    gd=sum(shapes[i]*inverse[i][j]*data[j].value for i in range(len(data)) for j in range(len(data)))
    scale=gd/gg
    if scale <= 0: raise ValueError("LCDM profiled scale must be positive")
    residual=[scale*x-point.value for x,point in zip(shapes,data)]
    chi2=_quad(residual,inverse); k=1
    return {"scale":scale,"chi2":chi2,"dof":len(data)-k,"aic":chi2+2*k,"bic":chi2+k*math.log(len(data))}


def _trajectory_diagnostics(t):
    z=1100.0; n=-math.log1p(z); u=t._at_n(t.u,n)
    ratio=kc.rho_k(u,t.amplitude,t.config.kappa)/(kc.OMEGA_DM0*(1+z)**3)
    present=[i for i,x in enumerate(t.n) if x <= 0.0]
    max_delta=max(t.u[i]/t.config.kappa for i in present)
    max_cs2=max(kc.cs2(t.u[i],t.config.kappa) for i in present)
    # rho+3p switches from positive to negative at the acceleration transition.
    values=[]
    for i in present:
        ni=t.n[i]; ui=t.u[i]; ti=t.tau[i]
        rb=kc.OMEGA_B0*math.exp(-3*ni); rr=kc.OMEGA_R0*math.exp(-4*ni)
        values.append(rb + 2*rr + kc.rho_k(ui,t.amplitude,t.config.kappa) + 3*kc.p_k(ui,t.amplitude,t.config.kappa) - 2*kc.rho_v(ti,t.amplitude,t.config.gamma))
    transition=None
    for i in range(len(values)-1):
        if values[i] >= 0.0 > values[i+1]:
            weight=values[i]/(values[i]-values[i+1]); n_cross=t.n[present[i]]+weight*(t.n[present[i+1]]-t.n[present[i]])
            transition=math.exp(-n_cross)-1.0; break
    return {"rho_k_over_dust_z1100":ratio,"max_delta_a_1e-4_to_1":max_delta,"max_cs2_a_1e-4_to_1":max_cs2,
            "future_current_reserve_q_N10_over_q_initial":t.current[-1]/t.current[0],"acceleration_transition_z":transition}


baseline=lcdm_baseline_profile()
rows=[]
for gamma in grid:
    try:
        t=solve_background(KineticClockConfig(gamma=gamma,steps=800,future_steps=800))
        p=profile_desi_bao(t)
        row={
            "gamma": gamma, "A": t.amplitude, "chi2": p.chi2, "scale": p.scale,
            "dof_fixed_gamma": p.dof, "posthoc_scan_dof": 11,
            "posthoc_scan_aic": p.chi2 + 4.0,
            "posthoc_scan_bic": p.chi2 + 2.0 * math.log(13),
            "current_fd": t.current_fd_residual,
            "continuity_fd": t.continuity_fd_residual, "stable": True,
            "future_tail_bound": t.future_tail_bound,
            "global_current_margin": t.global_current_margin,
        }
        row["delta_chi2_vs_lcdm"]=row["chi2"]-baseline["chi2"]
        row["delta_aic_vs_lcdm"]=row["posthoc_scan_aic"]-baseline["aic"]
        row["delta_bic_vs_lcdm"]=row["posthoc_scan_bic"]-baseline["bic"]
        row.update(_trajectory_diagnostics(t))
        rows.append(row)
    except (ArithmeticError,ValueError) as error:
        rows.append({"gamma":gamma,"stable":False,"reason":str(error)})
payload={"label":"posthoc boundary-calibrated gamma scan; not a prediction; selecting gamma consumes one extra parameter",
         "parameter_accounting":{"lcdm_k":1,"lcdm_dof":12,"kinetic_scan_k":2,"kinetic_scan_dof":11},
         "normalization":{"X_star":.5,"tau":"H0*T","gamma":"Gamma/H0","u":"kappa*delta","A":"rho_inf/rho_crit0"},
         "lcdm_same_fraction_baseline":baseline,"rows":rows}
Path(__file__).with_name("numerical-results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
