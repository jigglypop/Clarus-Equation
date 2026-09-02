"""Smallest verification for examples/physics/tick_fold_rule_scan.py.

Analytic expectation: an absolute creation rate of 3*Omega_Lambda per ln a into a
dust sink reproduces the LCDM background exactly (rho_D = Om_c a^-3 + Om_L), and a
negligible Hubble-tick fold keeps the DM/baryon ratio flat.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "physics"))

import tick_fold_rule_scan as m  # noqa: E402


def test_r5_constant_creation_reproduces_lcdm_background():
    rule = dict(rate="R5", source="const", w=0.0, conserve="copy", gamma=3 * m.OM_L, x_star=0.0, f=0.0)
    r = m.test_single(rule)
    assert r["ok"], r
    assert r["dev_rho"] < 1e-3 and r["dev_E"] < 1e-3


def test_r5_wrong_rate_fails():
    rule = dict(rate="R5", source="const", w=0.0, conserve="copy", gamma=1.5 * m.OM_L, x_star=0.0, f=0.0)
    r = m.test_single(rule)
    assert not r["ok"], r


def test_negligible_fold_keeps_dm_ratio_flat():
    rule = dict(rate="R1", source="S1", w=0.0, conserve="copy", gamma=1e-8, x_star=0.0, f=0.0)
    r = m.test_dm(rule)
    assert r["ok"], r


def test_growth_lcdm_matter_era_normalisation():
    g = m.growth(None)
    # delta grows ~a in matter era; suppression by Lambda gives D(a=1)/a_ini around 0.78 * 1000
    assert 600 < g["D0"] < 900, g
