"""Tick-fold rule scan: can a per-tick "fold" of a fraction of the state into a
dark sink reproduce the observed dark sector at the FLRW background level?

Exploration only (workspace note 20260902-de-틱접힘_규칙스캔.md). Tolerances are
pre-registered there and duplicated in TOL below; do not change after running.

Units: rho_crit0 = 1, H0 = 1, x = ln a. All integrations are backward in x from
x = 0 (today) to X_MIN, with explicit RK4 (numpy only).
"""
from __future__ import annotations

import itertools
import json
import math
import sys

import numpy as np

# ---------------------------------------------------------------- fiducial
OM_B, OM_C, OM_L, OM_R = 0.0493, 0.2645, 0.6847, 9.1e-5
OM_DARK0 = OM_C + OM_L
X_MIN = math.log(1e-6)
N_STEPS = 1200
Z_GRID = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1100.0])
TOL = dict(single_rho=0.05, single_E=0.02, dm_ratio=0.05, dm_late=0.05,
           de_w=0.10, de_early=0.02, de_E=0.02, growth=0.10)

RATES = ["R1", "R2", "R3", "R4", "R5", "R5b", "R6"]
SOURCES = ["S1", "S3"]
W_SINK = [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0]
CONSERVE = ["transfer", "copy"]
GAMMAS = np.logspace(-8, 2, 21)
A_STARS = np.logspace(-6, -0.3, 8)
F_ONESHOT = [0.5, 0.84, 0.95, 5.4 / 6.4, 19.0]


def lcdm_E2(a: np.ndarray) -> np.ndarray:
    return (OM_B + OM_C) * a ** -3 + OM_R * a ** -4 + OM_L


def lcdm_dark(a: np.ndarray) -> np.ndarray:
    return OM_C * a ** -3 + OM_L


# ---------------------------------------------------------------- rule rhs
def gamma_over_H(rate: str, gamma: float, a: float, E: float, x_star: float, f: float) -> float:
    """Return Gamma/H for the rate family (R5 returns absolute creation / H)."""
    if rate == "R1":
        return gamma
    if rate == "R2":
        return gamma / E
    if rate == "R3":
        return gamma * a ** -3 / E
    if rate == "R4":
        return gamma * a ** -4 / E
    if rate == "R5":
        return gamma  # absolute creation per ln a (rate proportional to H), src == 1
    if rate == "R5b":
        return gamma / E  # absolute creation per unit H0 time, src == 1
    if rate == "R6":  # one-shot: Gaussian bump in x, total fraction f
        sig = 0.05
        x = math.log(a)
        return f * math.exp(-0.5 * ((x - x_star) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    raise ValueError(rate)


def rhs(y: np.ndarray, x: float, rule: dict, lam: float) -> np.ndarray:
    rho_b, rho_r, rho_d = y
    a = math.exp(x)
    E2 = rho_b + rho_r + rho_d + lam
    if E2 <= 0 or not np.isfinite(E2):
        return np.full(3, np.nan)
    E = math.sqrt(E2)
    g = gamma_over_H(rule["rate"], rule["gamma"], a, E, rule["x_star"], rule["f"])
    if rule["rate"] in ("R5", "R5b"):
        src_b, src_r, src = 0.0, 0.0, 1.0
    elif rule["source"] == "S1":
        src_b, src_r, src = rho_b, 0.0, rho_b
    else:
        src_b, src_r, src = rho_b, rho_r, rho_b + rho_r
    take = 1.0 if rule["conserve"] == "transfer" else 0.0
    w = rule["w"]
    d_b = -3 * rho_b - take * g * src_b
    d_r = -4 * rho_r - take * g * src_r
    d_d = -3 * (1 + w) * rho_d + g * src
    return np.array([d_b, d_r, d_d])


def integrate_backward(rule: dict, rho_d0: float, lam: float):
    """RK4 from x=0 down to X_MIN. Returns (x_grid, Y) or None on failure."""
    h = X_MIN / N_STEPS  # negative
    xs = np.empty(N_STEPS + 1)
    Y = np.empty((N_STEPS + 1, 3))
    y = np.array([OM_B, OM_R, rho_d0], dtype=float)
    x = 0.0
    xs[0], Y[0] = x, y
    for i in range(1, N_STEPS + 1):
        k1 = rhs(y, x, rule, lam)
        k2 = rhs(y + 0.5 * h * k1, x + 0.5 * h, rule, lam)
        k3 = rhs(y + 0.5 * h * k2, x + 0.5 * h, rule, lam)
        k4 = rhs(y + h * k3, x + h, rule, lam)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + h
        if not np.all(np.isfinite(y)) or np.any(y < -1e-12):
            return None
        xs[i], Y[i] = x, y
    return xs, Y


def interp(xs: np.ndarray, col: np.ndarray, a_query: np.ndarray) -> np.ndarray:
    xq = np.log(a_query)
    order = np.argsort(xs)
    return np.interp(xq, xs[order], col[order])


# ---------------------------------------------------------------- hypotheses
def test_single(rule: dict) -> dict:
    out = integrate_backward(rule, OM_DARK0, 0.0)
    if out is None:
        return dict(ok=False, reason="nonfinite_or_negative")
    xs, Y = out
    a_q = 1.0 / (1.0 + Z_GRID)
    rho_d = interp(xs, Y[:, 2], a_q)
    rho_b = interp(xs, Y[:, 0], a_q)
    rho_r = interp(xs, Y[:, 1], a_q)
    E2 = rho_b + rho_r + rho_d
    dev_rho = float(np.max(np.abs(rho_d / lcdm_dark(a_q) - 1.0)))
    dev_E = float(np.max(np.abs(np.sqrt(E2 / lcdm_E2(a_q)) - 1.0)))
    ok = dev_rho < TOL["single_rho"] and dev_E < TOL["single_E"]
    return dict(ok=bool(ok), dev_rho=dev_rho, dev_E=dev_E)


def test_dm(rule: dict) -> dict:
    out = integrate_backward(rule, OM_C, OM_L)
    if out is None:
        return dict(ok=False, reason="nonfinite_or_negative")
    xs, Y = out
    a_rec = 1.0 / 1101.0
    ratio0 = OM_C / OM_B
    ratio_rec = float(interp(xs, Y[:, 2], np.array([a_rec]))[0] / interp(xs, Y[:, 0], np.array([a_rec]))[0])
    var = abs(ratio_rec / ratio0 - 1.0)
    # fraction of today's DM produced after recombination: compare free dilution
    rho_d_rec = float(interp(xs, Y[:, 2], np.array([a_rec]))[0])
    free_today = rho_d_rec * a_rec ** (3 * (1 + rule["w"]))
    late = abs(OM_C - free_today) / OM_C
    ok = var < TOL["dm_ratio"] and late < TOL["dm_late"]
    return dict(ok=bool(ok), ratio_var=float(var), late_fraction=float(late))


def test_de(rule: dict) -> dict:
    # CDM separate: fold it into the 'lam' slot is wrong (it dilutes); include as extra dust.
    rule_cdm = dict(rule)
    out = integrate_backward_with_cdm(rule_cdm, OM_L)
    if out is None:
        return dict(ok=False, reason="nonfinite_or_negative")
    xs, Y, cdm = out
    a_q = 1.0 / (1.0 + Z_GRID)
    rho_d = interp(xs, Y[:, 2], a_q)
    rho_b = interp(xs, Y[:, 0], a_q)
    rho_r = interp(xs, Y[:, 1], a_q)
    rho_c = interp(xs, cdm, a_q)
    E2 = rho_b + rho_r + rho_d + rho_c
    dev_E = float(np.max(np.abs(np.sqrt(E2 / lcdm_E2(a_q)) - 1.0)))
    early = float((rho_d / E2)[-1])
    # effective w for z<1 from log-derivative of rho_d
    order = np.argsort(xs)
    xs_s, rd = xs[order], Y[order, 2]
    mask = xs_s > math.log(0.5)
    dlnrho = np.gradient(np.log(np.maximum(rd[mask], 1e-300)), xs_s[mask])
    w_eff = -1.0 - dlnrho / 3.0
    dev_w = float(np.max(np.abs(w_eff + 1.0)))
    ok = dev_w < TOL["de_w"] and early < TOL["de_early"] and dev_E < TOL["de_E"]
    return dict(ok=bool(ok), dev_w=dev_w, early_fraction=early, dev_E=dev_E)


def integrate_backward_with_cdm(rule: dict, rho_d0: float):
    """Same as integrate_backward but with a separate non-interacting CDM dust
    (exactly OM_C a^-3) entering H. Implemented by wrapping rhs."""
    h = X_MIN / N_STEPS
    xs = np.empty(N_STEPS + 1)
    Y = np.empty((N_STEPS + 1, 3))
    cdm = np.empty(N_STEPS + 1)
    y = np.array([OM_B, OM_R, rho_d0], dtype=float)
    x = 0.0
    xs[0], Y[0], cdm[0] = x, y, OM_C

    def f(yv, xv):
        return rhs(yv, xv, rule, OM_C * math.exp(-3 * xv))

    for i in range(1, N_STEPS + 1):
        k1 = f(y, x)
        k2 = f(y + 0.5 * h * k1, x + 0.5 * h)
        k3 = f(y + 0.5 * h * k2, x + 0.5 * h)
        k4 = f(y + h * k3, x + h)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x + h
        if not np.all(np.isfinite(y)) or np.any(y < -1e-12):
            return None
        xs[i], Y[i], cdm[i] = x, y, OM_C * math.exp(-3 * x)
    return xs, Y, cdm


# ---------------------------------------------------------------- growth
def growth(rule: dict | None, a_ini: float = 1e-3, n: int = 4000) -> dict:
    """Linear growth with homogeneous creation at rest: delta' = -theta - G delta,
    theta' = -(1+G) theta - 1.5 Om_m(a) delta   (prime = d/dln a, theta scaled by 1/H).
    rule=None gives LCDM. Uses the H_single background (D = all dark, dust-like part
    clusters; for w=0 sink all of D is matter)."""
    if rule is None:
        def bg(x):
            a = math.exp(x)
            E2 = lcdm_E2(a)
            return E2, (OM_B + OM_C) * a ** -3 / E2, 0.0
    else:
        out = integrate_backward(rule, OM_DARK0, 0.0)
        assert out is not None
        xs, Y = out
        order = np.argsort(xs)
        xs_s, Ys = xs[order], Y[order]

        def bg(x):
            a = math.exp(x)
            rb = np.interp(x, xs_s, Ys[:, 0]); rr = np.interp(x, xs_s, Ys[:, 1]); rd = np.interp(x, xs_s, Ys[:, 2])
            E2 = rb + rr + rd
            E = math.sqrt(E2)
            g = gamma_over_H(rule["rate"], rule["gamma"], a, E, rule["x_star"], rule["f"])
            src = 1.0 if rule["rate"] in ("R5", "R5b") else (rb if rule["source"] == "S1" else rb + rr)
            # creation rate of clustered matter relative to its density
            Gm = g * src / (rb + rd) if rule["w"] == 0.0 else 0.0
            return E2, (rb + rd) / E2 if rule["w"] == 0.0 else rb / E2, Gm

    def dE_dx(x, eps=1e-4):
        return (math.log(bg(x + eps)[0]) - math.log(bg(x - eps)[0])) / (4 * eps)  # d ln E / dx

    def f(v, x):
        d, th = v
        E2, om, G = bg(x)
        dlnE = dE_dx(x)
        # theta := (div v)/(aH); its equation in ln a: th' = -(2 + dlnE + G) th - 1.5 om d
        return np.array([-th - G * d, -(2.0 + dlnE + G) * th - 1.5 * om * d])

    x0, x1 = math.log(a_ini), 0.0
    h = (x1 - x0) / n
    v = np.array([a_ini, -a_ini])  # matter-era growing mode delta ~ a, theta = -delta
    x = x0
    hist = []
    for _ in range(n):
        k1 = f(v, x); k2 = f(v + 0.5 * h * k1, x + 0.5 * h); k3 = f(v + 0.5 * h * k2, x + 0.5 * h); k4 = f(v + h * k3, x + h)
        v = v + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h
        hist.append((x, v[0], v[1]))
    hist = np.array(hist)
    D0 = hist[-1, 1] / a_ini
    i05 = int(np.argmin(np.abs(hist[:, 0] - math.log(1 / 1.5))))
    d05, th05 = hist[i05, 1], hist[i05, 2]
    fD05 = -th05  # f*delta = dδ/dlna = -theta - G delta; report -theta (velocity-based f sigma8 proxy)
    return dict(D0=float(D0), fD05=float(fD05 / a_ini))


# ---------------------------------------------------------------- scan
def all_rules():
    for rate in RATES:
        if rate == "R6":
            for w, cons, src, a_s, f in itertools.product(W_SINK, CONSERVE, SOURCES, A_STARS, F_ONESHOT):
                yield dict(rate=rate, source=src, w=w, conserve=cons, gamma=0.0, x_star=math.log(a_s), f=f)
        elif rate in ("R5", "R5b"):
            for w, g in itertools.product(W_SINK, GAMMAS):
                yield dict(rate=rate, source="const", w=w, conserve="copy", gamma=float(g), x_star=0.0, f=0.0)
        else:
            for w, cons, src, g in itertools.product(W_SINK, CONSERVE, SOURCES, GAMMAS):
                yield dict(rate=rate, source=src, w=w, conserve=cons, gamma=float(g), x_star=0.0, f=0.0)


def main() -> dict:
    results = dict(single=[], dm=[], de=[])
    n_rules = 0
    for rule in all_rules():
        n_rules += 1
        s = test_single(rule)
        if s.get("ok"):
            results["single"].append(dict(rule=rule, **s))
        if rule["w"] == 0.0:
            d = test_dm(rule)
            if d.get("ok"):
                results["dm"].append(dict(rule=rule, **d))
        if rule["w"] <= -1.0 / 3.0:
            e = test_de(rule)
            if e.get("ok"):
                results["de"].append(dict(rule=rule, **e))
    summary = dict(n_rules=n_rules, n_single=len(results["single"]), n_dm=len(results["dm"]), n_de=len(results["de"]))

    # characterise survivors compactly
    def key(r):
        return (r["rule"]["rate"], r["rule"]["source"], r["rule"]["w"], r["rule"]["conserve"])
    fam = {}
    for hyp in ("single", "dm", "de"):
        fam[hyp] = {}
        for r in results[hyp]:
            k = "|".join(str(v) for v in key(r))
            fam[hyp].setdefault(k, []).append(r["rule"]["gamma"] if r["rule"]["rate"] != "R6" else (math.exp(r["rule"]["x_star"]), r["rule"]["f"]))
    summary["families"] = {h: {k: (min(v), max(v), len(v)) if fam[h][k] and not isinstance(v[0], tuple) else (len(v),) for k, v in fam[h].items()} for h in fam}

    # growth for single survivors: one representative per family (median gamma)
    g_l = growth(None)
    summary["growth_lcdm"] = g_l
    summary["growth"] = {}
    for k, v in fam["single"].items():
        reps = [r for r in results["single"] if "|".join(str(t) for t in key(r)) == k]
        rep = sorted(reps, key=lambda r: r["dev_rho"])[0]
        try:
            g = growth(rep["rule"])
            summary["growth"][k] = dict(rule=rep["rule"], D0_ratio=g["D0"] / g_l["D0"], fD05_ratio=g["fD05"] / g_l["fD05"],
                                        pass_growth=bool(abs(g["D0"] / g_l["D0"] - 1) < TOL["growth"] and abs(g["fD05"] / g_l["fD05"] - 1) < TOL["growth"]))
        except AssertionError:
            summary["growth"][k] = dict(rule=rep["rule"], error="background failed")
    return summary


if __name__ == "__main__":
    out = main()
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False, default=str)
    print()
