"""Follow-up scans for H1 critical kappa and H2 occupancy splits."""

from __future__ import annotations

import importlib.util

SPEC = importlib.util.spec_from_file_location(
    "re",
    r"_workspace/ce/agi-universe-sim-life-climb-20260814/artifacts/_route_eval.py",
)
RE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RE)

print("=== H1 dividing-margin kappa scan q- ===")
for i in range(0, 501):
    kappa = i / 1000
    growth = float(RE.R0) * (1 + kappa * (-0.5))
    sol = RE.solve_dividing(growth, RE.LAM0, RE.RHO0)
    if not sol["roots"]:
        print("no root k", kappa)
        break
    root = sol["roots"][0]
    if root["m"] < 0.375 or not root["divides"]:
        print(
            "loses divide k",
            kappa,
            "m",
            root["m"],
            "tilde",
            root["tilde"],
        )
        break

print("=== H1 disc<=0 scan ===")
for i in range(400, 520):
    kappa = i / 1000
    growth = float(RE.R0) * (1 + kappa * (-0.5))
    disc = RE.quadratic_mass(growth, RE.LAM0, RE.RHO0)[3]
    if disc <= 0:
        print("disc<=0 at", kappa, "R", growth, "disc", float(disc))
        break

print("=== H2 T=32 selected points ===")
extra = [
    ("lowb", 0.5, 0.25),
    ("lowb2", 0.45, 0.20),
    ("midlow", 0.42, 0.30),
    ("wedgeish", 0.62, 0.20),
    ("R0sw", 0.4, 4 / 9),
    ("belowR0", 0.38, 0.40),
    ("highm", 0.70, 0.50),
]
for kappa in (0.25, 0.5, 1.0):
    print("k", kappa)
    for name, mass0, bound0 in extra:
        outs = []
        for q in (0.25, 0.75):
            rho = float(RE.RHO0) * (1 + kappa * (2 * q - 1))
            outs.append(
                RE.iterate(mass0, bound0, q, float(RE.R0), float(RE.LAM0), rho, 32)
            )
        low, high = outs
        print(
            name,
            "q-",
            low["alive"],
            low["in_r0"],
            low["divisions"],
            tuple(round(x, 4) for x in low["final"]),
            "q+",
            high["alive"],
            high["in_r0"],
            high["divisions"],
            tuple(round(x, 4) for x in high["final"]),
        )

print("=== H2 T=80 low-b ===")
for kappa in (0.25, 0.5, 1.0):
    for name, mass0, bound0 in (("lowb", 0.5, 0.25), ("lowb2", 0.45, 0.20)):
        for q in (0.25, 0.75):
            rho = float(RE.RHO0) * (1 + kappa * (2 * q - 1))
            out = RE.iterate(
                mass0, bound0, q, float(RE.R0), float(RE.LAM0), rho, 80
            )
            print(
                kappa,
                name,
                q,
                out["divisions"],
                out["alive"],
                out["in_r0"],
                tuple(round(x, 4) for x in out["final"]),
            )

print("=== LEAKm T=32 ===")
for kappa in (0.25, 0.5, 1.0):
    print("k", kappa)
    for name, mass0, bound0 in (
        ("center", 0.5, 0.5),
        ("R0sw", 0.4, 4 / 9),
        ("lowb", 0.5, 0.25),
    ):
        for q in (0.25, 0.75):
            leak = float(RE.LAM0) * (1 - kappa * (2 * q - 1))
            out = RE.iterate(
                mass0, bound0, q, float(RE.R0), leak, float(RE.RHO0), 32
            )
            print(
                name,
                q,
                out["alive"],
                out["in_r0"],
                out["divisions"],
                tuple(round(x, 4) for x in out["final"]),
            )

print("=== H1 T=32 large kappa ===")
for kappa in (0.5, 1.0):
    print("k", kappa)
    for name, mass0, bound0 in (("center", 0.5, 0.5), ("R0sw", 0.4, 4 / 9)):
        for q in (0.25, 0.75):
            growth = float(RE.R0) * (1 + kappa * (2 * q - 1))
            out = RE.iterate(
                mass0, bound0, q, growth, float(RE.LAM0), float(RE.RHO0), 32
            )
            print(
                name,
                q,
                out["alive"],
                out["in_r0"],
                out["divisions"],
                tuple(round(x, 4) for x in out["final"]),
            )

print("=== H2 grid occupancy mismatch count T=32 ===")
for kappa in (0.25, 0.5, 1.0):
    mismatch_alive = 0
    mismatch_r0 = 0
    mismatch_div = 0
    total = 0
    for i in range(11):
        for j in range(11):
            mass0 = 0.2 + 0.06 * i
            bound0 = 0.15 + 0.06 * j
            total += 1
            outs = []
            for q in (0.25, 0.75):
                rho = float(RE.RHO0) * (1 + kappa * (2 * q - 1))
                outs.append(
                    RE.iterate(
                        mass0, bound0, q, float(RE.R0), float(RE.LAM0), rho, 32
                    )
                )
            if outs[0]["alive"] != outs[1]["alive"]:
                mismatch_alive += 1
            if outs[0]["in_r0"] != outs[1]["in_r0"]:
                mismatch_r0 += 1
            if outs[0]["divisions"] != outs[1]["divisions"]:
                mismatch_div += 1
    print(
        "k",
        kappa,
        "total",
        total,
        "alive",
        mismatch_alive,
        "r0",
        mismatch_r0,
        "div",
        mismatch_div,
    )
