"""Independent verifier for the P0.2 stoichiometric protocell ledger.

No simulator module is imported.  The registered reaction graph, event
schedules, ledgers, units, and division geometry are rebuilt here and compared
fail-closed with the submitted certificate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


RAW, PRECURSOR, CHARGED, DISCHARGED, WASTE, BOUNDARY = "R", "P", "F", "D", "W", "B"
TEMPLATES = tuple(f"T{index}" for index in range(7))
CELL_SPECIES = (RAW, PRECURSOR, CHARGED, DISCHARGED, WASTE, BOUNDARY, *TEMPLATES)
ENVIRONMENT_SPECIES = (RAW, DISCHARGED, WASTE)
MATERIAL = {
    RAW: 1,
    PRECURSOR: 1,
    CHARGED: 0,
    DISCHARGED: 0,
    WASTE: 1,
    BOUNDARY: 1,
    **{template: 1 for template in TEMPLATES},
}
CARRIER = {
    RAW: 0,
    PRECURSOR: 0,
    CHARGED: 1,
    DISCHARGED: 1,
    WASTE: 0,
    BOUNDARY: 0,
    **{template: 0 for template in TEMPLATES},
}
MOLECULAR_VOLUME_M3 = 1.0e-28
BOUNDARY_AREA_PER_MOLECULE_M2 = 1.0e-18


@dataclass(frozen=True)
class _Reaction:
    name: str
    reactants: tuple[tuple[str, int], ...]
    products: tuple[tuple[str, int], ...]
    rate: Fraction

    @property
    def order(self) -> int:
        return sum(value for _, value in self.reactants)


@dataclass(frozen=True)
class StoichiometricVerificationReport:
    verified: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]


def _q(compartment: str, species: str) -> str:
    return f"{compartment}:{species}"


def _rxn(
    name: str,
    reactants: Mapping[str, int],
    products: Mapping[str, int],
    rate: Fraction,
) -> _Reaction:
    return _Reaction(name, tuple(sorted(reactants.items())), tuple(sorted(products.items())), rate)


def _reactions() -> tuple[_Reaction, ...]:
    cell = lambda species: _q("cell", species)
    env = lambda species: _q("environment", species)
    values = [
        _rxn(
            "carrier_recharge",
            {cell(RAW): 1, cell(DISCHARGED): 1},
            {cell(WASTE): 1, cell(CHARGED): 1},
            Fraction(1, 8),
        ),
        _rxn(
            "precursor_activation",
            {cell(RAW): 1, cell(CHARGED): 1},
            {cell(PRECURSOR): 1, cell(DISCHARGED): 1},
            Fraction(1, 16),
        ),
        _rxn(
            "boundary_synthesis",
            {cell(PRECURSOR): 1, cell(CHARGED): 1},
            {cell(BOUNDARY): 1, cell(DISCHARGED): 1},
            Fraction(1, 32),
        ),
        _rxn(
            "maintenance_discharge",
            {cell(CHARGED): 1},
            {cell(DISCHARGED): 1},
            Fraction(1, 64),
        ),
        _rxn(
            "precursor_decay",
            {cell(PRECURSOR): 1},
            {cell(WASTE): 1},
            Fraction(1, 128),
        ),
        _rxn(
            "boundary_decay",
            {cell(BOUNDARY): 1},
            {cell(WASTE): 1},
            Fraction(1, 256),
        ),
        _rxn("raw_uptake", {env(RAW): 1}, {cell(RAW): 1}, Fraction(1, 16)),
        _rxn(
            "discharged_carrier_uptake",
            {env(DISCHARGED): 1},
            {cell(DISCHARGED): 1},
            Fraction(1, 16),
        ),
        _rxn("waste_export", {cell(WASTE): 1}, {env(WASTE): 1}, Fraction(1, 16)),
    ]
    for template in TEMPLATES:
        values.extend(
            [
                _rxn(
                    f"copy_{template}",
                    {cell(template): 1, cell(PRECURSOR): 1, cell(CHARGED): 1},
                    {cell(template): 2, cell(DISCHARGED): 1},
                    Fraction(1, 64),
                ),
                _rxn(
                    f"decay_{template}",
                    {cell(template): 1},
                    {cell(WASTE): 1},
                    Fraction(1, 512),
                ),
            ]
        )
    return tuple(values)


def _base(qualified: str) -> str:
    parts = qualified.split(":", 1)
    if len(parts) != 2:
        raise ValueError("invalid qualified species")
    compartment, species = parts
    allowed = CELL_SPECIES if compartment == "cell" else ENVIRONMENT_SPECIES
    if compartment not in {"cell", "environment"} or species not in allowed:
        raise ValueError("unknown qualified species")
    return species


def _total(state: Mapping[str, int], weights: Mapping[str, int]) -> int:
    return sum(weights[_base(species)] * count for species, count in state.items())


def _balance(reaction: _Reaction) -> tuple[int, int]:
    material = carrier = 0
    for sign, side in ((-1, reaction.reactants), (1, reaction.products)):
        for species, coefficient in side:
            base = _base(species)
            material += sign * MATERIAL[base] * coefficient
            carrier += sign * CARRIER[base] * coefficient
    return material, carrier


def _apply(state: Mapping[str, int], reaction: _Reaction, events: int = 1) -> dict[str, int]:
    if events < 1:
        raise ValueError("events must be positive")
    updated = dict(state)
    for species, coefficient in reaction.reactants:
        required = coefficient * events
        if updated.get(species, 0) < required:
            raise ValueError("insufficient reactant")
    for species, coefficient in reaction.reactants:
        updated[species] -= coefficient * events
    for species, coefficient in reaction.products:
        updated[species] = updated.get(species, 0) + coefficient * events
    if min(updated.values()) < 0:
        raise AssertionError("negative reference count")
    return updated


def _state() -> dict[str, int]:
    state = {
        **{_q("cell", species): 0 for species in CELL_SPECIES},
        **{_q("environment", species): 0 for species in ENVIRONMENT_SPECIES},
    }
    state.update(
        {
            _q("cell", RAW): 128,
            _q("cell", PRECURSOR): 96,
            _q("cell", CHARGED): 128,
            _q("cell", DISCHARGED): 128,
            _q("cell", WASTE): 0,
            _q("cell", BOUNDARY): 80,
            _q("environment", RAW): 256,
            _q("environment", DISCHARGED): 256,
            _q("environment", WASTE): 0,
            **{_q("cell", template): 4 for template in TEMPLATES},
        }
    )
    return state


def _closed() -> dict[str, object]:
    by_name = {reaction.name: reaction for reaction in _reactions()}
    state = _state()
    initial_state = dict(state)
    initial_material = _total(state, MATERIAL)
    initial_carrier = _total(state, CARRIER)
    schedule = [
        ("raw_uptake", 8),
        ("discharged_carrier_uptake", 8),
        ("carrier_recharge", 16),
        ("precursor_activation", 16),
        *[(f"copy_{template}", 2) for template in TEMPLATES],
        ("boundary_synthesis", 8),
        ("maintenance_discharge", 4),
        *[(f"decay_{template}", 1) for template in TEMPLATES],
        ("boundary_decay", 2),
        ("precursor_decay", 2),
        ("waste_export", 10),
    ]
    for name, events in schedule:
        state = _apply(state, by_name[name], events)
    final_material = _total(state, MATERIAL)
    final_carrier = _total(state, CARRIER)
    return {
        "passed": final_material == initial_material and final_carrier == initial_carrier,
        "event_batches": len(schedule),
        "initial_state": initial_state,
        "schedule": [{"reaction": name, "events": events} for name, events in schedule],
        "initial_material": initial_material,
        "final_material": final_material,
        "initial_carrier": initial_carrier,
        "final_carrier": final_carrier,
        "material_residual": final_material - initial_material,
        "carrier_residual": final_carrier - initial_carrier,
        "minimum_final_count": min(state.values()),
    }


def _open() -> dict[str, object]:
    by_name = {reaction.name: reaction for reaction in _reactions()}
    state = _state()
    initial_state = dict(state)
    initial_material = _total(state, MATERIAL)
    initial_carrier = _total(state, CARRIER)
    intervention = {
        "raw_inflow_per_tick": 4,
        "discharged_carrier_inflow_per_tick": 4,
        "waste_outflow_per_tick": 1,
    }
    schedule_label = [
        "raw_uptake",
        "raw_uptake",
        "discharged_carrier_uptake",
        "discharged_carrier_uptake",
        "carrier_recharge",
        "precursor_activation",
        "copy_T[tick mod 7]",
        "boundary_synthesis",
        "maintenance_discharge",
        "waste_export",
    ]
    material_in = material_out = carrier_in = carrier_out = 0
    for tick in range(16):
        state[_q("environment", RAW)] += intervention["raw_inflow_per_tick"]
        material_in += intervention["raw_inflow_per_tick"]
        state[_q("environment", DISCHARGED)] += intervention[
            "discharged_carrier_inflow_per_tick"
        ]
        carrier_in += intervention["discharged_carrier_inflow_per_tick"]
        schedule = [
            "raw_uptake",
            "raw_uptake",
            "discharged_carrier_uptake",
            "discharged_carrier_uptake",
            "carrier_recharge",
            "precursor_activation",
            f"copy_{TEMPLATES[tick % 7]}",
            "boundary_synthesis",
            "maintenance_discharge",
            "waste_export",
        ]
        for name in schedule:
            state = _apply(state, by_name[name])
        outflow = intervention["waste_outflow_per_tick"]
        state[_q("environment", WASTE)] -= outflow
        if state[_q("environment", WASTE)] < 0:
            raise AssertionError("reference outflow exceeded waste")
        material_out += outflow
    final_material = _total(state, MATERIAL)
    final_carrier = _total(state, CARRIER)
    material_residual = final_material - (initial_material + material_in - material_out)
    carrier_residual = final_carrier - (initial_carrier + carrier_in - carrier_out)
    return {
        "passed": material_residual == 0 and carrier_residual == 0,
        "ticks": 16,
        "initial_state": initial_state,
        "intervention": intervention,
        "reaction_schedule_per_tick": schedule_label,
        "schedule_constant_after_initialization": True,
        "initial_material": initial_material,
        "final_material": final_material,
        "material_in": material_in,
        "material_out": material_out,
        "material_residual": material_residual,
        "initial_carrier": initial_carrier,
        "final_carrier": final_carrier,
        "carrier_in": carrier_in,
        "carrier_out": carrier_out,
        "carrier_residual": carrier_residual,
        "minimum_final_count": min(state.values()),
    }


def _geometry(counts: Mapping[str, int]) -> dict[str, float]:
    volume = sum(counts.values()) * MOLECULAR_VOLUME_M3
    area = counts[BOUNDARY] * BOUNDARY_AREA_PER_MOLECULE_M2 / 2.0
    sphere_area = (36.0 * math.pi) ** (1.0 / 3.0) * volume ** (2.0 / 3.0)
    return {
        "volume_m3": volume,
        "membrane_area_m2": area,
        "minimum_sphere_area_m2": sphere_area,
        "membrane_slack": area / sphere_area,
    }


def _division() -> dict[str, object]:
    parent = {
        RAW: 20,
        PRECURSOR: 20,
        CHARGED: 20,
        DISCHARGED: 20,
        WASTE: 20,
        BOUNDARY: 80,
        **{template: 4 for template in TEMPLATES},
    }
    equal = {species: count // 2 for species, count in parent.items()}
    parent_geometry = _geometry(parent)
    daughter_geometry = _geometry(equal)
    expected_slack = parent_geometry["membrane_slack"] / 2.0 ** (1.0 / 3.0)
    equal_pass = (
        math.isclose(parent_geometry["volume_m3"], 2 * daughter_geometry["volume_m3"], rel_tol=1e-12)
        and math.isclose(
            parent_geometry["membrane_area_m2"],
            2 * daughter_geometry["membrane_area_m2"],
            rel_tol=1e-12,
        )
        and math.isclose(daughter_geometry["membrane_slack"], expected_slack, rel_tol=1e-12)
    )
    residual_max = 0
    nonnegative = True
    parent_material = sum(MATERIAL[s] * n for s, n in parent.items())
    parent_carrier = sum(CARRIER[s] * n for s, n in parent.items())
    for seed in range(64):
        rng = random.Random(seed)
        daughters = ({s: 0 for s in CELL_SPECIES}, {s: 0 for s in CELL_SPECIES})
        for species, count in parent.items():
            for _ in range(count):
                daughters[rng.randrange(2)][species] += 1
        first, second = daughters
        residual = sum(abs(first[s] + second[s] - parent[s]) for s in CELL_SPECIES)
        residual += abs(
            sum(MATERIAL[s] * first[s] for s in CELL_SPECIES)
            + sum(MATERIAL[s] * second[s] for s in CELL_SPECIES)
            - parent_material
        )
        residual += abs(
            sum(CARRIER[s] * first[s] for s in CELL_SPECIES)
            + sum(CARRIER[s] * second[s] for s in CELL_SPECIES)
            - parent_carrier
        )
        residual_max = max(residual_max, residual)
        nonnegative = nonnegative and min(first.values()) >= 0 and min(second.values()) >= 0
    return {
        "passed": equal_pass and residual_max == 0 and nonnegative,
        "parent_counts": parent,
        "parent_geometry": parent_geometry,
        "equal_daughter_geometry": daughter_geometry,
        "equal_split_slack_relation": "alpha_daughter=alpha_parent/2^(1/3)",
        "stochastic_partition_trials": 64,
        "maximum_species_or_ledger_residual": residual_max,
        "hidden_division_injection": 0,
        "all_daughter_counts_nonnegative": nonnegative,
    }


def _reaction_row(reaction: _Reaction) -> dict[str, object]:
    material, carrier = _balance(reaction)
    order = reaction.order
    unit = "s^-1" if order == 1 else f"molecule^{1-order} s^-1"
    return {
        "name": reaction.name,
        "reactants": dict(reaction.reactants),
        "products": dict(reaction.products),
        "order": order,
        "rate_constant": {"exact": str(reaction.rate), "decimal": float(reaction.rate)},
        "rate_constant_unit": unit,
        "material_delta": material,
        "carrier_delta": carrier,
    }


def _expected() -> dict[str, object]:
    reactions = _reactions()
    rows = [_reaction_row(reaction) for reaction in reactions]
    stoich = all(row["material_delta"] == 0 and row["carrier_delta"] == 0 for row in rows)
    units = all(reaction.order >= 1 and reaction.rate > 0 for reaction in reactions)
    no_spontaneous = all(
        any(species == _q("cell", template) for species, _ in reaction.reactants)
        for template in TEMPLATES
        for reaction in reactions
        if any(species == _q("cell", template) for species, _ in reaction.products)
    )
    closed = _closed()
    opened = _open()
    division = _division()
    shortage = False
    try:
        _apply({key: 0 for key in _state()}, {r.name: r for r in reactions}["carrier_recharge"])
    except ValueError:
        shortage = True
    positivity = shortage and closed["minimum_final_count"] >= 0 and opened["minimum_final_count"] >= 0
    gates = {
        "reaction_stoichiometry": {
            "passed": stoich,
            "reactions": rows,
            "material_weights": MATERIAL,
            "carrier_weights": CARRIER,
        },
        "si_dimension_manifest": {
            "passed": units,
            "time": "s",
            "amount": "molecule count",
            "volume": "m^3",
            "area": "m^2",
            "molecular_volume_m3": MOLECULAR_VOLUME_M3,
            "boundary_area_per_molecule_m2": BOUNDARY_AREA_PER_MOLECULE_M2,
        },
        "closed_batch_ledger": closed,
        "open_flow_ledger": opened,
        "positivity_without_clipping": {
            "passed": positivity,
            "insufficient_reactant_event_rejected": shortage,
            "negative_count_repair_used": False,
        },
        "no_spontaneous_template": {
            "passed": no_spontaneous,
            "rule": "every reaction producing T_l also consumes T_l as catalyst",
        },
        "division_geometry_and_conservation": division,
        "constant_external_intervention": {
            "passed": opened["schedule_constant_after_initialization"],
            "schedule": opened["intervention"],
            "per_generation_manual_reset": False,
        },
    }
    all_passed = all(gate["passed"] for gate in gates.values())
    return {
        "artifact_type": "clarus_stoichiometric_protocell_ledger_certificate",
        "artifact_version": 1,
        "arithmetic": "integer molecule events, exact conserved ledgers, SI geometry readout",
        "model": {
            "cell_species": list(CELL_SPECIES),
            "environment_species": list(ENVIRONMENT_SPECIES),
            "module_count": 7,
            "material_ledger": "R+P+W+B+sum(T_l)",
            "carrier_ledger": "F+D",
            "geometry": "V=v_molecule*sum(N); A=a_boundary*N_B/2",
        },
        "gates": gates,
        "claim_scope": {
            "registered_reactions_are_stoichiometrically_balanced": stoich,
            "closed_and_open_ledgers_verified": closed["passed"] and opened["passed"],
            "division_has_no_hidden_material_or_template_injection": division["passed"],
            "positivity_is_enforced_without_clipping": positivity,
            "autonomous_metabolism_proven": False,
            "thermodynamic_feasibility_calibrated": False,
            "mature_offspring_supercriticality_proven": False,
            "empirical_autonomous_protocell_proven": False,
        },
        "all_stoichiometric_gates_passed": all_passed,
    }


def _finite(value: object, path: str = "certificate") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def verify_stoichiometric_certificate(
    certificate: Mapping[str, object],
) -> StoichiometricVerificationReport:
    checks: list[str] = []
    errors: list[str] = []
    try:
        if not isinstance(certificate, Mapping):
            raise ValueError("certificate must be a mapping")
        _finite(certificate)
        expected = _expected()
        if set(certificate) != set(expected):
            raise ValueError("top-level schema differs from the registered certificate")
        checks.append("schema_and_finite_numbers")
        for section in (
            "artifact_type",
            "artifact_version",
            "arithmetic",
            "model",
            "gates",
            "claim_scope",
            "all_stoichiometric_gates_passed",
        ):
            if certificate.get(section) != expected[section]:
                raise ValueError(f"{section} differs from independent recomputation")
            checks.append(section)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
    return StoichiometricVerificationReport(not errors, tuple(checks), tuple(errors))


def independently_verified(certificate: Mapping[str, object]) -> bool:
    return verify_stoichiometric_certificate(certificate).verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
    report = verify_stoichiometric_certificate(payload)
    print(json.dumps({"verified": report.verified, "checks": report.checks, "errors": report.errors}))
    return int(not report.verified)


if __name__ == "__main__":
    raise SystemExit(main())
