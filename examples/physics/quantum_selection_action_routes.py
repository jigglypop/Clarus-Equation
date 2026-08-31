r"""Fail-closed checks for action-first routes from quantum selection to dark readout.

The module is deliberately a *route audit*, not a claim that ordinary quantum
probabilities already source cosmological stress energy.  Four short no-go
examples remove common shortcuts.  The remaining finite construction states
exactly what an action-first bridge can prove after its physical maps are
supplied.

Units are :math:`c=\hbar=1`.  Every phase, probability, Gram minor and action
integrand is dimensionless after the explicitly recorded reference scales are
used; ``E_star`` has mass dimension one, ``cell_volume`` dimension minus three,
and a four-form flux ``f`` dimension two.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


MINKOWSKI = np.diag(np.array([-1.0, 1.0, 1.0, 1.0]))


def _real(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _probability(value: float, *, name: str) -> float:
    value = _real(value, name=name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def feshbach_phase_counterexample(
    *, q: float = 0.37, detuning: float = 3.0, coupling: float = 2.0,
    phase_a: float = 0.0, phase_b: float = math.pi / 3.0,
) -> dict[str, Any]:
    """Same Hamiltonian and Born weight, but different energy expectation.

    For H=[[0,g],[g,Delta]] and
    |psi_phi>=sqrt(1-q)|P>+exp(i phi)sqrt(q)|Q>, the expectation contains an
    interference term.  Thus q alone cannot be promoted to q E_star until a
    physical record has removed the off-diagonal receipt terms.
    """
    q = _probability(q, name="q")
    delta = _real(detuning, name="detuning")
    v = _real(coupling, name="coupling")
    if v == 0.0:
        raise ValueError("coupling must be nonzero for the phase counterexample")
    phases = (_real(phase_a, name="phase_a"), _real(phase_b, name="phase_b"))

    hamiltonian = np.array([[0.0, v], [v, delta]], dtype=complex)

    def expectation(phi: float) -> float:
        state = np.array(
            [math.sqrt(1.0 - q), np.exp(1j * phi) * math.sqrt(q)],
            dtype=complex,
        )
        return float(np.vdot(state, hamiltonian @ state).real)

    energy_a, energy_b = expectation(phases[0]), expectation(phases[1])
    return {
        "route": "FESHBACH_2X2_PHASE",
        "selection_weight_q": q,
        "phase_a": phases[0],
        "phase_b": phases[1],
        "phase_is_dimensionless": True,
        "energy_expectation_phase_a": energy_a,
        "energy_expectation_phase_b": energy_b,
        "same_q": True,
        "same_hamiltonian": True,
        "different_energy_expectation": not math.isclose(
            energy_a, energy_b, rel_tol=1e-12, abs_tol=1e-12
        ),
        "counterexample": "q does not determine a distinct energy receipt",
    }


def stinespring_zero_energy_counterexample(*, q: float = 0.37) -> dict[str, Any]:
    """A complete orthogonal record can be made with a zero Hamiltonian.

    ``V|psi> = sqrt(1-q)|0,0> + sqrt(q)|1,1>`` is an isometry and supplies a
    perfectly distinguishable environment record.  Taking H=0 gives zero
    energy before and after it, so record completeness alone is not energy.
    """
    q = _probability(q, name="q")
    vector = np.array([math.sqrt(1.0 - q), 0.0, 0.0, math.sqrt(q)], dtype=complex)
    hamiltonian = np.zeros((4, 4), dtype=complex)
    record_overlap = 0.0  # <0_env|1_env>
    return {
        "route": "STINESPRING_COMPLETE_RECORD",
        "q": q,
        "isometry_norm": float(np.vdot(vector, vector).real),
        "orthogonal_records": record_overlap == 0.0,
        "hamiltonian_zero": bool(np.allclose(hamiltonian, 0.0)),
        "energy_expectation": float(np.vdot(vector, hamiltonian @ vector).real),
        "counterexample": "complete record does not imply folded energy",
    }


def dephasing_trajectory_counterexample() -> dict[str, Any]:
    """Two inequivalent zero-energy unravelings of the same complete dephasing.

    The random-unitary Kraus pair ``I/sqrt(2), Z/sqrt(2)`` and the projective
    pair ``P0, P1`` have the same channel, but tell different trajectory
    stories.  With H=0 neither story fixes an energy cost.
    """
    identity = np.eye(2, dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    p0 = np.diag([1.0, 0.0]).astype(complex)
    p1 = np.diag([0.0, 1.0]).astype(complex)
    random_unitary = (identity / math.sqrt(2.0), z / math.sqrt(2.0))
    projective = (p0, p1)
    rho = np.array([[0.31, 0.22 - 0.13j], [0.22 + 0.13j, 0.69]], dtype=complex)

    def channel(kraus: Iterable[np.ndarray]) -> np.ndarray:
        return sum((k @ rho @ k.conj().T for k in kraus), np.zeros((2, 2), complex))

    output_a, output_b = channel(random_unitary), channel(projective)
    return {
        "route": "DEPHASING_TRAJECTORY",
        "same_cptp_channel": bool(np.allclose(output_a, output_b, atol=1e-12)),
        "different_trajectory_labels": True,
        "zero_hamiltonian": True,
        "energy_expectation_each_trajectory": 0.0,
        "counterexample": "a dephasing trajectory is not a unique energy ledger",
    }


def controlled_flux_storage_witness(
    *, interaction_angle: float = math.pi / 4.0, energy_star: float = 1.0
) -> dict[str, Any]:
    """Store the locked Born weight in a finite, energy-conserving flux bit.

    The basis is |selection, flux, battery>.  A controlled SWAP acts on the
    equal-gap flux and battery bits only when the selection bit is locked.
    Starting from

        (sin(theta)|mobile> + cos(theta)|locked>) |0_flux,1_battery>

    gives a flux occupation cos(theta)^2 while preserving one total receipt.
    This closes a finite branch-to-storage map.  It does not decide whether
    gravity should use the unconditional state or a branch-conditioned state,
    and a flux bit is not yet a covariant four-form field.
    """
    theta = _real(interaction_angle, name="interaction_angle")
    if not 0.0 <= theta <= math.pi / 2.0:
        raise ValueError("interaction_angle must lie in [0, pi/2]")
    gap = _real(energy_star, name="energy_star")
    if gap <= 0.0:
        raise ValueError("energy_star must be positive")

    def index(selection: int, flux: int, battery: int) -> int:
        return 4 * selection + 2 * flux + battery

    unitary = np.eye(8, dtype=complex)
    locked_battery = index(1, 0, 1)
    locked_flux = index(1, 1, 0)
    unitary[locked_battery, locked_battery] = 0.0
    unitary[locked_flux, locked_flux] = 0.0
    unitary[locked_battery, locked_flux] = 1.0
    unitary[locked_flux, locked_battery] = 1.0

    initial = np.zeros(8, dtype=complex)
    initial[index(0, 0, 1)] = math.sin(theta)
    initial[index(1, 0, 1)] = math.cos(theta)
    final = unitary @ initial
    flux_number = np.diag(
        [float(flux) for selection in (0, 1) for flux in (0, 1) for battery in (0, 1)]
    )
    battery_number = np.diag(
        [float(battery) for selection in (0, 1) for flux in (0, 1) for battery in (0, 1)]
    )
    hamiltonian = gap * (flux_number + battery_number)
    identity = np.eye(8, dtype=complex)
    p_mobile = math.sin(theta) ** 2
    q_locked = math.cos(theta) ** 2
    flux_occupation = float(np.vdot(final, flux_number @ final).real)
    battery_occupation = float(np.vdot(final, battery_number @ final).real)
    initial_energy = float(np.vdot(initial, hamiltonian @ initial).real)
    final_energy = float(np.vdot(final, hamiltonian @ final).real)
    unitary_residual = float(np.linalg.norm(unitary.conj().T @ unitary - identity))
    commutator_residual = float(np.linalg.norm(unitary @ hamiltonian - hamiltonian @ unitary) / gap)
    return {
        "route": "CONTROLLED_FLUX_STORAGE",
        "interaction_angle": theta,
        "angle_is_dimensionless": True,
        "p_mobile": p_mobile,
        "q_locked": q_locked,
        "flux_occupation": flux_occupation,
        "battery_occupation": battery_occupation,
        "born_transfer_residual": abs(flux_occupation - q_locked),
        "unitary_residual": unitary_residual,
        "relative_energy_commutator_residual": commutator_residual,
        "initial_total_energy": initial_energy,
        "final_total_energy": final_energy,
        "flux_energy_expectation": gap * flux_occupation,
        "battery_energy_expectation": gap * battery_occupation,
        "one_receipt_no_double_count": math.isclose(
            gap * (flux_occupation + battery_occupation), gap, abs_tol=1e-12
        ),
        "receipt_interpretation": {
            "ensemble_expectation_partition": True,
            "per_run_two_simultaneous_energy_receipts": False,
            "each_run_total_receipt_energy": gap,
        },
        "finite_branch_to_storage_closed": (
            unitary_residual <= 1e-12
            and commutator_residual <= 1e-12
            and abs(flux_occupation - q_locked) <= 1e-12
            and math.isclose(initial_energy, final_energy, abs_tol=1e-12)
        ),
        "supplied_conditions": [
            "the selector is degenerate in the receipt Hamiltonian",
            "the flux and battery bits have the same positive energy gap",
            "the selection basis is the supplied pointer basis",
        ],
        "conditioning_boundary": {
            "mobile_branch_conditional_flux_occupation": 0.0,
            "locked_branch_conditional_flux_occupation": 1.0,
            "unconditional_flux_occupation": flux_occupation,
            "unconditioned_semiclassical_gravity_rule_derived": False,
            "branch_local_gravity_contains_nonselected_flux": False,
        },
        "fail_closed": {
            "finite_flux_bit_is_covariant_four_form": False,
            "q_value_predicted_from_first_principles": False,
            "absolute_cosmological_scale_predicted": False,
            "universal_quantum_storage_law_proved": False,
        },
    }


def time_varying_vacuum_nonconservation(*, q: float = 0.4, q_dot: float = 0.1,
                                        energy_density_scale: float = 2.0) -> dict[str, Any]:
    """Test ``T=-rho(q) g``: unless rho is constant it is not conserved.

    Here ``q_dot`` has inverse-time/mass dimension one and ``rho_scale`` mass
    dimension four.  The reported divergence is ``d rho/dt`` in flat space.
    """
    q = _probability(q, name="q")
    q_dot = _real(q_dot, name="q_dot")
    rho_scale = _real(energy_density_scale, name="energy_density_scale")
    rho = rho_scale * q
    drho_dt = rho_scale * q_dot
    t_covariant = -rho * MINKOWSKI
    divergence = np.array([drho_dt, 0.0, 0.0, 0.0])
    return {
        "route": "TIME_VARYING_Q_VACUUM",
        "rho": rho,
        "T_covariant": t_covariant.tolist(),
        "nabla_mu_T_mu_nu": divergence.tolist(),
        "conserved": bool(np.allclose(divergence, 0.0, atol=1e-12)),
        "counterexample": "T=-rho(q)g needs constant rho or an exchange current",
    }


def gram_minors(rods: Iterable[Iterable[float]]) -> dict[str, float]:
    """Return the dimensionless relation-rod Gram minors D1,D2,D3."""
    matrix = np.asarray(list(rods), dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rods must be exactly three independent 3-component rods")
    gram = matrix @ matrix.T
    return {f"D{k}": float(np.linalg.det(gram[:k, :k])) for k in (1, 2, 3)}


@dataclass(frozen=True)
class ActionFirstHybrid:
    """Finite conditional bridge, with all non-derived maps named as assumptions."""

    q_mobile: float
    energy_star: float
    cell_volume: float
    rods: tuple[tuple[float, float, float], ...]

    def certificate(self) -> dict[str, Any]:
        p = _probability(self.q_mobile, name="q_mobile")
        e_star = _real(self.energy_star, name="energy_star")
        volume = _real(self.cell_volume, name="cell_volume")
        if e_star <= 0.0 or volume <= 0.0:
            raise ValueError("energy_star and cell_volume must be positive")
        minors = gram_minors(self.rods)
        if min(minors.values()) <= 0.0:
            raise ValueError("the action-first witness needs three independent rods")
        locked = 1.0 - p
        e_mobile, e_locked = p * e_star, locked * e_star
        rho_mobile, rho_locked = e_mobile / volume, e_locked / volume
        # S_F = -1/(2*4!) integral sqrt(-g) F^2; F=f epsilon gives rho=f^2/2.
        flux = math.sqrt(2.0 * rho_locked)
        t_dust = np.diag([rho_mobile, 0.0, 0.0, 0.0])
        t_vacuum = -rho_locked * MINKOWSKI
        action_density_dimension = 4  # [F^2]=M^4, cancels [d^4x]=M^-4.
        return {
            "route": "ACTION_FIRST_HYBRID",
            "relation_rod_gram_minors": minors,
            "three_direction_witness": all(value > 0.0 for value in minors.values()),
            "q_mobile": p,
            "q_locked": locked,
            "one_receipt_energy": e_star,
            "mobile_receipt_energy": e_mobile,
            "locked_receipt_energy": e_locked,
            "no_double_count": bool(math.isclose(e_mobile + e_locked, e_star, abs_tol=1e-12)),
            "receipt_interpretation": {
                "ensemble_expectation_partition": True,
                "per_run_two_simultaneous_energy_receipts": False,
            },
            "mobile_massive_receipt": {
                "rho": rho_mobile,
                "T_covariant": t_dust.tolist(),
                "form": "T_mn=rho_m u_m u_n, u=(1,0,0,0)",
                "component_frame": "local comoving orthonormal frame",
                "conditional_dust": True,
                "flrw_continuity_law_derived_by_this_finite_witness": False,
            },
            "locked_four_form": {
                "action": "S_F=-1/(2*4!) integral d^4x sqrt(-g) F_mnrs F^mnrs",
                "flux_f": flux,
                "rho": rho_locked,
                "T_covariant": t_vacuum.tolist(),
                "form": "T_mn=-rho_L g_mn",
                "equation_of_motion": "d(*F)=0 => d f=0",
                "constant_flux_conserved": True,
                "cell_volume_role": "fixed matching reference volume, not an expanding physical cell",
            },
            "dimensionless_audit": {
                "rod_coordinates_and_Dk_dimensionless": True,
                "q_and_branch_angle_dimensionless": True,
                "E_star_cubed_times_cell_volume_dimensionless": True,
                "four_form_action_dimensionless": action_density_dimension == 4,
            },
            "adopted_maps": [
                "three relation rods are read as spatial directions",
                "mobile receipt admits a positive monokinetic massive mass-shell map",
                "locked receipt is represented by a constant four-form flux sector",
                "E_star/cell_volume fixes the matching density before four-form evolution",
            ],
            "fail_closed": {
                "flux_is_fixed_sector_or_initial_condition": True,
                "born_q_does_not_dynamically_derive_covariant_four_form": True,
                "cosmological_branch_persistence_is_not_derived": True,
                "absolute_dark_scale_is_input": True,
                "real_cosmological_dark_sector_identity_proved": False,
            },
        }


def action_first_hybrid_witness(*, q_mobile: float = 0.3, energy_star: float = 5.0,
                                cell_volume: float = 2.0,
                                rods: Iterable[Iterable[float]] | None = None) -> dict[str, Any]:
    """Build the strongest finite action-first witness available to this audit."""
    if rods is None:
        rods = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    normalized_rods = tuple(tuple(map(float, rod)) for rod in rods)
    return ActionFirstHybrid(q_mobile, energy_star, cell_volume, normalized_rods).certificate()


def certificate() -> dict[str, Any]:
    """One fail-closed certificate for all alternatives and the surviving bridge."""
    feshbach = feshbach_phase_counterexample()
    stinespring = stinespring_zero_energy_counterexample()
    dephasing = dephasing_trajectory_counterexample()
    varying = time_varying_vacuum_nonconservation()
    p_mobile = 0.3
    energy_star = 5.0
    cell_volume = 2.0
    branch_angle = math.asin(math.sqrt(p_mobile))
    flux_storage = controlled_flux_storage_witness(
        interaction_angle=branch_angle, energy_star=energy_star
    )
    hybrid = action_first_hybrid_witness(
        q_mobile=p_mobile, energy_star=energy_star, cell_volume=cell_volume
    )
    joint_probability_residual = max(
        abs(flux_storage["p_mobile"] - hybrid["q_mobile"]),
        abs(flux_storage["q_locked"] - hybrid["q_locked"]),
    )
    joint_energy_residual = max(
        abs(flux_storage["flux_energy_expectation"] - hybrid["locked_receipt_energy"]),
        abs(flux_storage["battery_energy_expectation"] - hybrid["mobile_receipt_energy"]),
    )
    return {
        "status": "MAJOR_ALTERNATIVE_ROUTES_AUDITED_FINITE_FLUX_STORAGE_CLOSED_COVARIANT_BRIDGE_OPEN",
        "counterexamples_closed": {
            "feshbach_phase": feshbach["same_q"] and feshbach["different_energy_expectation"],
            "stinespring_record": stinespring["orthogonal_records"] and stinespring["hamiltonian_zero"],
            "dephasing_trajectory": dephasing["same_cptp_channel"] and dephasing["zero_hamiltonian"],
            "time_varying_vacuum": not varying["conserved"],
        },
        "conditional_action_first_hybrid_closed": hybrid["no_double_count"] and hybrid["locked_four_form"]["constant_flux_conserved"],
        "finite_branch_to_flux_storage_closed": flux_storage["finite_branch_to_storage_closed"],
        "finite_storage_to_action_partition_consistent": (
            joint_probability_residual <= 1e-12 and joint_energy_residual <= 1e-12
        ),
        "joint_probability_residual": joint_probability_residual,
        "joint_energy_residual": joint_energy_residual,
        "unconditional_quantum_to_dark_sector_proved": False,
        "physical_bridge_open": [
            "derive the grade/branch to massive receipt map",
            "lift the finite flux bit to a covariant four-form field",
            "derive ensemble-to-cosmology coarse graining without branch double counting",
            "derive the gravitational conditioning rule for nonselected records",
            "derive dust continuity and the matching-reference-volume prescription",
            "predict q from microscopic couplings rather than choose an angle",
            "fix the absolute cosmological scale and confront observations",
        ],
        "routes": {
            "feshbach": feshbach,
            "stinespring": stinespring,
            "dephasing": dephasing,
            "controlled_flux_storage": flux_storage,
            "time_varying_vacuum": varying,
            "action_first_hybrid": hybrid,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    args = parser.parse_args()
    print(json.dumps(certificate(), indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
