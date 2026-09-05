"""접착 전 길이 불일치의 측도와 기록·조건부 선택·이완을 구분한다.

Post hoc diagnostic of supplied models, not a physical CE selection law.
The Gaussian uses unglued cell lengths, with no gauge or gluing imposed.
The finite channels are controls, not derived Regge dynamics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from conditional_composition import reference as r, row_basis
from examples.physics.record.coarse_observation import projective_dephasing

HERE = Path(__file__).resolve().parent


def apply_channel(kraus, state):
    return sum(k @ state @ k.conj().T for k in kraus)


def soft_record_operators(labels, resolution):
    """Binary QND record: only conditioning on outcome zero gives soft selection."""
    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 1 or not np.isfinite(labels).all():
        raise ValueError("labels must be a finite vector")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    exponent = -0.5 * (labels / resolution)**2
    return np.diag(np.exp(exponent / 2)), np.diag(np.sqrt(-np.expm1(exponent)))


def relaxation_operators(dimension, probability):
    """Supplied amplitude damping to label zero; the target is an explicit input."""
    if dimension < 2 or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid relaxation dimension or probability")
    diagonal = np.full(dimension, math.sqrt(1 - probability))
    diagonal[0] = 1
    operators = [np.diag(diagonal)]
    for label in range(1, dimension):
        jump = np.zeros((dimension, dimension))
        jump[0, label] = math.sqrt(probability)
        operators.append(jump)
    return tuple(operators)


def gaussian_mismatch(spectrum, resolution):
    """Exact moments of a supplied centered Gaussian after a soft likelihood."""
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 1 or not len(spectrum) or not np.isfinite(spectrum).all() or np.any(spectrum <= 0):
        raise ValueError("spectrum must be finite and positive")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    log_ratio = np.log(spectrum) - 2 * math.log(resolution)
    log_denominator = np.logaddexp(0, log_ratio)
    posterior = np.exp(np.log(spectrum) - log_denominator)
    return {
        "resolution": resolution,
        "prior_total_squared_mismatch": float(spectrum.sum()),
        "posterior_total_squared_mismatch": float(posterior.sum()),
        "posterior_per_mode_squared_mismatch": float(posterior.mean()),
        "log_acceptance": -0.5 * float(log_denominator.sum()),
    }


def chain_boundary_amplitude(sites, time):
    """Boundary survival for H=3I+adjacency on a finite chain; time is g*t/hbar."""
    if not isinstance(sites, int) or isinstance(sites, bool) or sites < 2:
        raise ValueError("sites must be an integer >= 2")
    if not math.isfinite(time) or time < 0:
        raise ValueError("time must be finite and nonnegative")
    wave_numbers = np.arange(1, sites + 1) * math.pi / (sites + 1)
    weights = 2 * np.sin(wave_numbers)**2 / (sites + 1)
    return complex(np.sum(weights * np.exp(-1j * (3 + 2*np.cos(wave_numbers)) * time)))


def autonomous_bath_control():
    times = (0., 1., 5., 10., 100.)
    primary = [abs(chain_boundary_amplitude(2048, time))**2 for time in times]
    check = [abs(chain_boundary_amplitude(4096, time))**2 for time in times]
    # A boundary detuning of 2g creates a bound state, even on an infinite chain.
    sites, detuning = 64, 2.
    ratio = 1 / detuning
    state = math.sqrt(1 - ratio**2) * ratio**np.arange(sites)
    hamiltonian = 3*np.eye(sites) + np.diag(np.ones(sites-1), 1) + np.diag(np.ones(sites-1), -1)
    hamiltonian[0, 0] += detuning
    energy = 3 + detuning + 1/detuning
    return {
        "model": "vacuum plus one excitation in a supplied semi-infinite hopping chain",
        "dimensionless_time": list(times), "finite_chain_sites": 2048,
        "boundary_survival_probability": primary,
        "4096_site_probability_difference": float(np.max(np.abs(np.array(primary) - check))),
        "environment_reset": False, "time_dependent_drive": False,
        "infinite_uniform_chain_local_decay": "integrable boundary spectral density implies amplitude tends to zero",
        "finite_chain_infinite_time_decay_claimed": False,
        "detuned_negative_control": {
            "boundary_detuning_over_hopping": detuning,
            "localized_state_energy_over_hopping": energy,
            "finite_truncation_eigenvector_residual": float(np.linalg.norm(hamiltonian @ state - energy*state)),
            "infinite_chain_boundary_survival_limit": (1 - ratio**2)**2,
        },
        "metric_mismatch_to_excitation_map_supplied": True,
        "new_CE_dynamics_derived": False,
    }


def finite_controls():
    labels = np.array([0., 1., 2.])
    cost = np.diag(labels**2)
    state = np.ones((3, 3)) / 3
    record = soft_record_operators(labels, 0.5)
    nonselective = apply_channel(record, state)
    selected = record[0] @ state @ record[0].T
    probability = float(np.trace(selected))
    relaxation = relaxation_operators(3, 0.6)
    relaxed = apply_channel(relaxation, state)
    # Each environment label carries the lost excitation energy.
    isometry = np.stack(relaxation, axis=1).reshape(9, 3)
    total_energy = np.kron(cost, np.eye(3)) + np.kron(np.eye(3), cost)
    mean = lambda rho: float(np.trace(cost @ rho).real)
    unitary = np.eye(9)
    cosine, sine = math.sqrt(0.4), math.sqrt(0.6)
    for label in (1, 2):
        system_excited, environment_excited = 3 * label, label
        unitary[np.ix_([system_excited, environment_excited], [system_excited, environment_excited])] = [
            [cosine, -sine], [sine, cosine],
        ]
    environment = np.diag([1., 0., 0.])
    joint = np.kron(state, environment)
    reset_state = state.copy()
    reuse, reset = [], []
    for _ in range(4):
        joint = unitary @ joint @ unitary.T
        reduced = np.trace(joint.reshape(3, 3, 3, 3), axis1=1, axis2=3)
        reset_state = apply_channel(relaxation, reset_state)
        reuse.append(mean(reduced))
        reset.append(mean(reset_state))
    return {
        "labels": labels.tolist(), "initial_probabilities": np.diag(state).tolist(),
        "initial_mismatch": mean(state),
        "projective_record_mismatch": mean(projective_dephasing(state)),
        "soft_record_unconditional_mismatch": mean(nonselective),
        "soft_record_conditioned_mismatch": mean(selected / probability),
        "soft_record_acceptance": probability,
        "record_completeness_residual": float(np.linalg.norm(sum(k.T @ k for k in record) - np.eye(3))),
        "unconditional_population_residual": float(np.linalg.norm(np.diag(state) - np.diag(nonselective))),
        "relaxation_probability": 0.6, "relaxed_mismatch": mean(relaxed),
        "relaxation_completeness_residual": float(np.linalg.norm(isometry.T @ isometry - np.eye(3))),
        "relaxation_energy_intertwining_residual": float(np.linalg.norm(total_energy @ isometry - isometry @ cost)),
        "collision_unitarity_residual": float(np.linalg.norm(unitary.T @ unitary - np.eye(9))),
        "collision_energy_commutator_residual": float(np.linalg.norm(total_energy @ unitary - unitary @ total_energy)),
        "same_environment_mismatch_steps_1_to_4": reuse,
        "fresh_environment_mismatch_steps_1_to_4": reset,
        "relaxation_target_and_energy_supplied": True,
    }


def unglued_model(depth, step):
    points = r.points_from_squared(np.full(10, 2.0))
    cells = [tuple(range(5))]
    for _ in range(depth):
        cells = r.refine(cells, points)
    kappas = r.equal_split_kappas(cells, tuple(range(5)), np.full(10, math.pi))
    dimension = 10 * len(cells)
    covariance = np.zeros((dimension, dimension))
    for i, (cell, kappa) in enumerate(zip(cells, kappas)):
        lengths = r.cell_lengths(cell, points)
        hessian = r.richardson_hessian(lambda x: r.simplex_action(x, kappa), lengths, step)
        values, vectors = np.linalg.eigh(hessian)
        if np.min(np.abs(values)) <= 0:
            raise ValueError("singular supplied leaf precision")
        covariance[10*i:10*(i+1), 10*i:10*(i+1)] = (vectors / np.abs(values)) @ vectors.T
    # Orthonormal mismatch metric, not an arbitrary first-owner row penalty.
    basis = row_basis(r.gluing_rows(cells))
    mismatch = basis.T @ covariance @ basis
    spectrum = np.linalg.eigvalsh((mismatch + mismatch.T) / 2)
    rank = basis.shape[1]
    return {
        "depth": depth, "cells": len(cells), "unglued_coordinates": dimension,
        "mismatch_rank": rank, "no_gluing_or_gauge_conditioned": True,
        "mismatch_metric": "orthogonal projector in supplied occurrence-length coordinates",
        "spectrum_min": float(spectrum.min()), "spectrum_max": float(spectrum.max()),
        "prior_mean_squared_mismatch": float(spectrum.mean()),
        "exact_gluing_probability": 0.0,
        "soft_conditioning": [gaussian_mismatch(spectrum, eps) for eps in (0.1, 1., 10.)],
        "fixed_rank_concentration_requires_resolution_to_zero": True,
        "physical_resolution_law_derived": False,
    }


def run(step=2e-5):
    controls = finite_controls()
    if max(controls[k] for k in (
        "record_completeness_residual", "unconditional_population_residual",
        "relaxation_completeness_residual", "relaxation_energy_intertwining_residual",
        "collision_unitarity_residual", "collision_energy_commutator_residual",
    )) > 1e-12:
        raise RuntimeError("finite channel control failed")
    return {
        "scope": "supplied Gaussian and finite-channel mechanism diagnostics; not a new physical prediction",
        "python": sys.version.split()[0], "numpy": np.__version__, "fd_step": step,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),
        "finite_controls": controls,
        "autonomous_bath_control": autonomous_bath_control(),
        "unglued_models": [unglued_model(depth, step) for depth in (1, 2)],
        "physical_common_metric_selection_proved": False,
        "sources": ["https://arxiv.org/abs/quant-ph/0312059", "https://arxiv.org/abs/0803.1447", "https://arxiv.org/abs/quant-ph/0611164"],
    }


if __name__ == "__main__":
    result = run()
    (HERE / "selection_mechanisms.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
