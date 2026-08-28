"""Toy PreEq residual-injection ablation.

This is a tiny executable version of ``paper/9_등호이전/07b``.  It is not an
LLM benchmark.  It only verifies that the alpha_phi control/treatment split is
well-defined and can detect a delayed-disambiguation toy effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from reality_stone.clarus.pre_eq import gibbs_reweight, nonselected_residual, normalize_weights


@dataclass(frozen=True)
class ToyGateConfig:
    beta: float = 1.0
    alpha_phi: float = 0.0
    lambda_phi: float = 0.0
    eta_phi: float = 1.0
    shuffle_residual: bool = False


@dataclass(frozen=True)
class ToyGateStep:
    posterior: np.ndarray
    selected: int
    true_label: int
    residual_mass: float
    phi_next: np.ndarray


def _softmax_prior(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    return normalize_weights(np.exp(shifted))


def toy_gate_step(
    logits: np.ndarray,
    base_energy: np.ndarray,
    embeddings: np.ndarray,
    phi: np.ndarray,
    true_label: int,
    config: ToyGateConfig,
) -> ToyGateStep:
    """Run one finite PreEq token/action selection step."""
    logits = np.asarray(logits, dtype=float)
    base_energy = np.asarray(base_energy, dtype=float)
    embeddings = np.asarray(embeddings, dtype=float)
    phi = np.asarray(phi, dtype=float)
    prior = _softmax_prior(logits)
    injection = embeddings @ phi
    raw_energy = base_energy - float(config.alpha_phi) * injection
    energy = raw_energy - float(np.min(raw_energy))
    posterior = gibbs_reweight(prior, energy, beta=float(config.beta))
    selected = int(np.argmax(posterior))

    residual = nonselected_residual(posterior, selected=[selected])
    residual_weights = residual.raw.copy()
    if config.shuffle_residual and residual_weights.size > 1:
        residual_weights = np.roll(residual_weights, 1)
    phi_next = (
        float(config.lambda_phi) * phi
        + float(config.eta_phi) * (residual_weights[:, None] * embeddings).sum(axis=0)
    )
    return ToyGateStep(
        posterior=posterior,
        selected=selected,
        true_label=int(true_label),
        residual_mass=residual.mass,
        phi_next=phi_next,
    )


def delayed_disambiguation_trial(config: ToyGateConfig) -> dict:
    """Two-step task where the non-selected first-step candidate becomes useful."""
    embeddings = np.eye(2, dtype=float)
    phi = np.zeros(2, dtype=float)
    logits_seq = [
        np.array([1.5, 1.0]),  # step 0: label 0 should manifest, label 1 remains residual
        np.array([1.1, 1.0]),  # step 1: weak base preference is wrong; residual can correct it
    ]
    energy_seq = [
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    ]
    truth = [0, 1]

    steps: list[ToyGateStep] = []
    for logits, energy, label in zip(logits_seq, energy_seq, truth):
        step = toy_gate_step(logits, energy, embeddings, phi, label, config)
        steps.append(step)
        phi = step.phi_next

    correct = sum(int(step.selected == step.true_label) for step in steps)
    return {
        "accuracy": correct / len(steps),
        "selected": [step.selected for step in steps],
        "truth": truth,
        "residual_mass": [step.residual_mass for step in steps],
        "final_phi_norm": float(np.linalg.norm(phi)),
    }


def run_ablation() -> dict:
    control = delayed_disambiguation_trial(ToyGateConfig(alpha_phi=0.0))
    treatment = delayed_disambiguation_trial(ToyGateConfig(alpha_phi=3.0))
    shuffled = delayed_disambiguation_trial(ToyGateConfig(alpha_phi=3.0, shuffle_residual=True))
    return {
        "control_alpha_phi_0": control,
        "treatment_alpha_phi_3": treatment,
        "shuffled_residual_control": shuffled,
    }


def main() -> None:
    print(json.dumps(run_ablation(), indent=2))


if __name__ == "__main__":
    main()
