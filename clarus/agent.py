"""Agent loop: Critic, Action, Bootstrap, Consciousness, Working Memory.

Implements 17_AgentLoop.md F.4, F.7, F.9-F.10, F.17, F.20.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from .constants import (
        BOOTSTRAP_CONTRACTION, ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        CRITIC_W_PRED, CRITIC_W_CONS, CRITIC_W_NOV,
        CONSCIOUSNESS_TAU, CONSCIOUSNESS_CD, META_MAX_DEPTH,
        WM_CAPACITY, CEREBELLUM_ALPHA, CEREBELLUM_ETA, NORM_EPS,
    )
except ImportError:
    from clarus.constants import (
        BOOTSTRAP_CONTRACTION, ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        CRITIC_W_PRED, CRITIC_W_CONS, CRITIC_W_NOV,
        CONSCIOUSNESS_TAU, CONSCIOUSNESS_CD, META_MAX_DEPTH,
        WM_CAPACITY, CEREBELLUM_ALPHA, CEREBELLUM_ETA, NORM_EPS,
    )


# ---------------------------------------------------------------------------
# F.4  Critic
# ---------------------------------------------------------------------------
@dataclass
class CriticResult:
    c_pred: float
    c_cons: float
    c_nov: float
    score: float


def compute_critic(
    observation: torch.Tensor,
    prediction: torch.Tensor,
    relaxed_state: torch.Tensor,
    recalled_state: torch.Tensor,
    obs_prior: torch.Tensor | None = None,
    w_pred: float = CRITIC_W_PRED,
    w_cons: float = CRITIC_W_CONS,
    w_nov: float = CRITIC_W_NOV,
) -> CriticResult:
    """c_{t+1} = w_p||c_pred|| + w_c||c_cons|| + w_n||c_nov|| (F.4)."""
    c_pred = float((observation - prediction).norm().item())
    c_cons = float((relaxed_state - recalled_state).norm().item())
    if obs_prior is not None and obs_prior.numel() == observation.numel():
        p_obs = torch.softmax(observation.float(), dim=-1).clamp(min=1e-8)
        p_prior = torch.softmax(obs_prior.float(), dim=-1).clamp(min=1e-8)
        c_nov = float(torch.sum(p_obs * (p_obs.log() - p_prior.log())).item())
        c_nov = max(c_nov, 0.0)
    else:
        c_nov = 0.0
    score = w_pred * c_pred + w_cons * c_cons + w_nov * c_nov
    return CriticResult(c_pred=c_pred, c_cons=c_cons, c_nov=c_nov, score=score)


# ---------------------------------------------------------------------------
# F.7  Action selection
# ---------------------------------------------------------------------------
def select_action_discrete(
    z_out: torch.Tensor,
    action_embeddings: torch.Tensor,
) -> int:
    """a_t = argmax sim(enc(a), z_out) (F.7 discrete)."""
    z_norm = z_out / z_out.norm().clamp(min=NORM_EPS)
    a_norm = action_embeddings / action_embeddings.norm(dim=1, keepdim=True).clamp(min=NORM_EPS)
    similarity = a_norm @ z_norm
    return int(similarity.argmax().item())


def select_action_continuous(
    z_out: torch.Tensor,
    w_act: torch.Tensor,
    b_act: torch.Tensor,
) -> torch.Tensor:
    """a_t = W_act z_out + b_act (F.7 continuous)."""
    return w_act @ z_out + b_act


# ---------------------------------------------------------------------------
# F.9-F.10  Bootstrap operator
# ---------------------------------------------------------------------------
def bootstrap_operator(
    state: torch.Tensor,
    target: torch.Tensor | None = None,
    rho: float = BOOTSTRAP_CONTRACTION,
) -> torch.Tensor:
    """B: X -> p* + rho*(X - p*) (F.9)."""
    if target is None:
        target = torch.zeros_like(state)
    return target + rho * (state - target)


def agent_step(
    x_t: torch.Tensor,
    relax_delta: torch.Tensor,
    obs_delta: torch.Tensor,
    critic_delta: torch.Tensor,
    suppress_delta: torch.Tensor,
    lambda_r: float = 0.3,
    lambda_o: float = 0.2,
    lambda_c: float = 0.2,
    lambda_s: float = 0.1,
    rho: float = BOOTSTRAP_CONTRACTION,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """X_{t+1} = B[X_t + lambda_R*R + lambda_O*dO + lambda_C*C - lambda_S*S] (F.9)."""
    inner = (
        x_t
        + lambda_r * relax_delta
        + lambda_o * obs_delta
        + lambda_c * critic_delta
        - lambda_s * suppress_delta
    )
    return bootstrap_operator(inner, target=target, rho=rho)


# ---------------------------------------------------------------------------
# F.17  Consciousness / Metacognition
# ---------------------------------------------------------------------------
class ConsciousnessMonitor:
    """C3 self-consistency monitor and consciousness depth tracker."""

    def __init__(
        self,
        tau: float = CONSCIOUSNESS_TAU,
        c_d: float = CONSCIOUSNESS_CD,
        max_depth: int = META_MAX_DEPTH,
        rho: float = BOOTSTRAP_CONTRACTION,
    ) -> None:
        self.tau = tau
        self.c_d = c_d
        self.max_depth = max_depth
        self.rho = rho
        self._deviation_history: deque[float] = deque(maxlen=int(tau))

    def record_deviation(self, active_frac: float, target: float = ACTIVE_RATIO) -> None:
        p_current = torch.tensor([active_frac, 1.0 - active_frac])
        p_target = torch.tensor([target, 1.0 - target])
        dev = float((p_current - p_target).norm().item())
        self._deviation_history.append(dev)

    def d_tau(self) -> float:
        """d_tau(t) = (1/tau) * integral of ||p(s) - p*|| ds (F.17.2)."""
        if not self._deviation_history:
            return 0.0
        return sum(self._deviation_history) / max(len(self._deviation_history), 1)

    def consciousness_depth(self) -> float:
        """depth = exp(-c_d * d_tau) (F.17.2). Range [0, 1]."""
        return math.exp(-self.c_d * self.d_tau())

    def metacognition_step(self, deviation: float) -> list[float]:
        """Recursive self-evaluation: d_{n+1} <= rho * d_n (F.17.3)."""
        steps = [deviation]
        d = deviation
        for _ in range(self.max_depth - 1):
            d = self.rho * d
            steps.append(d)
        return steps


# ---------------------------------------------------------------------------
# F.20  Working Memory + Cerebellum
# ---------------------------------------------------------------------------
class WorkingMemory:
    """Finite FIFO buffer |h_t| <= T_h (F.20)."""

    def __init__(self, capacity: int = WM_CAPACITY) -> None:
        self.capacity = capacity
        self._buffer: deque[tuple[Any, Any]] = deque(maxlen=capacity)

    def append(self, action: Any, observation: Any) -> None:
        self._buffer.append((action, observation))

    def contents(self) -> list[tuple[Any, Any]]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


class CerebellumPredictor:
    """Forward model: o_hat_{t+1} = o_hat_t + alpha*(o_t - o_hat_t) (F.20)."""

    def __init__(self, dim: int, alpha: float = CEREBELLUM_ALPHA, eta: float = CEREBELLUM_ETA) -> None:
        self.alpha = alpha
        self.eta = eta
        self.prediction = torch.zeros(dim)

    def predict(self) -> torch.Tensor:
        return self.prediction.clone()

    def update(self, observation: torch.Tensor) -> torch.Tensor:
        error = observation.float() - self.prediction
        self.prediction = self.prediction + self.alpha * error
        correction = -self.eta * error
        return correction
