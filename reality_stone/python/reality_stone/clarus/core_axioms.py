"""Executable gates for the conditional mathematical core of CE.

The functions in this module deliberately separate three questions:

1. Does a proposed law satisfy the structural axioms?
2. Does the resulting fixed-point equation have the claimed branch?
3. Is that mathematical branch identified with a physical observable?

Only the first two questions are answered here.  Observable identification is
a separate bridge and must not be inferred from a small algebraic residual.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math


SurvivalLaw = Callable[[float], float]
FeedbackLaw = Callable[[float], float]


def poisson_probability(count: int, expected_triggers: float) -> float:
    """Probability of ``count`` independent rare fold triggers.

    In the microscopic strengthening of the CE core, ``expected_triggers`` is
    the folding optical depth.  The unit coefficient in ``exp(-depth)`` then
    follows from the operational definition of depth rather than from a fitted
    exponential rate.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if expected_triggers < 0.0:
        raise ValueError("expected_triggers must be non-negative")
    if expected_triggers == 0.0:
        return 1.0 if count == 0 else 0.0
    return math.exp(
        -expected_triggers
        + count * math.log(expected_triggers)
        - math.lgamma(count + 1.0)
    )


def zero_trigger_survival(expected_triggers: float) -> float:
    """Survival probability when and only when no fold trigger occurs."""
    return poisson_probability(0, expected_triggers)


def exponential_survival(depth: float, *, rate: float = 1.0) -> float:
    """The positive character of the additive depth semigroup."""
    if depth < 0.0:
        raise ValueError("depth must be non-negative")
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    return math.exp(-rate * depth)


def stretched_exponential_survival(
    depth: float,
    *,
    rate: float = 1.0,
    power: float = 2.0,
) -> float:
    """A useful countermodel that generally fails exact composition."""
    if depth < 0.0:
        raise ValueError("depth must be non-negative")
    if rate <= 0.0 or power <= 0.0:
        raise ValueError("rate and power must be positive")
    return math.exp(-rate * depth**power)


def power_survival(depth: float, *, rate: float = 1.0) -> float:
    """Another positive normalized countermodel without the semigroup law."""
    if depth < 0.0:
        raise ValueError("depth must be non-negative")
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    return 1.0 / (1.0 + rate * depth)


def composition_residual(law: SurvivalLaw, depths: Iterable[float]) -> float:
    """Maximum residual of S(a+b)=S(a)S(b) over a finite audit grid."""
    grid = tuple(float(depth) for depth in depths)
    if not grid:
        raise ValueError("depths must not be empty")
    if any(depth < 0.0 for depth in grid):
        raise ValueError("depths must be non-negative")
    return max(
        abs(law(left + right) - law(left) * law(right))
        for left in grid
        for right in grid
    )


def complement_feedback(survival: float) -> float:
    """Suppressed fraction for an exhaustive normalized binary partition."""
    if not 0.0 <= survival <= 1.0:
        raise ValueError("survival must lie in [0, 1]")
    return 1.0 - survival


def powered_feedback(survival: float, *, power: float = 2.0) -> float:
    """A nonlinear closed feedback countermodel."""
    if not 0.0 <= survival <= 1.0:
        raise ValueError("survival must lie in [0, 1]")
    if power <= 0.0:
        raise ValueError("power must be positive")
    return (1.0 - survival) ** power


def mixture_affinity_residual(
    feedback: FeedbackLaw,
    values: Iterable[float],
    weights: Iterable[float],
) -> float:
    """Audit K(tx+(1-t)y)=tK(x)+(1-t)K(y).

    Affinity is the operational statement that randomly mixing two normalized
    path ensembles before measuring suppression gives the same aggregate as
    measuring each ensemble and mixing the two results.
    """
    xs = tuple(float(value) for value in values)
    ts = tuple(float(weight) for weight in weights)
    if not xs or not ts:
        raise ValueError("values and weights must not be empty")
    if any(not 0.0 <= value <= 1.0 for value in xs):
        raise ValueError("values must lie in [0, 1]")
    if any(not 0.0 <= weight <= 1.0 for weight in ts):
        raise ValueError("weights must lie in [0, 1]")
    return max(
        abs(
            feedback(weight * left + (1.0 - weight) * right)
            - (
                weight * feedback(left)
                + (1.0 - weight) * feedback(right)
            )
        )
        for left in xs
        for right in xs
        for weight in ts
    )


@dataclass(frozen=True)
class ElectroweakCoherence:
    """Normalized tree-level neutral electroweak mixing data.

    ``g`` and ``g_prime`` are the positive SU(2) and U(1) couplings.  Dividing
    the neutral mass matrix by its non-zero eigenvalue produces a rank-one
    projector.  ``intensity`` is the square of its unique off-diagonal entry.
    """

    g: float
    g_prime: float

    def __post_init__(self) -> None:
        if self.g <= 0.0 or self.g_prime <= 0.0:
            raise ValueError("electroweak couplings must be positive")

    @property
    def norm(self) -> float:
        return self.g * self.g + self.g_prime * self.g_prime

    @property
    def sin2_theta(self) -> float:
        return self.g_prime * self.g_prime / self.norm

    @property
    def cos2_theta(self) -> float:
        return self.g * self.g / self.norm

    @property
    def amplitude(self) -> float:
        return self.g * self.g_prime / self.norm

    @property
    def intensity(self) -> float:
        return self.amplitude * self.amplitude

    @property
    def normalized_mass_matrix(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (self.cos2_theta, -self.amplitude),
            (-self.amplitude, self.sin2_theta),
        )


def additive_effective_depth(
    spatial_dimension: int,
    channel_intensities: Iterable[float],
) -> float:
    """Trace-like count of unit spatial modes plus independent intensities."""
    if spatial_dimension < 0:
        raise ValueError("spatial_dimension must be non-negative")
    channels = tuple(float(value) for value in channel_intensities)
    if any(not 0.0 <= value <= 1.0 for value in channels):
        raise ValueError("channel intensities must lie in [0, 1]")
    return float(spatial_dimension) + math.fsum(channels)


def electroweak_effective_depth(
    spatial_dimension: int,
    *,
    g: float,
    g_prime: float,
) -> float:
    """Conditional D_eff when the EW cross-channel is counted exactly once."""
    coherence = ElectroweakCoherence(g=g, g_prime=g_prime)
    return additive_effective_depth(spatial_dimension, (coherence.intensity,))


def hodge_target_dimension(source_degree: int, target_degree: int) -> int:
    """Dimension required by *: Lambda^p -> Lambda^q, namely d=p+q."""
    if source_degree < 0 or target_degree < 0:
        raise ValueError("form degrees must be non-negative")
    return source_degree + target_degree


def bivector_vector_closure_dimensions(max_dimension: int = 16) -> tuple[int, ...]:
    """Non-zero d where dim(Lambda^2 V)=dim(V)."""
    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")
    return tuple(
        dimension
        for dimension in range(1, max_dimension + 1)
        if dimension * (dimension - 1) // 2 == dimension
    )


def bootstrap_residual(
    survival: float,
    depth: float,
    *,
    feedback: FeedbackLaw = complement_feedback,
    rate: float = 1.0,
) -> float:
    """Residual of x=exp[-rate*D*K(x)] for a chosen feedback law."""
    if not 0.0 <= survival <= 1.0:
        raise ValueError("survival must lie in [0, 1]")
    if depth < 0.0:
        raise ValueError("depth must be non-negative")
    return survival - math.exp(-rate * depth * feedback(survival))


def thinned_trigger_mean(
    survival: float,
    depth: float,
    *,
    feedback: FeedbackLaw = complement_feedback,
    rate: float = 1.0,
) -> float:
    """Conditional trigger mean ``rate * depth * K(survival)``.

    With affine complement thinning this is the microscopic statement
    ``N | x ~ Poisson(rate * depth * (1-x))``.
    """
    if not 0.0 <= survival <= 1.0:
        raise ValueError("survival must lie in [0, 1]")
    if depth < 0.0:
        raise ValueError("depth must be non-negative")
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    return rate * depth * feedback(survival)


def bootstrap_map(
    survival: float,
    depth: float,
    *,
    feedback: FeedbackLaw = complement_feedback,
    rate: float = 1.0,
) -> float:
    """One zero-trigger update of the generalized bootstrap map."""
    return zero_trigger_survival(
        thinned_trigger_mean(
            survival,
            depth,
            feedback=feedback,
            rate=rate,
        )
    )


def low_bootstrap_fixed_point(
    depth: float,
    *,
    rate: float = 1.0,
    tolerance: float = 1e-14,
    max_iterations: int = 256,
) -> float:
    """Return the stable non-trivial root for ``rate * depth > 1``."""
    if depth < 0.0:
        raise ValueError("depth must be non-negative")
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    effective_depth = rate * depth
    if effective_depth <= 1.0:
        raise ValueError(
            "a non-trivial root in (0, 1) requires rate * depth > 1"
        )
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("invalid solver controls")

    low = 0.0
    high = 1.0 / effective_depth
    f_low = bootstrap_residual(low, depth, rate=rate)
    f_high = bootstrap_residual(high, depth, rate=rate)
    if not f_low < 0.0 < f_high:
        raise RuntimeError("the analytic low-branch bracket was not valid")

    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        f_middle = bootstrap_residual(middle, depth, rate=rate)
        if abs(f_middle) <= tolerance or high - low <= tolerance:
            return middle
        if f_middle > 0.0:
            high = middle
        else:
            low = middle
    raise RuntimeError("low-branch bisection did not converge")


def bootstrap_fixed_points(
    depth: float,
    *,
    rate: float = 1.0,
) -> tuple[float, ...]:
    """Enumerate every fixed point of the complement-feedback map on [0, 1].

    For ``rate * depth <= 1`` the identity point is the only root.  Above the
    transcritical threshold, the stable non-trivial root and the identity root
    are both returned.  This prevents a numerical bracket from silently
    becoming a physical branch-selection rule.
    """
    if depth < 0.0:
        raise ValueError("depth must be non-negative")
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    if rate * depth <= 1.0:
        return (1.0,)
    return (low_bootstrap_fixed_point(depth, rate=rate), 1.0)


def bootstrap_orbit(
    initial_survival: float,
    depth: float,
    *,
    rate: float = 1.0,
    iterations: int = 64,
) -> tuple[float, ...]:
    """Iterate the physical zero-trigger update, including its initial state."""
    if not 0.0 <= initial_survival <= 1.0:
        raise ValueError("initial_survival must lie in [0, 1]")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    orbit = [float(initial_survival)]
    for _ in range(iterations):
        orbit.append(bootstrap_map(orbit[-1], depth, rate=rate))
    return tuple(orbit)


def bootstrap_stability_multiplier(
    survival: float,
    depth: float,
    *,
    rate: float = 1.0,
) -> float:
    """``F'(x)=rate * depth * x`` at a complement-feedback fixed point."""
    if not 0.0 <= survival <= 1.0:
        raise ValueError("survival must lie in [0, 1]")
    if depth < 0.0:
        raise ValueError("depth must be non-negative")
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    return rate * depth * survival


def low_branch_depth_sensitivity(
    survival: float,
    depth: float,
    *,
    rate: float = 1.0,
) -> float:
    """Implicit derivative dx/dD on the non-trivial branch."""
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    denominator = 1.0 - rate * depth * survival
    if abs(denominator) < 1e-15:
        raise ZeroDivisionError("fixed point is at the branch bifurcation")
    return -rate * survival * (1.0 - survival) / denominator


def low_branch_rate_sensitivity(survival: float, depth: float, *, rate: float = 1.0) -> float:
    """Implicit derivative dx/d(rate) on the non-trivial branch."""
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    denominator = 1.0 - rate * depth * survival
    if abs(denominator) < 1e-15:
        raise ZeroDivisionError("fixed point is at the branch bifurcation")
    return -depth * survival * (1.0 - survival) / denominator


def effective_depth_from_fixed_point(survival: float) -> float:
    """Invert the non-trivial fixed-point law: D=-log(x)/(1-x)."""
    if not 0.0 < survival < 1.0:
        raise ValueError("survival must lie strictly between zero and one")
    return -math.log(survival) / (1.0 - survival)


def weak_mixing_from_fixed_point(
    survival: float,
    *,
    spatial_dimension: int = 3,
) -> float:
    """Infer the smaller p=sin^2(theta) root under D=d+p(1-p)."""
    correction = effective_depth_from_fixed_point(survival) - spatial_dimension
    if not 0.0 <= correction <= 0.25:
        raise ValueError("fixed point is incompatible with p(1-p) in [0, 1/4]")
    return 0.5 * (1.0 - math.sqrt(1.0 - 4.0 * correction))
