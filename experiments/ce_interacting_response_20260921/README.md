# CE-IR1: interacting common response and joint coefficient inversion

[Chapter 34](../../paper/후속연구_기록과_상태선택/34_상호작용에서_공통반응식의_보정과_계수역산.md)
derives the contribution linear in the quartic coupling
lambda (sum Phi-dagger Phi)^2 / 2, with the original three scalar channels.

The same renormalized two-loop functional gives the vacuum potential, the
coefficient linear in curvature, and the external Abelian F-squared response.
The first mass-derivative curvature relation survives at this order.
The second mass-derivative gauge relation has a calculable extra term.
It must be included before interpreting a residual as the finite coefficient
introduced in chapter 33.

The script verifies Wick factors and subdivergence algebra symbolically, compares
the gauge term with both a mass insertion and an independent finite magnetic
proper-time integral, and encloses two nonzero determinants with outward intervals.
It also reconstructs beta and lambda from two synthetic independent responses and
reports their sensitivity to input or truncation error.
An additional inverse uses curvature and gauge responses at a single shape;
it explicitly requires a nonconformal curvature coupling and no unaccounted
independent finite curvature operator.

    .venv\Scripts\python.exe experiments/ce_interacting_response_20260921/derive_response.py

Scope: MSbar, fixed renormalized mass sources, a common Abelian charge, and the
coefficient linear in lambda. Internal photon contributions, higher powers of
lambda, general non-Abelian representations, dynamic curvature, observed data,
and the selection of physical input values are not supplied by this calculation.
