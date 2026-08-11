# Dynamic delayed orbit quotient: preregistration

## Claim under test

On the cyclic cover `C_N x {0, ..., Q-1}`, let every edge be tied by target
orbit, source orbit, spatial shift, and an integer causal delay:

\[
x_{k,n,q}=\phi\!\left(
 b_q+u_{k-1,n,q}+
 \sum_e K_e x_{k-d_e,n-\delta_e,r_e}
\right),\qquad d_e\ge 1.
\]

Indices in space are modulo `N`; states before time zero are zero.  For a
spatially constant input and state, define the lift and projection

\[
(L_Nz)_{n,q}=z_q,\qquad
(P_Nx)_q=N^{-1}\sum_nx_{n,q}.
\]

The registered theorem candidate is

\[
P_NL_N=I,\qquad F_NL_N=L_NF_Q,
\]

where the quotient keeps delay classes and sums tied spatial shifts.  This is
an exact invariant-sector statement, not a claim that arbitrary trajectories
are recoverable from their spatial mean.

## Dimension audit

`x`, `z`, `b`, `u`, and the argument and output of `tanh` are normalized neural
activities and therefore dimensionless.  `K` is a dimensionless gain.  The
integer delay is `d = Delta t / tau_step`, so it is dimensionless.  Cell index,
orbit index, and spatial shift are discrete labels.  The sufficient small-gain
quantity

\[
g=\max_q\sum_{e:\,q_e=q}|K_e|<1
\]

is dimensionless.  Passing this audit establishes dimensional consistency,
not biological validity or global nonlinear stability outside the stated
bound.

## Locked validation

- covers: `N = 32, 64, 128, 256`; orbit count `Q = 3`; horizon `T = 8`;
- `max |P L - I| <= 1e-12`;
- uniform full trajectory versus lifted quotient `<= 1e-10` at every size;
- quotient trajectory invariant with respect to `N` to `<= 1e-12`;
- translation-equivariance residual `<= 1e-10`;
- an impulse has exactly zero effect before the shortest integer path delay;
- sparse causal-cone reconstruction versus the full perturbed cover `<= 1e-10`;
- pre-wrap active-cell bound `Q(2RT+1)` for maximum absolute shift `R`;
- quotient update work is independent of `N`;
- snapshot continuation is identical to uninterrupted execution to `<= 1e-12`;
- zero-delay edges are rejected, and active-budget overflow raises an error.

## Destructive controls

Each control must break its targeted identity:

1. one untied node bias breaks quotient closure;
2. a matched-norm spatially varying kernel breaks quotient closure;
3. an orbit-channel permutation at one cell breaks the registered lift;
4. an open boundary breaks cyclic translation equivariance;
5. index-based Top-K selection breaks translation equivariance;
6. a zero-delay edge is rejected as a same-tick causal read.

## Interpretation gate

`GO` requires every exact gate and every destructive control.  It promotes the
finite-cover delayed quotient and sparse-cone executor to a validated
implementation theorem.  It does not establish a literal infinite runtime,
real-brain translation symmetry, learned orbit discovery, or AGI performance.
Failure of any exact gate is `STOP`; thresholds will not be tuned after seeing
the result.
