# Mathematical verification

Status: COMPLETE

## Runtime direction and intervention

BrainRuntime uses `W @ pre`, so rows are receivers and columns are senders. A seed-fixed coordinate
permutation defines disjoint physical blocks before outcomes. The treatment block `W[T,S]`
therefore transmits source activation into target activation. The scrambled `W[N,S]` block
preserves count, sign, density, and Frobenius mass while changing only the receiver block. A small
seed-specific background matrix is shared by every arm within a circuit and supplies genuine
between-circuit variation.

## Empirical finite-horizon reachability

Let `U` be the fixed 48-by-3 physical injection, `F=U^T` the 3-by-48 output chart, and `R_H(Uu)` the
endpoint after a one-tick input followed by `H` free native steps from a reset state. Central
differences give

$$
B_H[:,j]=\frac{F R_H(aUe_j)-F R_H(-aUe_j)}{2a},\qquad a=0.5.
$$

Thus `B_H` is a calibration-only empirical endpoint Jacobian for the declared physical control
interface, not a fitted cortical tensor. `C_H=B_HB_H^T+lambda R0` is SPD for positive `lambda` and
SPD reference tensor `R0`; `g_H=C_H^-1` is a precision summary by definition.

The reset runtime initially has no active recurrent senders. With a 16-coordinate unit block axis,
each nonzero entry is `1/4`; a calibration pulse therefore contributes `0.35*(0.5/4)=0.04375` to
salience even before activation magnitude is added. Freezing `active_threshold=0.04` guarantees the
driven coordinates enter the active set on the pulse tick. The default `0.22` would make the first
recurrent pre-vector exactly zero on every subsequent tick and render the intervention untestable.
This is an input-interface viability condition, not evidence for an effect.

In the original chart `R0=I`. The source-conditioned target variance is

$$
V_{T\leftarrow S}=e_T^{\mathsf T}
(b_Sb_S^{\mathsf T}+\lambda R_0)e_T
=B_H[T,S]^2+\lambda.
$$

This equality is a useful implementation invariant. It also makes clear that G1 tests directed
source-to-target reachability under a declared input axis, not curvature.

## Coordinate transform

For fixed invertible `P`, only the output chart is re-expressed: `F'=PF`; physical injection `U`
and runtime pulses remain fixed. The regularizer reference tensor transforms as
`R0'=P R0 P^T`. Hence

$$
B'_H=PB_H,
\quad C'_H=B'_HB_H'^{\mathsf T}+\lambda R'_0=PC_HP^{\mathsf T},
\quad g'_H=P^{-\mathsf T}g_HP^{-1}.
$$

The named physical target covector becomes `ell_T'=P^-T ell_T` when a directional quantity is
re-expressed. First passage stays in the original named-factor chart and is deliberately not a
coordinate-invariance claim. Numerical residuals verify implementation covariance only. They do
not show a unique physical geometry.

## Held-out behavior

Calibration amplitudes are `+/-0.5`; held-out amplitudes are `+/-0.65`. The central-difference
linear prediction is `B_H u`. Endpoint error is evaluated only on held-out pulses. First passage is
computed from the native target-chart trajectory at zero-input ticks `1..6` with crossing `>=0.05`
and no-crossing code `7`; the pulse state `t=0` is excluded. It is not inferred from the metric.

## Identification boundary

Precommitted receiver placement distinguishes targeted from norm/sign/density-matched scrambled
delta within seed-varying, arm-common background circuits. Gain and noise controls show whether a
similar summary/trajectory shift occurs without the declared weight delta. Sham bounds numerical
drift. Passing G1 identifies `do(W) -> Delta summary` and `do(W) -> Delta behavior` jointly inside
this runtime. It does not prove `summary -> behavior`, metric mediation, or unique prediction. G2
tests predictive utility; G3 can test only randomized-contingency association unless an independent
mediator intervention is later added.
