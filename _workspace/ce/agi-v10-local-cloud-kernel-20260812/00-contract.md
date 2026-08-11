# V10 compositional local-cloud kernel contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/agi-v9-loop-engineering-20260812

Mode: full

## Correction

The stopped V9 benchmark gave every nested shell the same weak recurrence and compared it with
an explicit slow-mode bank. That architecture is not reused. Its result remains a negative
artifact and its seed blocks are retired.

## Claim

V10 tests whether a typed bidirectional transition kernel joining explicit local temporal
states and a shared cloud state adds held-out contextual-memory information beyond matched
local-only and cloud-only kernels.

The monadic object is the transition kernel and its composition, not a neuron, SCC, or physical
brain part.

## State and update

For action width $A$ and registered retentions $\alpha_1,\ldots,\alpha_L$,

$$
h'_{\ell}=\tanh(\alpha_\ell h_\ell+g_o o+g_{CL}c),
$$

$$
c'=\tanh(\gamma c+g_{LC}L^{-1}\sum_\ell h_\ell).
$$

The update is synchronous and dimensionless. The block gain matrix is

$$
M=\begin{pmatrix}\alpha_{\max}&g_{CL}\\g_{LC}&\gamma\end{pmatrix}.
$$

Implementation requires $\rho(M)<1$ and a certified weighted max norm.

## Fair models

- `full`: local bank plus bidirectional shared cloud.
- `local_only`: five explicit local timescale states, no shared transition.
- `cloud_only`: five shared/current-input timescale states, no unit-local identity.
- `cross_cut`: full state and full-trained readout, but both cross gains zero during test.
- `local_reset`: full-trained readout, local state reset at decision.
- `cloud_reset`: full-trained readout, cloud state reset at decision.

Every primary model exposes exactly `20` features and uses the same fixed ridge penalty and
train-only fitting procedure. State scalars, fitted coefficients, effective degrees of freedom,
and executed MAC estimates are reported separately.

## Fresh evidence rule

The stopped V9 development and confirmation seed ranges are forbidden. V10 will receive new
seed roles only after kernel, composition, leakage, control, and capacity tests pass. No
development is authorized by this contract alone.

## Success boundary

Full must beat both local-only and cloud-only on untouched episodes, and the same trained full
readout must lose under cross-cut, local-reset, and cloud-reset. Passing a synthetic task is
still not AGI or biological CloudCell evidence.
