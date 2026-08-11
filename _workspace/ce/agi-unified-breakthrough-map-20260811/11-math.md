# Unified model mathematics

Status: COMPLETE

## Model name

Adaptive Causal Belief-State Model (`ACBSM`).

## Frozen structural core

Let `f_G` be the current frozen sparse causal transition and let its observed
innovation be

`e_t = x_t - f_G(x_{t-1})`.

The first development model replaces the single residual score with a
two-timescale latent state:

`z_t = [z_fast,t, z_slow,t]^T`

`z_{t+1} = A z_t + w_t`, where

`A = diag(rho_fast, rho_slow)` and
`|rho_fast| < |rho_slow| < 0.98`.

Residual observations obey

`e_t = L z_t + v_t`.

A prefix-only Kalman/robust linear observer produces posterior mean `m_T` and
covariance `P_T`. The completed H20 target and hidden simulator state are not
inputs.

## Forecast

For each lead, propagate the belief and inject its correction inside the
structural rollout:

`m_{T+h} = A^h m_T`

`P_{T+h} = A^h P_T (A^h)^T + sum_j A^j Q (A^j)^T`

`c_h = signal_h / (signal_h + uncertainty_h + epsilon)`

`xhat_{T+h+1} = f_G(xhat_{T+h}) + c_h L m_{T+h+1}`.

The trust `c_h` is determined by posterior signal-to-noise and horizon, not by
a new free scalar fitted to the final output. Thus the model changes the
predictive state while retaining V8 as the degenerate one-state/constant-trust
case.

## Optional locked-origin regime layer

The same container reserves a later regime posterior

`pi_T = softmax(W phi(prefix) + b)`

and `A(pi_T) = sum_k pi_T,k A_k`. If enabled, `pi_T` is computed once at the
origin and frozen throughout H20. It must not be recomputed from self-generated
future predictions.

## Stability

The augmented Jacobian has block-triangular form

`J_aug = [[J_fG, c_h L A], [0, A]]`.

Stable eigenvalues of `J_fG` and `A` are necessary but not sufficient because
of transient amplification. Development must additionally require a weighted
block norm or common Lyapunov certificate and persist per-lead pathwise radii.

## Identifiability constraints

- Order the poles as fast then slow to prevent label switching.
- Normalize columns of `L` and fix their deterministic signs.
- Keep rank exactly two in the first model.
- Use positive-semidefinite `Q`, `R`, and posterior covariance.
- Shrink covariance/loading estimates across training episodes.
- Use robust residual likelihood so isolated shocks cannot define the slow mode.
