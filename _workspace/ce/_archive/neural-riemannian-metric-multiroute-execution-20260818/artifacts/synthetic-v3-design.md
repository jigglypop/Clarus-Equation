# Synthetic v3 frozen design

Status: FROZEN_NOT_EXECUTED

This estimator-only tournament has six generators, with `D=3`, eight nodes,
20 paired, independently fit training/test circuits per generator, 24 trajectories,
320 steps, and `dt=0.02`. Every circuit draws a directed edge probability
uniformly on `[0.08,0.48]`; its realized density is `phi`. The common field is
`h(phi,x)=(phi-0.28)*(1+0.2*tanh(x0))`; `K=diag(.55,.8,1.05)` and
`sigma=.2`. Training uses e1/e2; test uses e3 during the second half only.
The fitter is blind to generator and truth. Each test circuit is scored from
its own paired one-circuit training fit (`fit_id`, train/test seeds are stored).
This one-circuit calibration deliberately risks identifiability failure when
the realized density is near `phi0`; no pooling or post-outcome repair is
allowed. All candidate coefficients use the
positive-beta convention `exp(-beta*h)`. Diagonal K is constrained to the
declared stable family with the predeclared floor `K_j >= 1e-6` before sigma is
recomputed.

G1 uses `M=diag(1,1,exp(-1.2h))`, drift `-MKx`, and `Q=sigma^2 M`.
G2 keeps that drift but uses `Q=sigma^2 I`; G3 maps Euclidean latent OU data
through `(z0,z1+.22*z0*z2,z2)`; G4 uses conformal `M=exp(-h)I`; G5 adds a
frozen W shortcut with Euclidean dynamics; and G6 is Euclidean null.

Candidates are metric, parameter-richer direct-v/Q, conformal gain/noise,
noise-only, Euclidean, and determinant-one nonlinear flat pullback. They fit
only training increments with known force removed and score held-out increment
densities. Grids and parameter counts are fixed in the script. G1 requires at
least 18/20 beta recoveries and five Holm-corrected circuit-level sign tests.
Each recovery belongs to its own independently fit paired train/test circuit;
the 20 recoveries are not copies of a pooled fit.
G2--G6 record circuit-level, trajectory-nested sign tests and use exactly 100
non-G1 circuits for the any-reject false-positive numerator (threshold <=5).

The curvature check computes scalar curvature from finite-difference
Christoffel/Ricci tensors. Euclidean and nonlinear pullback fields must be
flat under three deterministic affine chart changes per independently fitted
G3 circuit; a known
curved conformal fixture must be nonflat. This is an estimator check, never
biological evidence.

The physical intervention is constant in observed y coordinates. G3 applies
the inverse pushforward in latent z during simulation; the flat-pullback
candidate applies its candidate inverse Jacobian to the same observed-y force.
The create-only compressed trace records paths, W, phi, directions, and seeds;
the result JSON stores its SHA-256. First passage is frozen as `||x||>=0.8`.

`x` and `h` are dimensionless. `K` and `sigma^2` have units `T^-1`; therefore
`Q*dt` is a dimensionless covariance and every log-density determinant is
taken only of that dimensionless quantity.
