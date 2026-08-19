# BrainRuntime weight--metric--dynamics intervention contract

Status: COMPLETE

PREDECESSORS:

- `_workspace/ce/brain-mechanism-alternative-routes-20260819`
- `_workspace/ce/brain-memory-contrastive-predictive-routes-20260819`
- `_workspace/ce/_archive/neural-riemannian-metric-multiroute-execution-20260818`

## Question and boundary

This simulator-only run separates three questions around the previously missing chain

$$
\operatorname{do}(W)\longrightarrow \Delta g,
\qquad
\operatorname{do}(W)\longrightarrow \Delta x,
\qquad
g\text{-based prediction of }x.
$$

without borrowing evidence from the released PFC pseudopopulation. G1 can identify only a joint
effect: direct weight intervention changes a declared SPD response summary and held-out behavior.
It does not manipulate the derived metric independently and therefore cannot identify
`Delta g -> Delta x` or metric mediation. G2 separately tests operational predictive utility; G3
tests randomized-contingency association consistent with the frozen metric procedure. A passing
result is limited to the declared BrainRuntime intervention and estimator. It cannot identify a
physical cortical metric, biological synaptic plasticity, or consciousness.

Development seeds are `97401..97416`. Confirmation seeds `99401..99432` remain sealed until source,
tests, controls, endpoints, and a freeze manifest are fixed. A seed-level circuit is the independent
unit.

## G1: precommitted directed edge intervention

Use a 48-dimensional Torch CPU runtime with automatic STDP off,
hippocampal writes off, `axon_delay=False`, `dale_law=False`, and active ratio `1.0`. Partition
coordinates before outcomes using a seed-fixed permutation into three disjoint 16-coordinate blocks
`S` (source), `T` (target), and `N` (neutral). Their unit constant vectors are the columns of a
frozen runtime injection `U in R^(48x3)` and the rows of the output chart `F=U^T`.

The runtime active-selection threshold is frozen at `0.04`. This is a pre-development mechanism
audit repair: a unit block vector has entries `1/4`, so a calibration pulse has driven-coordinate
external magnitude `0.5/4=0.125`; the salience contribution from the external term alone is
`0.35*0.125=0.04375`. The original runtime default threshold `0.22` therefore made every reset
pulse ineligible and disconnected all recurrent interventions. The repaired threshold is below the
known external-salience lower bound and is not selected from a task outcome. Every pulse must audit
that all driven coordinates became active after the pulse tick.

Each seed has a genuinely different, arm-common nuisance circuit
`W0 ~ Normal(0,0.01^2)` with zero diagonal. The Gaussian generator and coordinate permutation are
fixed from the seed before any arm outcome. Every arm within a seed starts from the same serialized
`W0` snapshot. The intervention delta, not the total post-intervention matrix, is structurally
matched across receiver-placement arms.

The treatment installs `+0.08` on every edge in `W[T,S]`. Its edge count is 256 and Frobenius norm
is `1.28`. Controls are:

- sham, with zero delta;
- scrambled block `W[N,S] += 0.08`, exactly matched in edge count, sign, density, and norm;
- gain-only, changing external gain from `0.45` to `0.60` while retaining `W0`;
- noise-only, changing runtime noise from `0` to `0.02` while retaining `W0`.

Every arm starts from an identical reset state and uses the same pulse order. Calibration uses only
paired `+0.5/-0.5` one-tick pulses along chart axes `S,T,N`, followed by six zero-input WAKE steps.
Every individual pulse begins from a freshly restored arm snapshot with reset step index and empty
stores; pulse history never carries across signs or axes.
For axis `j`, the empirical endpoint sensitivity is

$$
b_j=\frac{y_H(+0.5Ue_j)-y_H(-0.5Ue_j)}{1.0},
\qquad B_H=[b_S,b_T,b_N].
$$

The frozen full reachability summary and precision are

$$
C_H=B_HB_H^{\mathsf T}+10^{-3}R_0,
\qquad g_H=C_H^{-1}.
$$

The source-conditioned summary is

$$
C_H^{(S)}=b_Sb_S^{\mathsf T}+10^{-3}R_0,
\qquad R_0=I_3
\text{ in the original chart}.
$$

The single preregistered calibration sensitivity endpoint is `abs(B_H[T,S])`. Target variance
`e_T^T C_H^(S) e_T = B_H[T,S]^2 + 1e-3` is its explicitly redundant SPD packaging and is reported
descriptively, not counted as independent confirmation.

Held-out evaluation uses source-pulse amplitudes `+0.65` and `-0.65`, absent from calibration.
Primary behavior is mean absolute target-chart endpoint after six zero-input steps. The driven pulse
state is timestamp `t=0` and is excluded from first passage. First passage is the first zero-input
WAKE step `t in {1,...,6}` whose absolute target coordinate is `>=0.05`; no crossing is `7`.
The linear calibration prediction `B_H * [amplitude,0,0]` and actual endpoint are both logged, but a
small prediction error alone cannot establish the intervention chain.

The output-chart audit uses the precommitted non-orthogonal matrix

$$
P=\begin{pmatrix}1&0.2&0\\0&1&0.1\\0.1&0&1.1\end{pmatrix}.
$$

Its determinant, 2-norm condition number, and tensor bytes are logged; singular or nonfinite values
are automatic failures. Transformed `C'` is rebuilt from `B'=PB` and `R0'=P R0 P^T`, not merely
copied from `PCP^T`.

G1 GO requires, on at least 80% of circuits:

1. exact intended applied edge block and exact scrambled matching;
2. treatment cross-response above every control, with target variance reported as its monotone SPD
   representation;
3. held-out target endpoint advantage at least `0.05` over the strongest control;
4. treatment first passage at least one tick earlier than every control;
5. endpoint linearization error at most `0.10` in chart norm;
6. finite SPD eigenvalues, dense/CSR parity, snapshot/reset parity, and no memory rows;
7. covariance/metric transform-law residual at most `1e-6` under the fixed invertible output-chart
   change, using `F'=PF`, fixed physical injection `U`, and `R0'=P R0 P^T`.

The 256-edge support and per-entry value are checked at `1e-7`; norms are checked against `1.28`
at `1e-6` to accommodate float32 reduction. A non-frozen configuration is diagnostic-only and
cannot receive G1 `GO`.

For circuit `i`, average `+/-0.65` absolute target endpoints within arm, then define

`d_i = treatment_i - max(sham_i, scrambled_i, gain_i, noise_i)`.

The paired percentile-bootstrap 95% lower bound of the mean `d_i` must exceed zero, using 10,000
seed-level resamples and generator seed `97499`. Ties are literal numeric ties; nonfinite rows are
automatic failures. First-passage comparison uses the within-circuit earliest control crossing:
the treatment tick must be at most `min(control ticks)-1`. Pulses and ticks are never resampled as
independent units.

Cross-response is one calibration scalar per circuit. The linearization-error gate must pass
separately for both held-out signs. First passage must also pass sign by sign:
`tau_treatment(i,s) <= min_control tau_control(i,s)-1` for each `s in {+,-}`. Only the primary
endpoint advantage averages the two signs within circuit.

The noise arm uses BrainRuntime's native deterministic step-index generator. Every pulse restores
step index zero, so its noise sequence is exactly replayed across signs, axes, and reruns. There is
one pulse realization per sign/axis; no repeated noise trial is counted as a unit. Seed variation
comes from the arm-common `W0` and coordinate permutation, not from treating noise ticks as samples.

## G2: fixed-weight metric sufficiency

The original “metric sufficiency” wording is mathematically unavailable because `g` is a
deterministic reparameterization of calibration `C` and loses signed orientation carried by `B`.
The exact replacement is the separately audited `01-g2-contract.md`: fixed-W incremental utility of
one calibration-only quadratic metric feature for a scalar nonmetric path-access target. Raw `B/C`,
Euclidean, permuted metric, alternative SPD, persistence, and stronger nonlinear direct dynamics are
mandatory adverse baselines. G1 results do not tune G2 seeds, splits, targets, or gates.

## G3: learned mediation

G3 may use only the independently confirmed M1 fixed-clock learner, not supervised M0/M3 capacity.
It compares matched learned, target-shuffled, and no-replay contingencies from identical snapshots,
then applies the frozen G1 calibration estimator before zero-store recall. The treatment, applied
weight contrast, metric cross-response, and continuous recall endpoint are logged per circuit.
“Frozen estimator” means the procedure `(U,F,H,lambda,R0,pulse/reset schedule,P)`. Numeric `B,C,g`
are recomputed from calibration-only probes after each learned/control post-update weight; G1's
numeric matrices are never reused and recall outcomes cannot select a metric.

G3 requires a sign-correct cross-fitted seed-level association coefficient between the randomized
learning contingency, frozen-procedure metric change, and continuous recall, with a bootstrap 95%
interval excluding zero, plus successful recall and failure of matched learning controls. This is
not causal mediation because the derived metric is not independently manipulated. G3 remains
pending until G1 estimator and null behavior are fixed.

## Common exclusions

- No post-outcome chart, metric, horizon, threshold, or edge block selection.
- No hippocampal or temporal row during calibration/evaluation.
- No target codeword, decoder output, condition flag, or future state in an estimator input.
- No neuron/tick pseudoreplication.
- No inference from nonzero AIRM distance alone.
- No identification of curvature; this is a finite-dimensional SPD response summary.
