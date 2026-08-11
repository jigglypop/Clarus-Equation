# Loop 3 — observation/action-history latent state preregistration

Status: COMPLETE

Implementation audit note: the first produced artifact
`loop3-history-state-validation.json` is INVALID because its comparator was a
flattened two-unit tanh MLP, not the registered recurrent state. It is retained
for provenance and is not scored. The comparator was corrected to a true
two-state `tanh` RNN without changing seeds, gates, candidate, or any
hyperparameter; the scored rerun uses a separate artifact name.

Scored result: `85/100 GO` within the claim limit. The candidate passed all
causal ablations and calibration gates. It was noninferior, not superior, to
the corrected recurrent comparator (ID LCB `0.0`, OOD LCB `-0.00498`).

## Claim limit

Synthetic controlled-evidence state discovery only. This is not a general
world-model or AGI claim.

## Candidate

The learner receives only the ordered stream
`(previous_action, observation, elapsed_step, observed_mask)`. It is not given
the episode-level sufficient statistic or hidden goal. Its scalar controlled
state is

\[
h_t=\rho^{\Delta t}h_{t-1}+m_t a_{t-1}y_t,
\qquad p(g=+1\mid H_t)=\sigma(\beta h_t+b).
\]

`rho` is selected only by training negative log likelihood from the fixed grid
`{0.50, 0.70, 0.85, 0.95, 1.00}`. `beta,b` are then fitted on training rows and
frozen. No validation-selected threshold is allowed.

## Frozen comparisons

- reactive final-observation policy;
- observation-only state (action factor removed);
- action-shuffled history;
- truncated history (last observed event only);
- train-only fitted recurrent tanh state with a comparable scalar readout;
- oracle statistic as a ceiling, not as a pass comparator.

## Hard gates

On both ID and preregistered OOD noise:

1. candidate return improvement 95% paired bootstrap LCB over reactive,
   observation-only, action-shuffle, and truncation is positive;
2. candidate is noninferior to the recurrent baseline, LCB greater than
   `-0.03`;
3. success rate is above `0.70`, Brier below `0.20`;
4. ID ECE is below `0.08`, OOD ECE below `0.10`;
5. action-sign sensitivity flips the posterior across `0.5`;
6. future reads and environment clone calls are exactly zero.

The run scores zero if any hard gate fails. There is one implementation-debug
run on unit-scale seeds and one locked validation run. No post-validation
hyperparameter or threshold sweep is permitted.
