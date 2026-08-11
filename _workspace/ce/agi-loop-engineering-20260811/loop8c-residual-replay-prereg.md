# Loop 8C preregistration — feedback residual replay

Status: LOCKED BEFORE IMPLEMENTATION

## 1. Frozen checkpoint

All Loop 8B PFC-attractor and MD-modulation parameters remain unchanged. This
loop adds one scalar dimensionless residual state only. It does not add STN,
change the task reward, inspect latent context, or tune Loop 8B.

## 2. Narrow hypothesis

After an action, binary feedback and the trial's two opposite features identify
which context was behaviorally consistent with the observed outcome. Let that
causally reconstructed context sign be `c_fb in {-1,+1}`, and let the pre-action
MD estimate be `c_hat = 2 theta - 1`. The residual is

`phi_(t+1) = lambda_phi phi_t + eta_phi (c_fb - c_hat)`.

Before the next trial, the common context filter receives `replay_gain * phi_t`.
No future feedback, latent rule, or future observation enters `phi`.

Parameters are locked:

- `lambda_phi = 0.70`;
- `eta_phi = 0.50`;
- `replay_gain = 0.60`;
- observed feedback flip probability: ID `0.05`, OOD `0.10`;
- the first `4` trials after a context switch contain zero-mean context noise
  but no signed context signal;
- stationary-control context noise is `0.15` with no switches or cue depletion.

All variables and sigmoid arguments are dimensionless.

## 3. Arms

1. `md_checkpoint`: frozen Loop 8B MD attractor, no residual input.
2. `residual_replay`: the registered residual equation.
3. `residual_sign_flip`: same equation with the prediction-error sign inverted.
4. `oracle_context_md`: ceiling only.

Every arm receives its own action-contingent feedback. The candidate never uses
feedback before the action that generated it. Common random numbers are used
for stimuli, cue noise, feedback flips, distractors, and SDE increments.

## 4. Domains

- ID switch domain: Loop 8B delay/noise plus four cue-depleted post-switch trials.
- OOD switch domain: Loop 8B OOD delay/noise plus the same depletion protocol.
- Stationary control: one context for the whole episode, context noise `0.15`,
  no depletion.

ID/OOD/stationary each use 32 seeds x 192 trials.

## 5. Locked gates

All gates are conjunctive.

1. Residual minus checkpoint overall accuracy LCB >= `+0.03` ID and `+0.02` OOD.
2. Residual minus checkpoint post-switch trials 2–5 accuracy LCB >= `+0.08` ID
   and OOD.
3. Residual minus sign-flip overall accuracy LCB >= `+0.10` ID and OOD.
4. Stationary absolute mean accuracy change from checkpoint <= `0.01`.
5. The mean post-switch improvement exceeds the stationary mean change by at
   least `0.07` in ID and OOD.
6. Oracle is not worse than residual in every switch domain.
7. Every neural state is finite with `max(abs(z)) <= 4`; every residual is
   finite with `abs(phi) <= 4`.
8. `future_reads = 0` and `environment_clone_calls = 0`.

Score is `100 GO` only when all gates pass; otherwise `0 STOP`. Passing permits
a separately preregistered BG/STN decision-boundary experiment. It does not
authorize canonical runtime integration or a biological hippocampus claim.

## 6. Prohibited after-result actions

- no replay coefficient, cue-depletion, feedback-noise, or threshold sweep;
- no seed deletion or selective domain reporting;
- no modification of the frozen Loop 8B checkpoint;
- no reinterpretation of generic feedback correction as episodic replay if the
  switch-selectivity gate fails.
