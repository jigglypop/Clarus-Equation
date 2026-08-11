# Loop 8K preregistration — factorized outcome posterior

Status: LOCKED BEFORE IMPLEMENTATION

Loop 8K removes heuristic signed context updates and reset. Context state is a
probability simplex carried by a finite-state filter. The fixed transition model
uses the training/ID switch hazard `h=0.06`:

`q^-_{t,c} = (1-h)q_{t-1,c} + h/(C-1) sum_{j!=c} q_{t-1,j}`.

The current cue is consumed exactly once:

`q^cue_{t,c} proportional to q^-_{t,c} exp(cue_c)`.

The soft DAG already supplies

`ell_{t,c} = P(chosen action succeeds | context c, content evidence)`.

For observed binary outcome `y_t`,

`L_{t,c}=ell_{t,c}^y_t (1-ell_{t,c})^(1-y_t)`,

`q_{t,c} proportional to q^cue_{t,c} L_{t,c}`.

The stored logit state for the next forward call is the centered log of the
transition-predicted next prior. Legacy state decay, feedback gain, reset,
outcome target, future context, and switch labels are not used by this arm.
OOD retains the model hazard `0.06` while the environment switches at `0.12`.

All quantities entering exp/log are dimensionless probabilities or normalized
log-likelihoods. Machine `nextafter` bounds are numeric-domain protection, not
fitted probability floors.

## Arms

1. hard recurrent reference;
2. soft signed-heuristic parent;
3. Loop 8J directional-reset reference;
4. exact factorized outcome posterior candidate;
5. candidate with context support deranged;
6. candidate with outcome sign flipped;
7. soft feedforward reference.

New seeds: ID `879000..879031`, OOD `879100..879131`, matched stationary and
flat nulls from `879200`. No Loop 8H–8J validation seed is reused.

## Ten gates

1. posterior and predicted prior are finite, strictly positive, and normalized
   within `1e-12`;
2. exact likelihood and transition recomputation error <= `1e-12`;
3. candidate minus signed heuristic accuracy LCB >= `+0.02` ID/OOD;
4. candidate minus signed heuristic post-switch +1..+4 accuracy LCB >= `+0.05`
   ID/OOD;
5. candidate minus hard recurrent accuracy LCB >= `-0.01` ID/OOD;
6. hard recurrent minus candidate NLL LCB > `0` ID/OOD;
7. candidate minus support-deranged accuracy LCB >= `+0.05` ID/OOD;
8. candidate minus outcome-sign-flip accuracy LCB >= `+0.10` ID/OOD;
9. candidate minus directional-reset overall accuracy LCB > `0` ID/OOD;
10. stationary candidate/signed absolute accuracy difference <= `0.02`, flat
    candidate minus matched-flat accuracy <= `0.01`, and all causal/topology/
    finite/state-integrity counters pass.

Every gate contributes ten diagnostic points; GO is conjunctive.
