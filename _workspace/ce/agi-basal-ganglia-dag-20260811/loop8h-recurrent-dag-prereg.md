# Loop 8H preregistration — recurrent basal-ganglia DAGlet

Status: LOCKED BEFORE IMPLEMENTATION

## Mechanism

One decision tick evaluates a finite conditional DAG in topological order,
performs one reverse-topological competitive-inhibition reduction, chooses an
action, then commits signed feedback to the next-tick recurrent state. Same-tick
feedback edges and recursive forward calls are forbidden.

The task has eight actions, three noisy content bits, four persistent hidden
contexts, a noisy context cue, block switches, and the fixed compositional rule
`action = base_action XOR context_mask`. The DAG shares action leaves across
context experts. OOD uses held-out base combinations, higher observation noise,
weaker cues, and faster switches. A stationary null makes the cue sufficient; a
flat null removes the context transformation.

## Frozen arms

1. hard tree;
2. gradient-boosted multiclass stumps (XGBoost-like static baseline, not the
   XGBoost package or a biological claim);
3. feedforward conditional DAG;
4. recurrent inhibitory DAG;
5. recurrent DAG with feedback eligibility shuffled across context channels;
6. recurrent DAG with feedback sign flipped;
7. matched flat policy for the flat null.

All online arms receive the same immutable trial sequence. The recurrent arm
may update state only after its action feedback is observed. No future read or
environment clone is allowed.

## Frozen data roles and coefficients

- boosted-stump structure training seeds: `870000..870015`;
- ID validation seeds: `872000..872031`;
- OOD validation seeds: `872100..872131`;
- 256 trials per validation seed;
- 32 boosting rounds, shrinkage `0.15`, eight threshold quantiles;
- recurrent decay `0.82`, cue gain `1.0`, feedback gain `1.35`;
- policy temperature `0.35`, inhibition strength `0.20`;
- recurrent state norm cap `6.0`.

These values may be changed only in a later named development loop, never after
reading this validation.

## Metrics and gates

Each gate contributes 10 diagnostic promise points. `GO` requires all ten.

1. topology valid, finite-pass evaluation bound respected;
2. all probabilities/states finite and state norm bounded;
3. recurrent minus feedforward accuracy 95% paired LCB >= `0.03` ID and `0.02` OOD;
4. boosted-stump minus recurrent NLL 95% paired LCB > `0` ID and OOD;
5. recurrent minus feedback-shuffle accuracy LCB >= `0.05` ID/OOD;
6. recurrent minus sign-flip accuracy LCB >= `0.10` ID/OOD;
7. recurrent minus feedforward post-switch trials 2–5 accuracy LCB >= `0.08` ID/OOD;
8. stationary-null absolute recurrent/feedforward accuracy difference <= `0.02`;
9. flat-null recurrent minus matched-flat accuracy <= `0.01`;
10. future reads, environment clones, same-tick commits, nonfinite events, and
    topology cycles all equal zero.

The score is descriptive even when below 100. A failed conjunction remains
`STOP`; it cannot be promoted by a high partial score.
