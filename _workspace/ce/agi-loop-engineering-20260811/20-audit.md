# Status audit

Status: COMPLETE

| Claim | Status | Evidence | Gate |
|---|---|---|---|
| Optional action-conditioned controller is implemented without changing the default agent path | implementation | focused regression | PASS |
| Belief planning needs previous action polarity | synthetic learned validation | full vs action-agnostic LCB `0.5734` | PASS |
| Delayed horizon is causal | synthetic learned validation | full vs H1 LCB `0.5734` | PASS |
| Belief planner beats a matched recurrent sufficient-statistic policy | not shown | paired difference `0.0` | NONINFERIOR only |
| Primary learned DPC validation | empirical synthetic | return `0.768`, success `0.863`, Brier `0.0723`, ECE `0.0440` | PASS |
| OOD calibration | empirical synthetic | ECE `0.0928 > 0.08` | FAIL diagnostic |
| Signed eligibility carries delayed reward | tabular mechanism | success `1.0` vs three controls `0.4824` | PASS |
| BrainRuntime STDP task efficacy | unchanged | prior `NO-EFFECT`, held-out `FAIL` | OPEN / default off |

No AGI, world-model superiority, or runtime-learning superiority claim is made.

