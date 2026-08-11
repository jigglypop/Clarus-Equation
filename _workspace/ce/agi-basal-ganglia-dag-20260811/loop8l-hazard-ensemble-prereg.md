# Loop 8L preregistration — finite multi-timescale hazard ensemble

Status: LOCKED BEFORE IMPLEMENTATION

This is the final closure attempt for the current synthetic basal-ganglia DAG
track. It is an engineering joint HMM, loosely inspired by multiple neural
timescales; it is not a claim that basal-ganglia nuclei encode hazard functions.

The fixed, validation-independent hypothesis set is

`H=(0.00, 0.03, 0.06, 0.12, 0.24)`

with a uniform prior. Maintain one joint posterior over hazard model `m` and
context `c`:

`Jbar_t(m,c)=sum_j J^+_{t-1}(m,j) T^(h_m)_{j,c}`,

`J^-_t(m,c) proportional to Jbar_t(m,c) exp(cue_t,c)`,

`J^+_t(m,c) proportional to J^-_t(m,c) L_t(c; chosen_action, outcome)`.

Action selection uses only the context marginal `sum_m J^-_t(m,c)` and the
Loop 8I soft content support. Hazard weights are the marginal `sum_c J(m,c)`.
Cue and outcome likelihoods are each applied exactly once; there is no separate
model-weight update, reset, decay, fitted temperature, or hazard tuning.

## Arms

1. fixed-hazard `0.06` Loop 8K;
2. joint hazard ensemble candidate;
3. same ensemble with hazard weights frozen uniform after each update;
4. support derangement;
5. outcome sign flip;
6. signed heuristic, hard recurrent, and soft feedforward references.

New seeds: ID `881000..881031`, OOD `881100..881131`, matched stationary and
flat nulls from `881200`. No prior validation seed is reused.

## Ten final gates

1. joint posterior, cue posterior, model weights, and context marginal are
   finite, nonnegative, and normalized within `1e-12`;
2. transition, cue, and outcome joint-filter identities have max residual <=
   `1e-12`, with zero degenerate evidence;
3. candidate minus fixed-hazard accuracy LCB >= `0` ID/OOD;
4. candidate minus fixed-hazard post-switch +1..+4 accuracy LCB >= `+0.03`
   ID/OOD;
5. candidate minus hard recurrent accuracy LCB >= `-0.01` ID/OOD;
6. hard recurrent minus candidate NLL LCB > `0` ID/OOD;
7. candidate minus frozen-weight ensemble accuracy LCB > `0` ID/OOD;
8. candidate minus support derangement accuracy LCB >= `+0.05` ID/OOD and
   candidate minus outcome sign flip >= `+0.10` ID/OOD;
9. posterior mean hazard for OOD exceeds ID by >= `0.03`; stationary posterior
   mass on `h=0` exceeds every other individual hazard mass;
10. stationary candidate/fixed absolute accuracy difference <= `0.02`, flat
    candidate minus matched-flat accuracy <= `0.01`, and all causal, topology,
    finite, pending-order, reset, and decay integrity counters pass.

Every gate contributes ten diagnostic points. GO is conjunctive. Whether GO or
STOP, no further tuning loop is authorized in this track; remaining work moves
to broader tasks, real neural data, or runtime integration only if warranted.
