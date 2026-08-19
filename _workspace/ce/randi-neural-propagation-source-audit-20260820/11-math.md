# Intervention and identifiability lane

Status: COMPLETE

## Three objects must remain separate

For animal/session $i$, event $e$, canonical source $A$ and target $B$:

1. $R^{A\to B}$ is an observational held-out predictive log-score gain.
2. $\tau$ is a contrast between potential outcomes under stimulation policies.
3. $G$ is a Fisher tensor derived from a separately frozen outcome likelihood.

None determines or mediates either of the others without additional
assumptions or an independent mediator intervention.

## Why pre/post is not a no-stimulation effect

Let $Y^B_{ie}(t)$ denote target activity relative to event onset. The naive
within-event difference

$$
Y^B_{ie}(+\Delta)-Y^B_{ie}(-\Delta)
$$

does not identify a light-stimulation effect. If $Y(t)=\beta t+\epsilon(t)$
and light has no effect, its expectation is still $2\beta\Delta$. State drift,
calcium convolution, adaptation and carryover give the same counterexample.
A pre-period is a baseline covariate, not a randomized $do(u=0)$ arm.

Likewise a failed/autoresponse-negative event is post-treatment. If latent
excitability $L$ determines both source autoresponse and target response,
conditioning on autoresponse selects on $L$ and biases the effect. The
published atlas's autoresponse inclusion rule is therefore valid for its
“measured propagation among successful activations” estimand but not for an
intention-to-stimulate effect.

## Route A: randomized source-choice effect

The article states that the stimulated neuron sequence was mostly randomly
selected. If the event-level file records the actual assignment $Z_{ie}$ and
the randomization strata, the strongest identifiable contrast available
without a no-light arm is an active-source policy effect:

$$
\tau^{q}_{A:C\to B}(\Delta)
=\mathbb E\left[
Y^B_{ie}(Z=A;\Delta)-Y^B_{ie}(Z\in C;\Delta)
\right],
$$

where $C$ is a predeclared set of other stimulated neurons and $q$ fixes
duration, power and timing. Both arms receive light, so common stimulation
artifacts are partly controlled. This is a causal effect of targeting $A$
rather than the active-control source set under the experimental policy. It
is not $do($endogenous neural state $A)$ and is not a direct synaptic edge.

Identification requires all of the following:

- actual source choice was randomized with known/derivable strata and positive
  support within animal/session;
- target $B$ was observed under both source choices with the same identity and
  preprocessing;
- stable dose/timing or an explicitly stratified policy;
- predeclared washout/carryover model for the approximately 30-s schedule;
- no exclusion based on autoresponse or downstream outcome;
- complete assignment/missingness records and fixed analysis windows.

If “mostly random” cannot be reconstructed as an assignment mechanism, this
route becomes a conditional association between intervention labels and
outcomes, not a design-identified causal effect.

## Route B: light-versus-no-light effect

For binary assignment $U_{ie}$, the contract's original estimand is

$$
\tau^q_{A\to B}(\Delta)
=\mathbb E\left[Y^B_{ie}(U=1;q)-Y^B_{ie}(U=0;q)\right].
$$

It requires randomized no-light/sham events or a separately justified
randomized pseudo-onset scheme. WT versus `unc-31` is a genotype effect
modifier, not $U=0$. Time-shifted pseudo-onsets can diagnose drift but do not
by themselves create randomized no-light counterfactuals. Therefore this route
is `BLOCKED_CONTROL` until the event schema proves an eligible comparator.

## Counterexamples to stronger claims

1. Light may create a global optical/thermal artifact $L\to Y^B$ while neural
   $A$ has no effect. Active-source controls reduce but do not eliminate this.
2. $A\to C\to B$ and $A\to B$ can yield the same target response. The experiment
   identifies effective propagation, not a direct edge.
3. $Y^B_{obs}=Y^B+\kappa U$ yields a nonzero optical readout contrast without a
   physiological change in $B$.
4. Calcium kinetics at roughly 2 volumes/s do not establish millisecond
   latency, transmitter identity or monosynaptic direction.
5. `unc-31` changes organism-wide signalling and may change baseline state;
   a genotype difference is not automatically mediation by dense-core vesicles
   for a particular pair.

## Relation to the rebuilt metric

If the NWB event layer contains pre-stimulation neural state and target traces,
one may define a target-neural outcome Fisher tensor

$$
G^{B\leftarrow A}_{ab}(z)
=\mathbb E\left[
\partial_a\log p(x^B_{post}\mid z,h,Z)
\partial_b\log p(x^B_{post}\mid z,h,Z)
\right].
$$

This is an intervention-conditioned neural-output geometry, not behavioral
geometry and not physical anatomy. It must use a pre-event chart, calibration
only, a fixed target likelihood and common state support. Even if both $G$ and
$\tau$ change, $G\to\tau$ mediation remains `BLOCKED_NOT_IDENTIFIED`.

## Inference unit

Neuron pairs and time bins are repeated observations. The independent unit is
animal/session. Assignment tests must permute within the actual
animal/session/block randomization strata. Uncertainty is clustered or
hierarchical at animal/session; canonical pair-specific claims require the
same $A,B$ replicated across animals. Report animals, sessions, events and
pairs separately.

## Math verdict

| Claim | Verdict |
|---|---|
| event-triggered effective response among successful activations | `PASS_PUBLISHED_ESTIMAND` |
| randomized source-choice total effect | `CONDITIONAL_EVENT_SCHEMA` |
| light versus no-light total effect | `BLOCKED_CONTROL` |
| endogenous neural-state $do(A)$ | `BLOCKED` |
| direct/monosynaptic $A\to B$ | `BLOCKED` |
| intervention-conditioned target-neural $G$ | `CONDITIONAL_EVENT_SCHEMA` |
| $G$ mediation of $\tau$ | `BLOCKED_NOT_IDENTIFIED` |
