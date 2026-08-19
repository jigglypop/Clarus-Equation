# Routes — CloudCell eligibility and next causal route

Status: COMPLETE

## R1 — metric-only CloudCell route

**Status: PASS_INPUT, conditional on per-recording preflight.**

Use the AML32 moving panel only, preserve its native neural/behavior timebase,
fit the two-dimensional source chart and likelihood exclusively on training
time, and score the frozen output-Fisher estimator on held-out contiguous
windows.  The primary comparison is context-specific output likelihood versus
shared-context, gain-only, diagonal, nuisance-only, and reference-dominated
alternatives.  The replicate unit is a recording/animal, never time bins or
neurons.

Its result can be: “within the specified GCaMP measurement process, an
observational output-relative Fisher geometry was estimable.”  It cannot be a
physical Riemann metric, cortical geometry, curvature, or a mechanism of
locomotion.

## R2 — arbitrary neuron-group transfer diagnostic

**Status: DIAGNOSTIC_ONLY.**

For a fixed, outcome-blind partition of simultaneously recorded rows, a
three-group design can put source fluorescence in A, target fluorescence in
B, and a disjoint global-state proxy in C.  The held-out contrast is the
log-score gain from adding source history to target-history and nuisance
history.  Evaluate every fixed ordered group pair; do not select a best pair.

The partition has no neuron identity, anatomical label, or connectome join.
It is therefore a check for time-forward conditional predictability in the
recorded arrays, not a router candidate and not an empirical substitute for
the missing A-to-B population definition.  Its nulls are target-history-only,
block-preserving source shift, time reversal/negative lag as diagnostics, and
global-proxy augmentation.  None supplies causal identification.

## R3 — anatomical route on CloudCell

**Status: BLOCKED_SOURCE_TARGET_DEFINITION.**

The required entry condition is a per-unit canonical neuron identity plus a
documented registration/error model joining the recording to an anatomical
connectome.  `XYZcoord`, row order, correlation clustering, or an
outcome-selected assembly are not substitutes.  No parameter, threshold,
decoder, or synthetic seed can repair this missing apparatus input.

## R4 — causal route: Randi et al. E2SYT

**Status: recommended next route; input acquisition and source audit required.**

The next route should be the Randi et al. Nature 2023 E2SYT dataset named in
the contract, rather than a new CloudCell seed experiment.  It is selected
because the route is expected to contain a time-locked perturbation setting;
the source audit must verify the actual per-recording fields before any claim.

Admission requirements are:

1. synchronized neural, behavior, event, and intervention timestamps;
2. pre-specified, canonical source and target identities or region labels;
3. randomized source intervention assignment, sham, reverse-direction, and
   non-target controls;
4. a fixed pre-intervention chart, horizon, output likelihood, and held-out
   split; and
5. source bytes, release/version, and per-animal provenance frozen before
   fitting.

Only if `do(A)` changes future B and behavior against all matched controls may
the result be called a causal intervention effect.  Even then, it is not proof
of a specific synaptic edge and does not identify G as mediator without an
independent mediator intervention or a justified causal state model.

## Falsifier retained

AML18 is a GFP-only control.  Its matched positive lag signature falsifies
the inference from CloudCell lag predictability to neural calcium memory or
routing.  Any R1/R2 analysis must retain this boundary; a positive G or row
diagnostic R cannot override it.

