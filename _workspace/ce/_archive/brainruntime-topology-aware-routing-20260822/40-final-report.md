# Topology-aware routing on the delayed BrainRuntime

Status: COMPLETE

## Abstract

This run prepared a target-free, exact-budget comparison of full, magnitude,
cluster, path, return-aware topology, and matched control masks on the actual
Torch `BrainRuntime`.  The apparatus retained heterogeneous thresholds, STP,
refractory state, lifecycle selection, and a two-slot delay ring.  Focused
implementation checks passed `5/5`.  The mandatory unmasked M1 binding
baseline then failed `0/16`, so the topology endpoint was not scientifically
opened.  A diagnostic showed that the same learned weights decode all three
associations without delay but remain below the decoder threshold with the
current delayed propagation.  The formal result is an apparatus failure, not
evidence for or against topology-aware brain routing.

## Result and interpretation

The route family was designed so that mask construction sees only the sealed
learned weight, the present cue, architectural blocks, a public seed, route
name, and budget.  The topology score combines forward cue reachability with
one- and two-step return support.  `PATH_ONLY` and `RETURN_SHUFFLED` were added
to distinguish a genuine return-support effect from generic cue-conditioned
sparsity.

That comparison cannot yet be interpreted.  With the full learned weight and
no sparse mask, every development circuit failed clean and corrupt pairwise
binding.  Because the contract required at least 15/16 full-baseline passes,
all topology, path, cluster, and energy-efficiency claims stop at admission.

The diagnostic isolates the immediate engineering problem.  The weight is not
empty or intrinsically incapable: when evaluated without the delay ring it
recovers each trained target.  In the delayed path, recurrence appears only
weakly after several ticks and never approaches the fixed decoder boundary.
The current implementation gates an old buffered activation with the current
lifecycle mask, so a source that was active when written can be suppressed
when it arrives.  Repairing that event-time semantics is a new runtime route;
changing the horizon or decoder in this run is forbidden.

## Claim boundary

This negative result says nothing about whether biological brains select
motifs, whether graph morphology is memory, whether such routing saves
physical energy, or whether it has clinical value.  It establishes only that
the present delayed `BrainRuntime` is not an admissible platform for the
registered topology comparison.

## Reproduction

Focused tests and the machine summary are linked from `31-validation.md`.
Confirmation seeds were never opened.  Resume requires an independently
tested delay repair that preserves source-event gating through the ring buffer
and passes a shared Torch runtime continuation fixture; topology routing must
then start under a new contract rather than reinterpret this invalid run.
