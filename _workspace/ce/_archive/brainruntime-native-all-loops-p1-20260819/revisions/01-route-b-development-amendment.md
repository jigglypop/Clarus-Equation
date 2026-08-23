# Route B development amendment

Status: COMPLETE

After the three P1 audit corrections were closed, the active user objective still required an
actual-runtime attempt for every loop. The unchanged causal-STDP Route A continued to fail Loops 8
and 9, so development work proceeded on the already isolated supervised Route B. This amendment is
post-development disclosure, not a claim that the Route B mechanism was preregistered originally.

Changes explored only on seeds 97101--97108 were:

1. exclude a deleted temporal item from the recall denominator and score it separately as required
   abstention;
2. use a fixed cue drive sufficient to cross `BrainRuntime` activation thresholds;
3. stage cue/value rows in `HippocampusMemory` for the causal-STDP diagnostic route;
4. replace correlated random codewords with independently generated, seed-fixed orthonormal cue
   and target codebooks;
5. make Route B an explicitly supervised bounded recurrent projection, and make Loop 9's write
   factor-local so the held-out `(1,1)` result is compositional rather than a memorized pair.

Several intermediate artifacts are retained. The selected candidate is
`artifacts/route-b-development-results-p8-orthocode.json`. No seed in 98101--98132 was read during
selection.

Route B changes the real `BrainRuntime.weight` and all prediction comes from cue plus six native
zero-input `step()` calls after both temporal and hippocampal stores are physically empty. It is
nevertheless a supervised matrix projection, not evidence that the repository's local STDP rule
learned the association.
