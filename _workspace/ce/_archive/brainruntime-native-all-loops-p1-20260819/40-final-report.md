# BrainRuntime native loops report

Status: COMPLETE

## Current outcome

All five requested loops have now been executed through the real Torch `BrainRuntime` on the eight
development seeds.

- Loop 6: 8/8 GO through latest-valid temporal selection.
- Loop 7: 8/8 GO through the actual runtime-agent route, including supplied-context precedence.
- Loop 8: causal-STDP Route A remains 0/8 STOP; the explicitly supervised bounded recurrent
  projection Route B is 8/8 GO.
- Loop 9: Route A remains 0/8 STOP; factor-local bounded Route B is 8/8 GO on the held-out `(1,1)`
  intervention while the shuffled control is 0/8.
- Loop 10: 8/8 GO for the frozen next-state predictor against persistence.

Route B mutates the real recurrent matrix, rebuilds the runtime sparse matrix, physically removes
both episodic stores, then predicts only from cue plus six zero-input runtime steps. Thus it is a
valid native-weight/readout demonstration. It is a supervised projection, however, and must not be
reported as local-STDP learning or biological consolidation. Loop 10 remains bounded state
prediction, not evidence of phenomenal consciousness.

## Confirmation verdict

The untouched 98101--98132 confirmation set reproduced the development split:

- Route A: Loops 6, 7, and 10 each passed 32/32; Loops 8 and 9 passed 0/32.
- Route B: Loops 8B and 9B each passed 32/32 with zero-store rollout and matched shuffled/no-write
  controls.

Therefore the requested actual-runtime execution is complete for Loops 6--10. The strongest valid
claim is a composite engineering result: temporal selection, agent routing, and bounded
self-prediction work on Route A; memory binding and factorized intervention transfer work only after
an explicit supervised bounded projection into the native recurrent matrix. The experiment rejects
the stronger claim that the current local causal-STDP rule performs Loops 8 and 9.
