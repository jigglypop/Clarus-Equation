# Alternative routes

Status: COMPLETE

## Route A — functional agent closure (recommended main model)

Build one `ClosedLoopBrainRuntime` around the existing Layer A-E and `RuntimeAgent`. Add a typed critic contract, generalized causal belief/world model, actual neuromodulator feedback, explicit memory operations, persistence, and a real environment adapter. Keep every addition switchable for ablation.

## Route B — CE emergence validation (separate experiment)

Build a tiny explicit-spike SNN with membrane state, spike timestamps, eligibility traces, three-factor learning, and WAKE/NREM/REM cycling. Test whether the target activity ratio emerges without a forced mask. This tests the substrate claim and must not be mixed with Transformer regularization experiments.

## Route C — more sparse-bridge variants

Low priority. V7 was closed, V8 failed its confirmatory clause, later output-gain/prefix/multi-origin variants did not recover it, and ACBSM is HOLD. Only reopen after a genuinely new causal mechanism and fresh preregistration.

Rejected as immediate direction:

- rerunning all failed tests through a nominal V9;
- treating ACBSM's training screen as held-out validation;
- adding more graph/manifold/shared-cloud wrappers without a runtime causal role;
- forcing the 4.87% mask or using post-hoc top-k as evidence of natural emergence.

