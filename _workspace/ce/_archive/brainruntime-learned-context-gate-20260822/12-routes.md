# Routes

Status: COMPLETE

Chosen route: a separate two-actuator Hebbian gate trained from context cue and local entry-branch eligibility. It preserves the frozen BA-TR3 recurrent dynamics and never reads the endpoint decoder.

Rejected for this iteration:

- Reward/policy-gradient gating: stronger as a reinforcement-learning test but explicitly consumes endpoint correctness and would answer a different question.
- Direct context-to-mask labels or identity initialization: these merely rename the oracle.
- Adding context neurons or direct context-to-$Y$ edges: these introduce a trivial state/output channel.
- Edge-support discovery: the present two-context fixture cannot identify general morphology and would conflate selector learning with a much larger search problem.

If the chosen route passes, the next falsifier is a fresh task with more than two candidate masks and held-out cue compositions. If it fails, retain the failure and do not rescue it with reward, decoder gradients, thresholds, or endpoint tuning on the same seeds.
