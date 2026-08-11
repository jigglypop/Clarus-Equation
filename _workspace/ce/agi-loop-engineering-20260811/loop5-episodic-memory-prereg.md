# Loop 5 — audited episodic-memory completeness preregistration

Status: COMPLETE — corrected run `90/100 GO`

Implementation audit note: the first artifact
`loop5-episodic-memory-validation.json` omitted the registered no-memory arm and
is INVALID for scoring. The arm was added without changing candidate logic,
seeds, thresholds, or hard gates. The corrected run uses a separate artifact.

The corrected artifact includes all registered arms. Candidate latest-value,
evidence-id, abstention, and deletion scores are each `1.0`; composite LCBs are
`+0.75` over existing memory, `+0.75` over merge-off, and `+0.25` over FIFO.

## Claim limit

Synthetic bounded-capacity key/value memory mechanics only. This is not a
LoCoMo/LongMemEval result, biological hippocampus validation, or AGI claim.

## Candidate operations

- `ADD`: insert a normalized cue, value, evidence id, priority, and timestamp;
- `UPDATE`: if cosine similarity exceeds `0.92`, replace the matching value and
  evidence id instead of consuming another slot;
- `DELETE`: remove an explicit evidence id and append an audit event;
- `NOOP`: reject nonfinite/invalid operations without silent mutation;
- `RECALL`: return the top evidence only if similarity is at least `0.60` and
  the top-two margin is at least `0.05`; otherwise abstain.

At capacity, eviction uses a frozen utility combining registered priority,
novelty relative to other keys, and recency. All quantities are dimensionless.

## Fixed comparisons

- existing `HippocampusMemory` priority-only implementation;
- candidate with UPDATE merge disabled;
- candidate with abstention disabled;
- FIFO key/value memory;
- no-memory chance control.

## Evaluation blocks

For each fixed seed, eight latent concepts receive an initial fact and a noisy
same-cue update, followed by four interference items at capacity 12. Queries
test latest-value recall, evidence-id correctness, unseen-cue abstention, and
explicit deletion. Thirty-two seeds are paired across all arms.

## Hard gates

1. latest-value and evidence-id accuracy are each at least `0.90`;
2. unseen-cue abstention and deletion correctness are each at least `0.95`;
3. paired 95% bootstrap LCB of candidate composite score minus existing memory,
   merge-off, and FIFO is above `0.10`;
4. abstention-off unseen false-recall rate is at least `0.20` worse;
5. memory length never exceeds capacity and delete/update audit counts exactly
   match issued operations;
6. all arms use identical keys, noise, priorities, update order, and queries.

Any hard-gate failure scores `0/100 STOP`. One unit-scale implementation debug
is allowed, followed by one fixed-seed scored run. No post-run threshold,
capacity, or noise sweep is allowed.
