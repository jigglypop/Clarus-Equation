# Route map

Status: COMPLETE

| Route | Information used | Exact matched control question | Maximum claim |
|---|---|---|---|
| `FULL` | all learned edges | new delayed/heterogeneous apparatus baseline | unmasked runtime capability |
| `WEIGHT` | $|W|$ only | is magnitude pruning sufficient? | sparse weight routing |
| `CLUSTER` | cue-active blocks and block strengths | does mesoscale support help? | cue-conditioned cluster routing |
| `PATH_ONLY` | cluster support plus forward relevance | does cue-path ranking help without return structure? | path routing |
| `TOPOLOGY` | `PATH_ONLY` plus local return support | does path-plus-cycle ranking add value? | topology-specific routing only if its ablations lose |
| `RETURN_SHUFFLED` | identical support/path with permuted return values | is the location of return support necessary? | return-location falsifier |
| `RANDOM_MATCHED` | random learned edges, same budget | generic sparsity null | none |
| `WRONG_CONTEXT` | next cue, same constructor | is cue alignment necessary? | context falsifier |

Implementation order is full apparatus reproduction, exact-budget mask unit
tests, M1 binding preservation, and finally T1 held-out development.  No
confirmation seed is opened unless the registered development gate passes.

The topology-specific verdict requires `TOPOLOGY` to differ from and beat
both `PATH_ONLY` and `RETURN_SHUFFLED`.  If topology fails those ablations but
a cue-conditioned sparse route beats full or generic controls, only the
narrower path/cluster result survives.  Threshold, budget, partition, horizon,
decoder, and seed changes after results are forbidden.  Slow usage
consolidation is out of scope until routing itself passes.
