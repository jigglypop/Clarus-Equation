# Machine validation

Status: COMPLETE

## Results

| Gate | Result |
|---|---|
| metadata fetch | `PASS`: 223 assets, canonical list hash `ff206a13191908e92167817c60644265066ba578d8a58ea1ad1011466dcb47d5` |
| manifest file | `PASS`: 204,834 bytes, SHA-256 `8ae094546532cc654dd6d49f3ffe5284f734598e7f6e05369a762579bba60e88` |
| selected payload bytes | `PASS`: exactly `1,273,970` |
| selected payload SHA-256 | `PASS`: `40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e` |
| HDF5 signature | `PASS`: `89-48-44-46-0d-0a-1a-0a` |
| schema traversal | `PASS`: 151 objects, no dataset value reads |
| deterministic rerun | `PASS`: byte-identical schema JSON |
| source event columns | `PASS_SCHEMA`: ID/start/stop/power/target/pattern/site |
| target trace/identity apparatus | `PASS_SCHEMA_CANDIDATE`: 393x105 signals plus ROI-to-NeuroPAL fields |
| explicit canonical source join | `BLOCKED_EXPLICIT_JOIN` |
| no-light/sham comparator | `BLOCKED_CONTROL`: not established in the inspected exemplar schema |
| light-vs-no-light intervention effect | `BLOCKED_LIGHT_VS_NOLIGHT_TAU` |

The schema artifact SHA-256 is
`45e53bb20739b3e1bbe61e9108422ad5e0f85cce9516af37f0b1df631433e54f`.
The inspector source SHA-256 is
`d8a6ba3e7419cee2bc8d968fc572ee3e1a80878432cefe1e1b5ea3ef1ddfd8c1`.

## Scientific status

This validation establishes a real event/trace schema candidate, not an
effect.  The full contract's `PASS_EVENT_SCHEMA` is not met because the
canonical source identity join and comparator/missingness fields remain
unresolved.  It is therefore invalid to calculate the contract's
light-vs-no-light `tau`, claim a direct edge, or connect this result to the
output-Fisher metric as a mediator.

The only newly opened route is a future, separately preregistered
source-choice active-control comparison, contingent on validating assignment,
positivity, source identity and post-treatment exclusion rules across the
published event set.  The exemplar's absence of a named comparator does not
prove dataset-wide absence; broader acquisition must remain a separately gated
schema audit.
