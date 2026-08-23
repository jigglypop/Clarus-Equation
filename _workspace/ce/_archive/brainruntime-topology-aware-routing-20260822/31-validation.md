# Validation

Status: COMPLETE

Overall result: `APPARATUS_INVALID / TOPOLOGY_UNTESTED`.

## Focused implementation checks

Command:

```powershell
.codex\hooks\python.cmd pytest tests/test_runtime_topology_routing.py -q --junitxml=_workspace/ce/brainruntime-topology-aware-routing-20260822/artifacts/focused-tests.xml
```

Result: `5 passed` in `3.46s`.  The tests cover the constructor allowlist and
AST forbidden-name scan, exact budget and diagonal exclusion, return-term mask
separation, exclusion of other cue-source blocks, degenerate fail-closed
behavior, actual delayed heterogeneous-threshold Torch execution, snapshot
immutability, cutoff, finite receipts, and shared-snapshot parity across arms.
Only the existing PyTorch sparse-CSR beta warnings were emitted.

All named implementation obligations passed in the focused fixture:

| Obligation | Result |
|---|---|
| `V-INPUT` | PASS |
| `V-BUDGET` | PASS |
| `V-DEGENERATE` | PASS |
| `V-SNAPSHOT` | PASS |
| `V-CUTOFF` | PASS |
| `V-FINITE` | PASS |

## Frozen full-baseline admission

The required full delayed/heterogeneous-threshold M1 baseline was executed on
seeds `97201..97216`.  It passed the binding gate in `0/16` circuits: mean
clean and corrupt accuracy were both `0.0`, versus the required at least
`15/16`.  Snapshot immutability, finite state, and store cutoff each passed
`16/16`.  Mean simulator measurements were runtime energy `0.1385872731`,
active fraction `0.1309213790`, and exposed-edge fraction `0.0828512743`.

The admission failure terminates the experiment before a topology comparison.
Confirmation seeds were not opened.  One default factor seed `97301` and one
binding topology pilot were run during implementation before the full baseline
failure was established; both are discarded and carry no scientific verdict.
The full T1 development family was not executed.

## Failure localization

For diagnostic seed `97201`, the weight learned in the delay-on apparatus was
copied unchanged to a no-delay runtime.  At the frozen endpoint, the three
known associations had correct-target cosines `0.487134`, `0.820158`, and
`0.799856`, decoding indices `0`, `2`, and `3`.  Under delay-on rollout, their
largest correct-target cosines over ticks 0--10 were only `0.050863`,
`0.096260`, and `0.108974`, all below the frozen `0.20` decoder threshold.

Thus the learned recurrent matrix retains binding capacity, but the current
delay/lifecycle execution suppresses its propagation.  This is consistent
with the Torch path multiplying the delayed activation by the current
lifecycle mask rather than a mask receipted with the delayed source event.
The diagnostic does not rescue or retune this run.

Machine summary:
`artifacts/development-baseline-summary.json`.  Focused JUnit:
`artifacts/focused-tests.xml`.
