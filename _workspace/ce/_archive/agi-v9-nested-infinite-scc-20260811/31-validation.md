# Independent validation: V9 nested infinite-SCC isolated unit

Status: COMPLETE

## Verdict

The authorized mathematical/software unit passes independent validation. No P0
or P1 implementation defect remains in the reviewed scope. This is not security
work and does not promote `V9-1`, a development mechanism, biological identity,
AGI, physical infinity, arbitrary-state quotient exactness, or infinite-horizon
computability.

The review read `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`,
and `20-audit.md` completely, then inspected exactly the five implementation/test
files listed in `30-implementation.md`. No implementation source or test was
modified.

## Executed commands and exact results

Focused tests, with every warning promoted to an error:

```powershell
uv run --extra dev python -m pytest tests/test_nested_scc_tower.py tests/test_adaptive_scc_tower_controller.py -q -W error
```

Observed result:

```text
162 passed in 3.90s
exit code 0
```

Deterministic non-evidence demo, also with warnings promoted to errors:

```powershell
uv run --extra dev python -W error examples/agi/nested_scc_tower_demo.py
```

Observed result: exit code `0`. The demo reported four strongly connected nested
prefixes at depths `0..3`, one component per prefix, a complete finite causal
cone, a certified global-coordinate-sup bound `q=0.54`, exact zero-fixture
compatibility, refused generic append-zero compatibility with witness defect
`0.08275431183960441`, and a real consumed `CutUp` message of zero. Its emitted
claim boundary was `finite unit behavior only; no truncation, performance,
biological, or AGI claim`.

The first sandboxed demo attempt did not initialize the external `uv` cache due
to an environment access denial. Repeating the identical command with permitted
cache access produced the successful result above. This was an environment-only
failure before the program started, not a test or V9/SCC failure.

Raw file hashes were recomputed with:

```powershell
Get-FileHash -Algorithm SHA256 'reality_stone\python\reality_stone\clarus\nested_scc_tower.py','reality_stone\python\reality_stone\clarus\adaptive_scc_tower_controller.py','tests\test_nested_scc_tower.py','tests\test_adaptive_scc_tower_controller.py','examples\agi\nested_scc_tower_demo.py'
```

All five hashes matched the exact ledger in `30-implementation.md`.

CE phase gates were checked with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File '.codex\hooks\run.ps1' check '_workspace\ce\agi-v9-nested-infinite-scc-20260811' gate
powershell -NoProfile -ExecutionPolicy Bypass -File '.codex\hooks\run.ps1' check '_workspace\ce\agi-v9-nested-infinite-scc-20260811' build
```

Observed results were `OK gate` and `OK build`, both with exit code `0`. A
direct invocation of `run.ps1` was initially refused by the host PowerShell
execution policy; the explicit process-local `-ExecutionPolicy Bypass` rerun
executed the same repository hook successfully and did not change repository
policy.

## Requirement matrix

| Required boundary | Independent result |
| --- | --- |
| graph nesting and SCC typing | PASS: strict finite prefix nesting; one SCC per larger fixed graph; lower prefix not reported as a second maximal SCC |
| finite causal cone | PASS: total complete predecessor rule, finite BFS cone, birth-depth and manifest certificate |
| positive-delay unroll | PASS: every event edge advances time; unroll reported acyclic with singleton SCC count equal to event-vertex count |
| global-coordinate-sup `q` | PASS: schedule-specific, depth-independent row-sum bound; strict/cap and schedule mismatch gates exercised |
| exact versus refused append-zero boundaries | PASS: zero fixture exact; generic unit cube exact only for structural zero upward gain; every tested positive gain refused |
| canonical builtin input schemas | PASS: exact integer/string/digest/permutation schemas are revalidated; booleans, numeric text, subclasses, forged dataclasses, lying containers, and nonfinite fields fail closed |
| atomic state changes | PASS: generator-seal, reset, event, overflow, token-build, intervention, and snapshot failures preserve the prior committed state |
| process-local snapshot provenance | PASS: process-local HMAC boundary is explicit; tampering and parameter mismatch fail before commit |
| pending intervention continuation | PASS: each of six pending intervention types survives a snapshot roundtrip and reproduces token, trace, state, and readout exactly |
| generator/controller seals | PASS: spec, manifest, private operators, write flags, generator identity, and parameter identity are bound and checked |
| token-only readout | PASS: forecast signature is token-only; policy adds only an action feasibility mask; stale, foreign, unissued, or state-inconsistent tokens fail |
| all six interventions | PASS: reset, up-cut, down-cut, one-tick shift, sign flip, and hash-bound shuffle mutate actual next-update tensors with independent storage |
| dimensionless checks | PASS: positive frozen scales normalize observations; gains, `q`, defects, states, and readout share the normalized dimensionless convention; nonfinite/overflow paths are warning-free refusals |
| no runtime integration | PASS: only the isolated focused unit tests and deterministic demo were executed; no runtime adapter exists in the reviewed surface |

The test evidence supports the finite implementation and its causal
instrumentation. Finite tests are not a proof that every possible generated
level is correct, that an arbitrary completed state is exactly representable,
or that the ideal infinite tower is physically instantiated. The mathematical
theorems and counterexamples in `11-math.md` remain the source of those formal
boundaries.

## Baseline and unopened-data record

The previously recorded full-repository baseline was not rerun in this focused
validation. It had unrelated missing-artifact and policy failures, with no
V9/nested-SCC test failure. Those unrelated baseline conditions are neither
reclassified nor attributed to this unit.

No V8 locked test data, ACBSM reserved data, V9 development/evidence data,
historical seed block, or confirmatory data was opened. No seed-role scan,
training, fitting, scoring, registered development execution, or runtime
integration occurred. The deterministic unit demo contains only literal fixture
observations and produces no evidence artifact.

## Remaining P2 boundaries before development registration

These P2 items do not block the present isolated-unit PASS, but each must close
before any V9 development registration:

- `depth_error_tolerance` and `hysteresis_ticks` are dormant/reserved in the
  current grow-only unit. Remove them or make their registered semantics govern
  the actual controller before development.
- `generated_parameter_count` is not an active/free-coefficient count, a MAC
  count, or a matched-capacity measure. It must not enter a development budget
  until those quantities are separately defined and independently recomputed.
- Underscore-prefixed state, delayed-message, and diagnostic fields are non-public
  unit internals. An evidence runner must not expose or consume them as model
  inputs, readout bypasses, scorer information, or hidden control channels.
- The snapshot HMAC certifies same-process continuation only. It does not support
  cross-process provenance or persistence; those require a separately specified
  serialization and verification boundary.

## Closure

The implementation satisfies the authorized P1 software gates within its exact
finite unit scope: typed graph objects, schedule-specific contraction,
compatibility refusal, complete predecessor queries, immutable state-token
readout, real interventions, atomic state transitions, sealed provenance, and
dimensionless normalized inputs. No complete counterexample defeats those
narrow claims.

The following remain deliberately unclaimed and blocked:

- uniform truncation or infinite-horizon error certification;
- an exact quotient or arbitrary completed-state computation;
- adaptive-depth predictive benefit, matched-control superiority, or lesion
  effect floors;
- any development or confirmatory V9 result;
- any biological, cognitive, consciousness, physical-infinity, or AGI claim.

Final disposition:

```text
ISOLATED GRAPH/CONTROLLER UNIT       COMPLETE / PASS
P0 OR P1 DEFECT IN REVIEWED SCOPE   NONE FOUND
DEVELOPMENT DATA OPENED             NO
RUNTIME INTEGRATION                 NONE
V9-1 CAUSAL MECHANISM               UNTESTED
V9 CONFIRMATION                     BLOCKED / NOT REGISTERED
BIOLOGICAL OR AGI CLAIM             UNTESTED / NOT AUTHORIZED
```

CE_RUN=_workspace/ce/agi-v9-nested-infinite-scc-20260811
