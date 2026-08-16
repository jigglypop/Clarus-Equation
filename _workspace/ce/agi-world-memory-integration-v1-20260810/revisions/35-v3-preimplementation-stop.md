# G9-CBM V3 pre-implementation stop record

Status: BLOCKED_PRE_IMPLEMENTATION

Gate: STOP

## Scope and execution boundary

This record closes V3 at the implementation boundary. The earlier static
preregistration audits established that the registered merge and cue amendment
were internally reproducible; they did not establish that the full scientific
procedure was uniquely executable.

No V3 implementation-lock artifact was created. No registered
train/calibration, validation, or locked-test seed was run, and no calibration,
validation, test, or integrity artifact was created. The registered V3 seed
roles therefore remain unopened, but they may not be reused for a repaired
version of this design.

## Outcome-determining blockers

The prototype and the locked registration do not determine one auditable
scientific execution for all required paths:

1. **Shuffled direction and partial-rejection ambiguity.** The shuffled-binding
   attribution does not uniquely fix the direction and accounting treatment
   when only part of a candidate set is rejected. Different reasonable
   treatments can change its paired primitive vector and one-sided gate.
2. **Lesion and zero-synthetic semantics.** The unconstrained-lesion control and
   the zero-synthetic control are not specified tightly enough to determine
   which values are preserved, replaced, or exposed to the candidate and which
   work remains diagnostic-only. Those choices can change both attribution and
   provenance results.
3. **Unused owner ledger and resource constants.** Registered ownership and
   resource literals exist in the prototype, but constants or allocated arrays
   that are not connected to capability-instrumented owners cannot prove the
   registered 29-field budget vector or equal-work condition.
4. **Missing shadow-recall and lesion accounting.** The inactive LTM path does
   not yet demonstrate all 72 shadow facade-call slots and 576 masked-distance
   rows, and the lesion path does not yet provide complete, owner-derived
   accounting for its registered classification, acceptance, padding, and
   provenance work. Allocation alone is not an execution witness.
5. **Placeholder evaluator and checks.** The evaluator truth, common-noise
   futures, control metrics, factorial primitive vectors, summaries, and exact
   55/17/12 gates are not implemented end to end. The current split report uses
   placeholder check values, so it cannot produce a scientific PASS or FAIL.
6. **Stage failure, deletion, and unlock gaps.** The stage machinery does not
   yet preserve every required failure artifact, enforce that deletion never
   authorizes a registered rerun, and prove the complete committed-validation
   unlock chain without an impermissible pre-unlock test read. These gaps make
   the registered run-count and one-shot state machine incomplete.

Each item can affect a reported result, resource gate, or permission to open a
later split. They are therefore not clerical implementation fixes under the
locked V3 registration.

## Prototype-only static evidence

[산출] The unregistered prototype scaffold currently has the following local
engineering evidence:

- 28 tests passed: 26 unit tests and 2 integrity tests;
- Ruff passed;
- `py_compile` passed.

These checks support syntax, API, merge, cue-boundary, and selected static
invariants only. They used no registered scientific seed and are **not**
scientific confirmation of memory, dreaming, rollout, planning, attribution,
resource equality, or any AGI claim.

## Required successor

[미완성] Continuing this research requires a V4 amendment with fresh registered
train/calibration, validation, and locked-test seeds and fresh artifact paths.
Before V4 implementation lock, that amendment must make every blocker above
byte-level executable and independently auditable. V3 remains an unexecuted
negative implementation-boundary witness; it must not be silently patched or
reported as a scientific experiment.
