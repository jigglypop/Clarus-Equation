# Clarus-Equation Codex rules

## Default: direct implementation

- For ordinary code, test, documentation, and harness work: inspect the target, make the smallest scoped change, and run one focused validation.
- Do not create a CE research run, preregistration, audit bundle, or full report unless the user explicitly asks for research, a new scientific claim, formal closure, preregistration, or release evidence.
- Do not run bare `pytest`, the full suite, all benchmarks, or packaging by default. Use the narrowest changed test or a source-only check first.
- Keep one implementation owner. Use subagents only for independent read-only mapping or research, and audit a stable snapshot after the implementation owner stops editing.

## Validation tiers

- FAST (default, target <=15 s): source parse/compile or one focused test file/node.
- STANDARD (explicitly useful, target <=60 s): the changed subsystem and its adjacent integration test.
- FULL/LOCK (explicit request only): full pytest, release gates, scientific stages, or irreversible V5 workflows.

For pytest, disable the cache provider and use a unique temporary basetemp outside the repository. Never run an irreversible scientific stage as a routine validation.

## CE research

Use `$ce-research` only for genuinely research-grade work. A supplied `CE_RUN` or explicit audit task activates its contract -> lanes -> audit -> implementation workflow. Ordinary fixes bypass that workflow.

V5 source lock and one-shot execution must use a fresh independent clone outside OneDrive/reparse-backed paths.
