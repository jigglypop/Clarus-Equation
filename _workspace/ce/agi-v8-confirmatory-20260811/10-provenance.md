# V8 Git provenance audit

Status: READY FOR BASELINE; V8 EXECUTION MUST REMAIN LOCKED

Audit date: 2026-08-11 (Asia/Seoul)

## Conclusion

The repository can establish a defensible V8 chronology, but it cannot do so
by committing the V8 registration immediately from the current state.  The
current `HEAD` contains none of the sparse-causal V1--V7 lineage.  V1--V6 are
present only in the index, while V7 registration, implementation, tests,
validation artifact, and closure document are untracked.  A V8 registration
commit directly on `HEAD` would therefore refer to parents that its own Git
ancestor cannot reproduce.

The safest history is:

```text
fcb754e (current origin/main)
  |
  B0  historical V1--V7 + R1 checkpoint baseline
  |
  R8  V8 registration only, externally timestamped before implementation
  |
  I8  canonical V8 implementation + tests + exact hash lock
  |
  A8  V8 validation artifact (only after all locks verify)
  |
  T8  locked-test artifact, only if A8 passes unchanged
```

`B0` records the exact historical material used by V8.  It must explicitly
state that it is a retrospective checkpoint and does **not** prove that the V7
registration preceded the V7 implementation or result.  The prospective
claim begins only at `R8`: Git ancestry must prove `R8 < I8 < A8`.

No V8 validation or test seed occurrence was found outside the new research
contract.  Searches for `80100`, `80355`, `81100`, `81355`, and
`sparse_causal_bridge_v8` found only the declared seed ranges in
`00-contract.md` (apart from unrelated numeric substrings in benchmark data).

## Observed repository state

- branch: `main`, tracking `origin/main`, ahead/behind `0/0`;
- `HEAD`: `fcb754ee5b1f35324e9502d3b3f146387eb25823`;
- author/commit time of `HEAD`: `2026-08-10T22:13:52+09:00`;
- all V1--V6 registrations, parent modules, tests, and V4/V5 artifacts are
  staged additions but absent from `HEAD`;
- V7 registration, implementation, runner, test, validation artifact, and
  closure document are untracked and absent from every inspected Git history;
- `.gitignore` and three AGI status documents are modified but unstaged;
- `.gitattributes` is untracked;
- local-memory artifacts, pytest temporary directories, and the whole
  `_workspace/` tree are also untracked and must not be swept into a commit.

The current V7 raw locks reproduce the closure document:

| Object | SHA-256 |
|---|---|
| V7 registration | `134ddaa793170b898649b79e11407c10f35d1468ba95701544a06905d9448c3e` |
| V7 implementation | `7abf17f260f0046cb6eace7ed57e1115657c2dd4d32bd1024bc7c1940e910310` |
| V7 test definition | `866e9e89274419b17e4b33a63df519c89565e6763480b7c8537b5f7b0ec88041` |
| V7 validation artifact | `f172447ecc0d19ac206c6625bf5911805f28214bb5adc1d2a215c59dc3bc4e12` |

This is file-integrity evidence, not historical commit-order evidence.

## Required B0 historical baseline

Create a dedicated branch before making any commit:

```powershell
git switch -c agi/v8-confirmatory-20260811
```

Switching branches does not clean or absorb the existing worktree.  Immediately
re-run `git status --porcelain=v2` after switching and stop if any path changed
unexpectedly.

`B0` should contain only the exact lineage needed to reproduce V8:

1. sparse-causal V1--V7 registrations;
2. V1--V7 parent modules, runners, and tests;
3. V4/V5 parent integrity and validation artifacts;
4. the failed V7 validation artifact and closure document;
5. the AGI status edits that correctly delimit the failed V7 claim;
6. `.gitattributes`, because the V4/V5/V7 lock scheme deliberately distinguishes
   LF source/config files from CRLF result artifacts;
7. `.gitignore` additions required to expose the registered JSON files;
8. a compact R1 development checkpoint: at minimum
   `fresh_parent_anchor_pilot.py`, `21-pilot-lock.md`, `31-validation.md`, and
   `40-final-report.md`, or equivalent canonical copies plus a machine-readable
   R1 result artifact.

Do not add `_workspace/` recursively.  It contains Cargo build output, scratch
programs, pytest temporaries, and unrelated research runs.  Do not add the
local-memory AML32 artifacts to this commit; they are not parents of V8.

Use an explicit PowerShell path array and path-limited commit.  Do not use
`git add .`, `git add -A`, or a bare `git commit`:

```powershell
$v8BaselinePaths = @(
  '.gitattributes',
  '.gitignore',
  'artifacts/agi/sparse_causal_bridge_integrity_v4.json',
  'artifacts/agi/sparse_causal_bridge_integrity_v5.json',
  'artifacts/agi/sparse_causal_bridge_validation_v4.json',
  'artifacts/agi/sparse_causal_bridge_validation_v5.json',
  'artifacts/agi/sparse_causal_bridge_validation_v7.json',
  'docs/7_AGI/1_AGI.md',
  'docs/7_AGI/8_Roadmap.md',
  'docs/7_AGI/12_Equation.md',
  'docs/7_AGI/26_Sparse_Causal_Bridge_V7_Closure.md',
  'examples/agi/sparse_causal_bridge_gate.py',
  'examples/agi/latent_causal_bridge_gate.py',
  'examples/agi/free_rollout_bridge_gate.py',
  'examples/agi/reliability_rollout_bridge_gate.py',
  'experiments/preregistration/sparse_causal_bridge_v1.json',
  'experiments/preregistration/sparse_causal_bridge_v2.json',
  'experiments/preregistration/sparse_causal_bridge_v3.json',
  'experiments/preregistration/sparse_causal_bridge_v4.json',
  'experiments/preregistration/sparse_causal_bridge_v5.json',
  'experiments/preregistration/sparse_causal_bridge_v6.json',
  'experiments/preregistration/sparse_causal_bridge_v7.json',
  'reality_stone/python/reality_stone/clarus/sparse_causal_bridge.py',
  'reality_stone/python/reality_stone/clarus/latent_causal_bridge.py',
  'reality_stone/python/reality_stone/clarus/free_rollout_bridge.py',
  'reality_stone/python/reality_stone/clarus/reliability_rollout_bridge.py',
  'tests/test_sparse_causal_bridge.py',
  'tests/test_latent_causal_bridge.py',
  'tests/test_free_rollout_bridge.py',
  'tests/test_reliability_rollout_bridge.py',
  '_workspace/ce/agi-v8-breakthrough-20260811/fresh_parent_anchor_pilot.py',
  '_workspace/ce/agi-v8-breakthrough-20260811/21-pilot-lock.md',
  '_workspace/ce/agi-v8-breakthrough-20260811/31-validation.md',
  '_workspace/ce/agi-v8-breakthrough-20260811/40-final-report.md'
)
git add -- $v8BaselinePaths
git diff --cached --check -- $v8BaselinePaths
git diff --cached --name-status -- $v8BaselinePaths
git commit --only -m 'research(agi): checkpoint historical V1-V7 and R1 lineage' -- $v8BaselinePaths
```

Before the commit, compare the displayed name list with the array exactly.
`--only` is important: it prevents any staged path outside the explicit list
from entering `B0`.  It leaves unrelated staged changes in the index.  If a
listed file contains mixed unrelated edits, stop and split that file manually;
path limitation cannot separate hunks within one file.

After `B0`, record its commit ID in the V8 registration as the parent-baseline
commit.  A clean checkout of `B0` must reproduce all hashes that V8 names.

## Required R8 registration-only commit

The canonical registration should be a self-contained
`experiments/preregistration/sparse_causal_bridge_v8.json`.  It may name V7 and
R1 as lineage, but it should not depend on a mutable runtime merge with V7.
Copy every active V8 field into V8 itself:

- exact candidate equation and clipped gain estimator;
- training seeds `45100..45107`, origins `80,100,...,500`, 176 windows, and
  expected gain `0.7868543064870357` with tolerance;
- disclosed development seeds `79100..79355`;
- validation seeds `80100..80355` and locked-test seeds `81100..81355`;
- all primary conjunctions, margins, CI unit/method, critical values,
  normalization, leakage rules, stability thresholds, and claim boundary;
- exact B0 commit plus SHA-256 locks for every parent source, parent artifact,
  V7 failed result, and R1 development checkpoint;
- an `implementation_lock_path` and a rule requiring that lock to be committed
  after implementation but before any V8 seed is generated;
- `no_second_v8_route_after_validation` and test-unlock rules.

Do not put an invented future implementation hash in `R8`.  The defensible
two-phase rule is: `R8` fixes semantics and paths; `I8` later fixes the actual
code/test hashes before execution.  This avoids either a circular hash or a
hidden pre-registration implementation.

The `R8` commit must contain only the V8 registration and, if needed, the one
`.gitignore` line that exposes it:

```powershell
$v8RegistrationPaths = @(
  'experiments/preregistration/sparse_causal_bridge_v8.json',
  '.gitignore'
)
git add -- $v8RegistrationPaths
git diff --cached --check -- $v8RegistrationPaths
git diff --cached --name-status -- $v8RegistrationPaths
git commit --only -m 'research(agi): preregister V8 parent-anchored confirmation' -- $v8RegistrationPaths
git show --stat --oneline HEAD
git show HEAD:experiments/preregistration/sparse_causal_bridge_v8.json
```

For stronger independent time evidence, push this branch or a signed/annotated
tag **before creating the canonical V8 module, runner, tests, or artifacts**:

```powershell
git tag -a agi-v8-registration-20260811 -m 'V8 registration before canonical implementation' HEAD
git push -u origin agi/v8-confirmatory-20260811
git push origin agi-v8-registration-20260811
```

A local commit proves ancestry inside the eventual history, but a remote push
or other external timestamp is stronger evidence against later history
rewriting.  Pushing is an external mutation and requires the user's explicit
authorization.

## Required I8 implementation-and-lock commit

Only after `R8` is committed (and preferably externally timestamped) may the
canonical V8 implementation and tests be created.  Unit tests at this stage
must use tiny synthetic fixture seeds outside both V8 blocks; they must not
loop over or sample `80100..80355` or `81100..81355`.

Before any V8 evaluation, create a machine-readable implementation-lock
manifest containing at least:

- raw and canonical V8 registration SHA-256;
- B0 and R8 commit IDs;
- SHA-256 of canonical V8 implementation, runner, and test definitions;
- SHA-256 of every inherited parent source/artifact named by the registration;
- recomputed gain and exact tolerance result;
- declared validation/test seed ranges and an assertion that neither has run;
- line-ending policy used for each locked file.

Commit only the explicit implementation paths and lock manifest as `I8`, then
verify `git merge-base --is-ancestor <R8> HEAD`.  The implementation runner
must refuse to execute either split when the working files differ from the I8
lock.

## Result sequence

1. Run non-evaluative unit/integrity tests and verify all locks.
2. Generate `80100..80355` exactly once and write the V8 validation artifact.
3. Commit that artifact and closure report as `A8` with explicit pathspecs.
4. If and only if the frozen conjunction passes and all hashes still match,
   run `81100..81355` once.
5. Commit the test artifact separately as `T8`; otherwise preserve the failed
   validation artifact and leave the test absent.

At each step record:

```powershell
git log --reverse --format='%H %aI %s' <B0>^..HEAD
git diff --exit-code <R8> -- experiments/preregistration/sparse_causal_bridge_v8.json
git merge-base --is-ancestor <R8> <I8>
git merge-base --is-ancestor <I8> <A8>
```

The two `merge-base` commands must return exit code 0.  The registration diff
must be empty from `R8` through every result commit.

## Tradeoff assessment

| Choice | Assessment |
|---|---|
| Registration-only commit | Required. It gives the clearest prospective boundary and must precede canonical implementation/results. |
| Baseline commit | Required here because the parents are absent from `HEAD`. It binds the exact historical checkpoint but cannot retroactively prove V7 order. |
| Dedicated branch | Strongly recommended. It isolates the formal sequence and avoids placing retrospective baseline material directly on `main`; branch creation alone is not a timestamp. |
| Self-contained V8 config | Strongly recommended. It prevents mutable inheritance/merge behavior from silently changing active gates. Duplication is acceptable because reproducibility is more important here. |
| One commit containing baseline + V8 registration | Rejected. It weakens the prospective boundary and makes parent-vs-registration review ambiguous. |
| One commit containing registration + implementation | Rejected. Git cannot prove which existed first. |
| Committing all staged/untracked files | Rejected. The worktree contains unrelated local-memory artifacts, scratch files, build outputs, and temporary directories. |
| Local tag only | Helpful label but weak independent timing evidence; a pushed tag/branch or external timestamp is stronger. |

## Go/no-go gate

Proceed to V8 execution only when all of the following are true:

- B0 cleanly reproduces every parent and R1 lock named by V8;
- R8 is a distinct ancestor of I8;
- the registration is unchanged after R8;
- I8 and its exact implementation-lock manifest are committed;
- no V8 validation/test seed has been generated before I8;
- the runner rejects dirty or hash-mismatched V8 files;
- unrelated user changes remain outside every path-limited commit.

Until then the formal status is `READY FOR BASELINE`, not `READY TO RUN`.

