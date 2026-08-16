# G9-CBM V5 irreversible state-machine draft

Status: COMPLETE

Gate: PASS

Gate recommendation: PASS for incorporation into the frozen V5 amendment and
registration, followed by a fresh independent recursive-merge/state-machine
audit. This lane-level PASS does not authorize implementation or any registered
seed.

## Scope

This draft closes only the state-machine P0s in
`36-v4-independent-prereg-audit.md`,
`37-v4-executable-contract-redteam.md`, and
`39-v4-preimplementation-stop.md`:

1. every outcome-selecting role, including implementation lock, gets a durable
   Git-committed consumed witness before its work begins;
2. the consumed transition and the role work occur in one public invocation;
3. final bytes are published from one exact pending path without overwriting an
   existing final path;
4. missing, empty, partial, noncanonical, hard-crash, deletion, and mutation
   states are all terminally classified without pretending that a killed
   process serialized an exception;
5. caught exceptions and early per-origin feasibility failures use the exact
   non-self-referential `RegisteredStageFailureV5` contract from the V5 API
   closure; and
6. history, test unlock, implementation-lock test evidence, Git commits, and
   deletion probes have one executable Windows recipe.

The V4 registration and amendment remain immutable negative witnesses. No V4
or V5 registered seed, RNG, scientific world, calibration, validation, test,
or scientific runner was opened while preparing this draft. V5 inherits the V4
scientific equations, thresholds, cells, and controls unchanged; this document
defines integrity machinery only.

### Recursive-merge hygiene

V5 may not leave the V4 intent protocol active beside this protocol. Its
`delete_paths` must therefore contain these exact paths:

```json
[
  ["artifact_state_machine"],
  ["test_lock"],
  ["implementation_closure_v4", "stage_protocol"],
  ["implementation_closure_v4", "handcrafted_test_evidence"],
  ["implementation_closure_v4", "test_unlock"]
]
```

V5 then adds one complete `artifact_state_machine`, one complete `test_lock`,
and the state-machine, handcrafted-evidence, and unlock records under
`implementation_closure_v5`. Because the recursive merge hard-fails on a
missing delete target, no descendant deletion such as an old
`opened_failure_exact_fields` or `*_intent_path` may also appear after its
parent deletion. The V5 `allowed_new_override_paths` must list the three newly
added top-level/subtree paths exactly. This whole-subtree replacement is
required to remove V2/V4 `run_count=0` implementation-lock semantics, all three
intent stages, all three intent paths, the same-descriptor finalization rule,
the ambiguous exception hash, and `UnlockRecordV4`; textual supersession while
leaving those leaves in the merged object is not sufficient.

The new `artifact_state_machine.stages_ordered` is exactly
`["preregistration","implementation_lock","calibration","validation",` plus
`"test","integrity","verify"]`. The complete object must encode sections
1--12 below. The new `test_lock` contains exactly the five primary paths, four
consumed-marker paths, four pending paths, four final paths, integrity path,
eight protected refs, and the literal `UnlockRecordV5` designation. Other V5
lanes may add their independently audited records only outside these complete
state-machine objects.

## 1. Exact V5 paths and protected refs

All registered strings use repository-relative POSIX separators. Resolution is
from the Git top-level directory. A path is valid only when every existing
ancestor and the final object is inside that directory and is neither a symlink
nor a Windows reparse point. The pending and final paths are in the same
directory and therefore on the same volume.

The primary implementation paths are exactly:

```text
experiments/preregistration/agi_world_memory_integration_v5.json
reality_stone/python/reality_stone/clarus/agi_world_memory_integration_v5.py
examples/agi/agi_world_memory_integration_v5_gate.py
tests/test_agi_world_memory_integration_v5.py
tests/test_agi_world_memory_integration_integrity_v5.py
```

The four irreversible roles use this exact table:

| role | consumed marker | pending file | final artifact | expected registered seeds |
|---|---|---|---|---:|
| `implementation_lock` | `artifacts/agi/agi_world_memory_integration_implementation_lock_consumed_v5.json` | `artifacts/agi/agi_world_memory_integration_implementation_lock_v5.json.pending` | `artifacts/agi/agi_world_memory_integration_implementation_lock_v5.json` | 0 |
| `calibration` | `artifacts/agi/agi_world_memory_integration_train_calibration_consumed_v5.json` | `artifacts/agi/agi_world_memory_integration_train_calibration_v5.json.pending` | `artifacts/agi/agi_world_memory_integration_train_calibration_v5.json` | 40 |
| `validation` | `artifacts/agi/agi_world_memory_integration_validation_consumed_v5.json` | `artifacts/agi/agi_world_memory_integration_validation_v5.json.pending` | `artifacts/agi/agi_world_memory_integration_validation_v5.json` | 40 |
| `test` | `artifacts/agi/agi_world_memory_integration_test_consumed_v5.json` | `artifacts/agi/agi_world_memory_integration_test_v5.json.pending` | `artifacts/agi/agi_world_memory_integration_test_v5.json` | 60 |

The derived integrity ledger is exactly
`artifacts/agi/agi_world_memory_integration_integrity_v5.json`; its temporary
publication path is that string plus `.pending`. `verify` has no output path
and is repeatable/read-only. Integrity and verify execute zero registered seeds
and never authorize a scientific role. They therefore are not members of the
four-role irreversible set; a failed integrity-ledger write may be regenerated
from unchanged committed scientific bytes, but no scientific file may be
repaired, promoted, replaced, or reopened.

The local protected Git refs are exactly:

```text
refs/agi-world-memory-v5/consumed/implementation-lock
refs/agi-world-memory-v5/consumed/calibration
refs/agi-world-memory-v5/consumed/validation
refs/agi-world-memory-v5/consumed/test
refs/agi-world-memory-v5/final/implementation-lock
refs/agi-world-memory-v5/final/calibration
refs/agi-world-memory-v5/final/validation
refs/agi-world-memory-v5/final/test
```

The V5 registration must freeze Git object format `sha1`; each protected-ref
value is consequently exactly 40 lowercase hexadecimal characters. The
consume branch is exactly
`refs/heads/research/agi-world-memory-integration-v1`. Rewriting or deleting
Git objects, the protected refs, or the `.git` directory is outside the stated
failure model and makes verification fail; no filesystem-only protocol claims
survival after destruction of its append-only store.

Every V5 JSON and hash-locked Markdown path must be covered by `text eol=lf`.
Every marker and artifact writer emits UTF-8 without BOM and LF only.

## 2. Canonical bytes and non-self-referential hashes

`canonical_json_v5(x)` is exactly

```python
(json.dumps(
    x,
    sort_keys=True,
    indent=2,
    ensure_ascii=True,
    allow_nan=False,
) + "\n").encode("utf-8")
```

There are no trailing spaces. Raw SHA-256 always means lowercase
`sha256(path.read_bytes()).hexdigest()`. For every payload containing
`canonical_payload_sha256_excluding_this_field`, verification requires the
exact registered key set, removes that one key entirely, canonicalizes the
remaining object with `canonical_json_v5`, and hashes those bytes. Hashing the
field with a null value, hashing the final self-containing object, or omitting
any other field is forbidden.

No consumed marker contains its own raw SHA-256 or its commit OID. No final
artifact contains its own raw SHA-256 or final commit OID. Those values become
available only after the corresponding bytes or commit exists and occur only
in a later artifact, a protected ref, or the derived integrity history. This
removes every hash/commit cycle.

## 3. Exact consumed marker

`ConsumedMarkerV5` has exactly these 22 fields:

```text
artifact_kind
experiment
stage
status
final_artifact_path
pending_path
consumed_protected_ref
final_protected_ref
git_object_format
preconsume_head_oid
registration_raw_sha256
amendment_raw_sha256
merged_registration_sha256
predecessor_raw_sha256_by_stage
ordered_path_records_sha256
callable_records_sha256
ordered_allocation_ledger_sha256
numpy_version
expected_registered_seed_count
registered_seed_execution_count_before
unlock_record
canonical_payload_sha256_excluding_this_field
```

The fixed values are
`artifact_kind="agi_world_memory_integration_consumed_marker_v5"`,
`experiment="agi_world_memory_integration_v5"`, `stage` equal to one of the
four role strings, `status="CONSUMED"`, and `git_object_format="sha1"`.
`registered_seed_execution_count_before` is exactly zero. The two manifest
digests hash the complete canonical ordered path-record array and canonical
ordered callable-record array; the arrays themselves remain copied
byte-for-value through the registered scientific artifacts.

`predecessor_raw_sha256_by_stage` has exactly the keys
`implementation_lock`, `calibration`, and `validation`. Values unavailable
before the current role are JSON null. Thus all three are null for
implementation lock; only implementation lock is nonnull for calibration;
implementation lock and calibration are nonnull for validation; and all three
are nonnull for test. `unlock_record` is null for the first three roles and is
the exact `UnlockRecordV5` in section 10 for test.

The marker's payload digest uses section 2's exclusion recipe. The raw marker
SHA becomes `_RegisteredStageCapabilityV5.open_witness_sha256` and the failure
schema field `consumed_marker_raw_sha256`. V5 must replace the V4/API-draft name
`intent_raw_sha256`; retaining both names or treating them as aliases is
forbidden because there is no V5 intent stage.

## 4. One invocation and the durable consume commit

There is one public mutating call,
`run_agi_world_memory_integration_v5_gate(stage, output_path)`. `stage` is one
of `implementation_lock`, `calibration`, `validation`, `test`, or `integrity`.
For the first four, `output_path` must byte-equal the exact final path in
section 1. Omission, a different spelling, an alternate root, a symlink/reparse
traversal, or a pre-existing final/pending path fails before consumption. There
is no intent command, resume command, finalize command, recovery command, dry
run, alternate output, environment bypass, or callable that opens registered
seeds.

Before touching a role path, the invocation performs only deterministic reads
and requires all of the following:

1. `git rev-parse --show-toplevel`, `--show-object-format`, and
   `git symbolic-ref HEAD` match the registered repository, `sha1`, and branch;
2. no merge, rebase, cherry-pick, revert, or bisect control file is active;
3. the index and worktree are clean, including untracked nonignored files;
4. registration, amendment, every ordered source, and every predecessor are
   tracked, clean at their path, and byte-identical to `git show HEAD:<path>`;
5. every current-or-later V5 marker, pending, final, and protected ref that
   should still be unopened is absent;
6. `git log --all --full-history --format=%H -- <exact-path>`, the index, the
   worktree, all refs, and all reflogs contain no deletion/open witness for an
   unopened role; and
7. Git commit hooks are absent, `core.hooksPath` is unset, interactive prompts
   are disabled, and commit signing is disabled for this invocation.

The invocation then executes this exact consume transition:

1. Create the exact consumed-marker path directly with
   `os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_BINARY, 0o644)` on
   Windows. Write all canonical marker bytes with a checked write loop, call
   `os.fsync(fd)`, and close it. A short write or fsync/close error stops before
   any registered work.
2. Run `git add -f -- <consumed-path>` with `shell=False`. Verify the cached
   name/status record is exactly one `A` record for that path and that the Git
   blob bytes equal the worktree bytes.
3. Invoke exactly one Git commit attempt with message
   `agi-world-memory-v5 consume <role>`, `--no-gpg-sign`, `--no-verify`, fixed
   author and committer name `Clarus AGI V5 Gate`, fixed email
   `agi-v5@localhost.invalid`, `GIT_TERMINAL_PROMPT=0`, and `shell=False`.
   Author/committer timestamps are Git's current clock and are explicitly not
   scientific payload fields.
4. Require the new commit to have exactly one parent equal to
   `preconsume_head_oid`, an exact one-path add delta for the consumed marker,
   and marker blob bytes equal to the fsynced bytes.
5. Atomically create the role's consumed protected ref using
   `git update-ref --create-reflog <ref> <consume-commit> 0000000000000000000000000000000000000000`.
   Re-read it and require exact equality to the consume commit.
6. Only now construct the module-private `_RegisteredStageCapabilityV5` in the
   same process. Its `open_witness_sha256` is the raw marker SHA-256 and its
   private seal cannot be serialized or reconstructed.

The implementation must not return successfully between steps 1 and 6. The
consume commit and registered work are one invocation; a later invocation that
finds the marker, its commit, or its protected ref always refuses. If a process
dies before step 5, no registered RNG, world, or implementation-lock pytest
subprocess was permitted. If it dies after step 5, the role is irrevocably
consumed even if no pending or final file exists.

Only one commit attempt is allowed at step 3. Commit/amend/retry loops and
`git reset`, `checkout`, `restore`, `clean`, ref deletion, path deletion, or
index repair are forbidden. External HEAD movement after step 5 is detected;
it terminates the role and never authorizes a retry.

## 5. Pre-work pending envelope and single-use capability

After the consumed ref verifies, but before implementation-lock pytest or the
first registered RNG/world operation, the same invocation exclusively creates
the exact pending path. `OpenedUnfinishedV5` has exactly these 18 fields:

```text
artifact_kind
experiment
stage
status
failure_phase
expected_registered_seed_count
registered_seed_execution_count_at_checkpoint
registration_raw_sha256
amendment_raw_sha256
merged_registration_sha256
predecessor_raw_sha256_by_stage
consumed_marker_raw_sha256
consumed_commit_oid
preconsume_head_oid
ordered_path_records_sha256
callable_records_sha256
ordered_allocation_ledger_sha256
canonical_payload_sha256_excluding_this_field
```

Its fixed values are
`artifact_kind="agi_world_memory_integration_opened_unfinished_v5"`,
`status="OPENED_UNFINISHED"`, `failure_phase="OPENED_UNFINISHED"`, and
`registered_seed_execution_count_at_checkpoint=0`. Its payload hash follows
section 2. The writer uses `O_CREAT|O_EXCL|O_WRONLY|O_BINARY`, a checked write
loop, `os.fsync`, and retains the descriptor in the private capability. Only
after the exact bytes have been read back and verified may the implementation
lock launch pytest or a scientific evaluator claim its first seed ordinal.

The checkpoint count describes the instant the opening envelope was fsynced;
it is not later interpreted as the execution count after a hard crash. A hard
death can occur after an arbitrary number of seeds without updating these
bytes, so integrity reports a null execution count for an unfinalized
scientific role rather than inventing zero.

The capability is valid only for the role, ordered registered seed tuple,
pending descriptor, marker hash, and consume commit that created it. Claiming a
seed ordinal increments its in-memory execution count before that seed's first
RNG/world operation. It is single-use, noncopyable, nonpicklable, inaccessible
to candidate code, and invalidated before final publication. No child process
receives it. The implementation-lock role has no registered-seed capability
claims and its count remains zero.

## 6. Exact caught failure and normal terminal payloads

The V5 API closure's `RegisteredStageFailureV5` is authoritative for caught
Python exceptions and validation/test per-origin feasibility failure. Its exact
23 fields and exclusion hash recipe are inherited here except for one required
state-machine replacement:

```text
intent_raw_sha256  ->  consumed_marker_raw_sha256
```

The replacement key holds the exact marker raw SHA-256. Both keys may not
coexist. The API rules for `CAUGHT_EXCEPTION`, the eight phase codes, seed and
origin ordinals, completed/claimed counts, and the two feasibility failure
codes remain exact. Calibration, validation, and test never fabricate an
incomplete inherited full report. A deterministic per-origin failure is
written immediately as `RegisteredStageFailureV5` after the sealed candidate
digest and the six true valid-candidate costs exist.

Implementation lock has no scientific early-failure schema. A caught runner,
Git-independent pytest-launch, parser, or evidence-construction exception uses
the same 23-key layout with `stage="implementation_lock"`, expected/completed/
execution counts all zero, seed/origin/feasibility/candidate fields null,
`failure_code="CAUGHT_EXCEPTION"`, and the unavailable predecessor hashes
null. V5 must explicitly extend the schema's stage enum to include
`implementation_lock`.

Normal calibration, validation, and test artifacts retain their recursively
merged V4 exact field sets and add exactly these seven required provenance
fields:

```text
artifact_kind
amendment_raw_sha256
merged_registration_sha256
consumed_marker_raw_sha256
consumed_commit_oid
preconsume_head_oid
canonical_payload_sha256_excluding_this_field
```

Their `artifact_kind` is respectively
`train_calibration_v5`, `validation_report_v5`, and `locked_test_report_v5`.
The exact-key requirement is the inherited key set union these seven keys; an
already inherited `canonical_payload_sha256_excluding_this_field` occurs once,
not twice, and no other key is allowed. Their payload hash uses section 2.
Normal status is exactly `PASS` or `FAIL`; the calibration symbolic
`REJECT_ALL` and complete validation/test gates keep their previously
registered meanings.

An uncatchable process termination does not and cannot create
`RegisteredStageFailureV5`. It is represented only by the durable consumed
commit plus whatever raw pending/final bytes physically survived, and is
classified by section 9.

## 7. Atomic publication and final commit

The complete normal or caught-failure object is first constructed in memory
and canonicalized. Using the still-open pending descriptor, the runner seeks to
offset zero, writes the entire terminal byte string with a checked loop,
truncates to its exact length, calls `os.fsync`, closes the descriptor, re-reads
the pending bytes, and requires exact byte/hash equality. A hard interruption
during this rewrite may leave empty, mixed, or partial pending bytes; those raw
bytes are never parsed into a fictional exception.

The runner then calls `os.rename(pending_path, final_path)` on Windows, after
requiring both paths to be on the same volume and the final path to be absent.
Python's Windows `os.rename` fails when the destination exists, so it supplies
atomic no-replace publication; `os.replace` is forbidden. All handles are
closed before rename. Any rename error is terminal and leaves the raw pending
bytes untouched.

After publication, the runner requires `HEAD` still equal the consume commit,
the marker still equal its committed blob, the pending path absent, and the
only worktree change to be the final artifact. It then:

1. runs `git add -f -- <final-path>` and verifies an exact one-path `A` cached
   delta;
2. makes exactly one commit attempt with message
   `agi-world-memory-v5 finalize <role> <PASS|FAIL>` and the same noninteractive,
   unsigned, hook-free identity policy as the consume commit;
3. verifies the commit has exactly one parent equal to the consume commit, its
   only delta is the final path add, and its blob equals the published bytes;
4. creates the exact final protected ref with `git update-ref --create-reflog`
   against an all-zero old OID; and
5. re-verifies the ref, clean worktree/index, marker/final HEAD identity, and
   the complete raw hash chain.

The final artifact does not contain this final commit OID. The derived history
gets it from the protected ref. A commit or ref failure after publication is an
immutable terminal administrative failure: no retry, amend, second commit, or
new scientific invocation is permitted, and no later role may open.

## 8. Exact implementation-lock evidence

The implementation-lock consumed marker and protected ref must verify before
the handcrafted subprocess is launched. The exact subprocess is invoked with
`shell=False`, repository root as `cwd`, `PYTHONHASHSEED=0`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and this exact
argv tail:

```text
-m pytest -q --disable-warnings --maxfail=1 --color=no
  tests/test_agi_world_memory_integration_v5.py
  tests/test_agi_world_memory_integration_integrity_v5.py
```

The V5 registration must freeze the exact Python and pytest version strings
and the exact terminal-summary parser callable in the callable manifest. The
parser must return exactly one record with these 17 fields:

```text
argv_tail
unit_test_raw_sha256
integrity_test_raw_sha256
python_version
pytest_version
collected_count
passed_count
failed_count
error_count
skipped_count
xfailed_count
xpassed_count
deselected_count
exit_code
stdout_raw_sha256
stderr_raw_sha256
registered_seed_execution_count
```

Raw stdout and stderr are captured as bytes and never decoded before their
hashes are computed. Counts are integers parsed by the frozen parser; parse
failure raises `EvidenceParseErrorV5` and is finalized through the caught
`RegisteredStageFailureV5` route with phase `REPORT`; it does not invent count
integers. Valid evidence requires exit code zero, collected count positive,
passed count equal to collected count, every other outcome count zero, test
hashes equal the ordered source records, and registered-seed execution count
zero. The artifact stores the record, not a literal handcrafted-test success
Boolean. Mocking the subprocess/parser, assigning expected counts, or running
either registered evaluator invalidates the lock.

The normal implementation-lock artifact has exactly these 20 fields:

```text
artifact_kind
experiment
stage
status
failure_reasons
registration_raw_sha256
amendment_raw_sha256
merged_registration_sha256
consumed_marker_raw_sha256
consumed_commit_oid
preconsume_head_oid
ordered_path_raw_sha256
callable_source_sha256_by_symbol
numpy_version
ordered_allocation_ledger_sha256
registered_budget_vector
handcrafted_test_evidence
registered_seed_execution_count
source_tree_clean_after_tests
canonical_payload_sha256_excluding_this_field
```

`source_tree_clean_after_tests` is observed immediately after pytest and must
be true for `status="PASS"`. Here `true` means the index is empty and every
tracked or untracked worktree path is unchanged/absent except the exact pending
file that this invocation already opened; that pending file must equal the
current canonical opening envelope at the instant of the check. It is evidence,
not a substitute for the exact path/hash checks. A nonzero exit, nonzero
non-pass count, empty collection, hash mismatch, any other source-tree
mutation, or nonzero registered execution count yields a normal
implementation-lock `FAIL` with an exact sorted `failure_reasons` list. Parser
ambiguity and caught exceptions use section 6 instead.

## 9. Total terminal-state classification

Integrity inspects paths with `os.lstat`, hashes raw bytes before attempting
JSON decoding, reads committed blobs with `git show <protected-ref>:<path>`,
and never changes a role path. It applies this precedence for each role:

| terminal state | exact predicate | accepted status | seed execution count |
|---|---|---|---|
| `UNOPENED` | no marker/pending/final path, no protected ref, and no index/log/ref/reflog witness | null; eligible only if all predecessor gates hold | 0 |
| `PRECONSUME_DEBRIS` | marker or marker history exists but no valid consumed protected ref/commit | `INVALID` and no work permitted | 0 under the conforming protocol |
| `CONSUMED_NO_OUTPUT` | valid consumed ref/commit, no pending and no final | `HARD_CRASH` | 0 for implementation lock; null for scientific roles |
| `OPENED_UNFINISHED` | valid consumed ref and pending is the exact canonical opening envelope, final absent | `HARD_CRASH` | 0 for implementation lock; null for scientific roles |
| `PENDING_TERMINAL_BYTES` | valid consumed ref, pending is a canonical normal/failure terminal object, final absent | `HARD_CRASH`; pending PASS is never promoted to PASS | parsed count |
| `PENDING_CORRUPT` | valid consumed ref, final absent, and pending is empty, invalid JSON, noncanonical, wrong-key, mixed, or partial | `HARD_CRASH` | 0 for implementation lock; null for scientific roles |
| `FINAL_UNCOMMITTED` | valid consumed ref, final is canonical, but no valid final protected ref/commit | `INVALID`; no next stage | parsed count |
| `FINAL_CORRUPT` | valid consumed ref and current final bytes are empty, partial, noncanonical, wrong-key, or fail their payload hash | `INVALID` | 0 for implementation lock; null unless a valid count is parseable |
| `FINAL_COMMITTED_PASS` | consumed and final refs/commits, parent/delta/blob rules, current bytes, hashes, schema, and status all verify | `PASS` | exact final count |
| `FINAL_COMMITTED_FAIL` | same as prior row with exact status `FAIL` | `FAIL`; all later roles remain unopened | exact final count |
| `DELETED_OR_MUTATED` | a protected/history blob exists but marker/final is absent or differs in the worktree/index/HEAD | `INVALID` | exact committed count if recoverable, else null |
| `PATH_CONFLICT_OR_MULTIPLE_ATTEMPT` | pending and final both exist, more than one consumed/final add commit exists, a protected ref disagrees, or an unexpected later role exists | `INVALID` | null unless one unique committed terminal artifact supplies it |

An existing raw file is always preserved. No integrity command truncates,
deletes, renames, completes, promotes, or rewrites it. In particular, canonical
terminal bytes stranded at `.pending` are evidence of a crash window, not a
recoverable successful artifact. Missing and corrupt states use raw byte count
and raw SHA-256 in history without asserting a JSON failure type or message.

Deletion never restores `UNOPENED`: the consumed protected ref and commit are
the authoritative one-shot witness. Committing a deletion, moving a file,
resetting the index, changing branches, or making the current path absent only
changes the classification to `DELETED_OR_MUTATED`. A second marker commit,
second final commit, ref replacement, or non-fast-forward history is terminally
invalid. The runner never calls a destructive Git or filesystem command.

## 10. Exact unlock before test consumption

The test artifact path may be checked only with existence-only `lstat` before
unlock; its bytes may not be opened, read, mapped, or hashed. Before touching
the test consumed-marker or pending path, the runner requires validation state
`FINAL_COMMITTED_PASS`, a clean/index-empty HEAD containing the validation
final commit, and current marker/artifact bytes identical to their protected
commits.

It independently recomputes from validation `primitive_seed_vectors`, never
stored summaries, the exact registered key sets of 55 performance checks, 17
hard-zero checks, and 12 resource checks. It requires all 84 leaf values true,
exact equality to stored leaf Booleans, and exact recomputation/equality of
`performance_passed`, `integrity_passed`, `resource_passed`, and `passed`.
It then rehashes the V5 registration and amendment, recursive merged payload,
all ten ordered source records, every callable record, exact NumPy version,
ordered allocation ledger, and implementation-lock/calibration/validation raw
chain against current bytes and their protected commits.

Only after every comparison passes is `UnlockRecordV5` constructed. It has
exactly these 25 fields:

```text
artifact_kind
validation_raw_sha256
validation_commit_oid
registration_raw_sha256
amendment_raw_sha256
merged_registration_sha256
implementation_lock_raw_sha256
calibration_raw_sha256
ordered_path_raw_sha256
callable_source_sha256_by_symbol
numpy_version
ordered_allocation_ledger_sha256
performance_check_count
hard_zero_check_count
resource_check_count
total_check_count
performance_checks_sha256
hard_zero_checks_sha256
resource_checks_sha256
performance_passed
integrity_passed
resource_passed
passed
pre_test_head_oid
canonical_payload_sha256_excluding_this_field
```

The three check digests hash their complete canonical exact-key Boolean maps.
Counts are exactly 55, 17, 12, and 84; all four conjunctions are literal true.
`pre_test_head_oid` is the current HEAD before marker creation. The exclusion
hash uses section 2. The complete unlock object is copied byte-for-value into
the test consumed marker and later test artifact.

The unlock record cannot contain the not-yet-created test marker hash, consume
commit, test artifact hash, or final commit. The test marker may contain the
unlock object; its own exclusion hash then covers that object without a cycle.
After marker commit, the final test artifact adds the marker raw hash and consume
commit OID. A stored `passed=true` or `test_unlocked=true` is never sufficient
evidence by itself.

## 11. Exact history schema

The integrity ledger contains `stage_history` as exactly four records in this
order: `implementation_lock`, `calibration`, `validation`, `test`. Every
`StageHistoryRecordV5` has exactly these 17 fields:

```text
role
consumed_path
consumed_marker_raw_sha256
consumed_commit_oid
marker_worktree_matches_commit
final_path
pending_path
observed_path_kind
observed_raw_sha256
observed_byte_count
observed_parse_state
final_commit_oid
artifact_worktree_matches_commit
terminal_state
attempt_count
registered_seed_execution_count
status
```

`observed_path_kind` is one of `NONE`, `PENDING`, `FINAL`, `BOTH`, or
`DELETED_COMMITTED`. `observed_parse_state` is one of `ABSENT`,
`CANONICAL_OPEN`, `CANONICAL_TERMINAL`, `EMPTY`, `INVALID_JSON`,
`NONCANONICAL_JSON`, or `WRONG_SCHEMA`. Marker/commit/hash fields and match
Booleans are null when no corresponding object exists. `attempt_count` is the
number of distinct consumed-marker add commits found across all refs and
reflogs; it must be zero or one. `registered_seed_execution_count` is zero for
an unopened role and implementation lock, the exact final value for a unique
canonical terminal payload, and null for an unfinalized scientific hard-crash
state. `status` is null, `PASS`, `FAIL`, `HARD_CRASH`, or `INVALID` according to
section 9.

The integrity ledger also stores the exact protected-ref map, every raw current
and committed blob hash used by classification, the fixed four-record history,
`all_hashes_match`, `scientific_world_generation_count`, and
`scientific_artifact_mutation_count`. Its own
`canonical_payload_sha256_excluding_this_field` uses section 2; it never stores
its own raw hash or commit OID. `verify` recomputes the ledger in memory and
reports mismatches without writing.

## 12. Required zero-seed structural tests

Before V5 preregistration can receive an overall `Gate: PASS`, handcrafted
tests must establish without any registered role capability or seed:

1. a deleted uncommitted final cannot erase the committed consumed ref;
2. every second invocation refuses after each consume commit, including when
   marker, pending, or final worktree files are removed;
3. no pytest/RNG/world call occurs before consumed-ref verification and pending
   fsync;
4. a crash after marker creation, consume commit, protected-ref creation,
   pending creation, partial rewrite, complete pending rewrite, rename, final
   commit, and final-ref creation reaches the exact section-9 state;
5. empty, one-byte, truncated-at-every-byte, valid-but-noncanonical, wrong-key,
   and payload-hash-invalid pending/final bytes are preserved and classified;
6. Windows `os.rename` refuses an existing final path and never overwrites it;
7. marker/final commits have exactly one parent and one added path, and branch,
   protected refs, blobs, current bytes, and history agree;
8. caught exception and per-origin failure bytes match the exact 23-field API
   schema with `consumed_marker_raw_sha256` and its exclusion hash;
9. the implementation-lock subprocess produces observed structural counts and
   raw stream hashes with zero registered execution;
10. test unlock rejects each missing, extra, false, or mismatched one of the
    84 leaf checks, each conjunction mismatch, every source/hash/version change,
    and every premature test-byte read; and
11. no helper, CLI flag, environment switch, alternate path, second process,
    deserialized capability, Git retry, pending promotion, or integrity command
    can reopen or complete a consumed role.

These tests use temporary repositories and handcrafted off-range fixtures only.
They may not import or execute a V4 runner, construct a V4/V5 registered world,
or enumerate a registered V5 seed.

## Closure verdict

[Definition] A role becomes irrevocably consumed only when its exact marker
commit and protected ref verify. Registered work is impossible before that
point, and only the capability retained by that same invocation can continue.

[Derivation] Deleting any uncommitted pending/final file after consumption no
longer erases the witness, because the marker commit is reachable from a
write-once protected ref. A new process therefore cannot recreate the role.

[Derivation] A hard process death is totalized by raw-path state plus the
committed witness. Empty or partial bytes are preserved and labeled
`HARD_CRASH`/`INVALID`; no claim is made that the dead process wrote a canonical
exception. Caught failures alone use the canonical failure schema.

[Derivation] Marker and final commits cannot self-reference because their OIDs
are recorded only after creation in later evidence. Payload digests omit exactly
their named digest field. Test unlock is computed before the test marker and
contains no future test hash.

No open state-machine P0 from the V4 audits remains in this draft. The proper
lane recommendation is `Gate: PASS`, conditional on exact incorporation into
V5 frozen bytes and independent tests/audit. V4 remains
`BLOCKED_PRE_IMPLEMENTATION`; this draft alone is not authorization to create a
V5 implementation lock or open a registered role.
