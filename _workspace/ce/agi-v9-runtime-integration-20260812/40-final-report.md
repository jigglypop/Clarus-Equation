# V9 runtime integration final report

Status: COMPLETE

## Outcome

V9 is now connected to a real executable `RuntimeAgent` path. With one explicit flag, a finite
observation is converted to bounded dimensionless action evidence, recurrent tower state is
updated, a sealed token is issued, and the action is selected only from the token policy.

The default path is unchanged. This closes the missing runtime wiring but does not close the
scientific efficacy question and does not constitute AGI.

## What changed

- Added opt-in `nested_scc_enabled` control and public V9 package exports.
- Added causal observation→tower→token→policy→action wiring.
- Added exact action masks and forbidden ambiguous belief/V9 composition.
- Exposed evidence/token/policy for inspection without exposing private tower arrays.
- Removed two dormant configuration degrees of freedom.
- Renamed misleading parameter-count metadata to a serialized-scalar metadata field.
- Added focused causal, history, failure, legacy-compatibility, and demo coverage.
- Updated the V9 canonical document and CodeMap without opening any evidence block.

## Verification summary

- Related warning-as-error suite: `210 passed`.
- Dimensionless: `10 passed`; checker exit 0.
- Ruff/format/compile/demo: pass.
- Full suite: `2145 passed`, with unrelated missing-fixture and policy-mirror failures preserved
  and reported rather than hidden.

## Scientific status

| Claim | Status |
|---|---|
| finite nested-SCC mathematics | inherited conditional theorems survive |
| sealed finite controller | implemented |
| RuntimeAgent state-mediated V9 path | implemented and unit validated |
| same-current-input history sensitivity | demonstrated in a deterministic unit fixture |
| task utility / matched-control advantage | untested |
| 256-seed development | `0/256 BLOCKED` |
| AGI / whole-brain identity | untested and not claimed |

## Next authorized research step

The next step is not more architecture tuning. It is a separate preregistered small task where
V9, legacy, reset, cross-cut, and matched recurrent controls receive identical raw inputs and
compute accounting. Until that gate is written and audited, no development seeds should run.
