# G2 contract and implementation audit

Status: COMPLETE

Gate: PASS for G2 v4 implementation and fresh development execution. No result is implied.

Gate: REVISE until independent mathematics and adversarial audits accept the exact G2 contract.

The initial route wording, “metric sufficiency,” was rejected before implementation. Since
`g=(BB^T+lambda I)^-1`, `g` cannot contain information absent from raw `B/C`; and the map loses the
sign/right-orientation information needed for signed endpoint prediction. The replacement contract
tests only incremental utility of a precommitted quadratic metric feature for a scalar nonmetric
path outcome, with raw `B/C`, Euclidean, permuted metric, alternative SPD, and stronger nonlinear
direct baselines.

The first audit returned REVISE. Version 2 removes the unsupported fully-active guarantee, gates the
actual 48-bit mask at every tick, geometrically separates mixed directions from calibration axes,
renames `Q` as raw endpoint precision, rebuilds transformed tensors, makes the no-repackaging test
non-aliasing, adds a 12-coefficient `C6` comparator, replaces endpoint-only raw `B` with horizon-
matched `B_h` path prediction, assigns disjoint deterministic native-noise keys, and replaces ten
separate intervals with one conservative per-seed worst-adversary percentile bootstrap.

The v2 re-audit found no new structural math problem but required final precision repairs. Version 3
uses a collision-free stride-eight noise schedule, requires a separate G2 fixture/non-alias test,
renames the ambiguous future-looking feature symbols to present state `u:=y0`, narrows `Cterms` to a
termwise scalar decomposition, adds a separate six-entry raw-C head, and specifies the one-sided
empirical-quantile bootstrap exactly.

Two independent final audits returned PASS with no remaining P0/P1 blocker. They verified the
present-state feature provenance, raw-C scope, horizon-matched direct `B_h` baseline, collision-free
native-noise intervals, dedicated G2 fixture, covariant rebuild, non-aliased inverse duplicate,
coefficient ledger, and per-seed worst-adversary bootstrap. Implementation is authorized only under
the compressed-feature claim in `01-g2-contract.md`.

## Executable apparatus smoke: REVISE

Before the 16-seed development range, one seed-level smoke run confirmed the exact issue anticipated
by the prior audit: threshold zero did not force a full mask. In noise environments, refractory
made isolated coordinates ineligible on later free ticks (47/48), so the frozen eligibility gate
failed before model comparison. This is an apparatus failure, not an empirical G2 STOP.

Version 4 therefore proposes a default-false `force_all_active_selection` runtime switch and freezes
it true only in the G2 fixture. Default behavior must remain unchanged and snapshot parity plus
48/48 per-tick masks remain mandatory. Independent audit is required before runtime implementation
and development execution.

The adversarial v4 audit additionally quarantined the exposed seed-97501 smoke. Its pre-v4 source
hash is recorded in the contract; the full old `97501..97516` block is retired in favor of fresh
development `97601..97616` and confirmation `99601..99632`. The comparator/gate contract is
unchanged. Direct all-negative, legacy-bit-parity, config-hash, and snapshot tests are now explicit.

Both final v4 audits accepted the default-off/runtime-opt-in intervention after quarantine. The
focused G2 test file passed 5/5, including legacy bit parity, all-negative forced selection,
snapshot round-trip, dedicated fixture separation, disjoint noise intervals, all-tick 48/48 masks,
coefficient ledger, covariant feature, and non-repackaging identity. The adjacent pre-existing
runtime snapshot-continuity test also passed. Fresh development `97601..97616` is authorized;
confirmation remains sealed.
