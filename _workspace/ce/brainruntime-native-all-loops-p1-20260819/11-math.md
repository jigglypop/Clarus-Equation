# Mathematical verification -- P1 integration correction

Status: COMPLETE

## Concrete findings

- **P1 (required revision):** The predecessor's causal-orientation test proves only
  `EligibilityTracker.eligibility[post, pre] > 0`.  It does not prove the
  *applied* `BrainRuntime.weight` increment after `apply_stdp_update` and its
  structural projection.  Generic positive Frobenius drift is not a substitute:
  a drift can occur entirely in the opposite entry or through row normalization.
- **P1 (required revision):** The predecessor Loop 8 records temporal events but
  iterates `zip(cues, targets)` independently for replay.  Therefore no fact in
  its result proves that a latest-valid temporal row selected the replayed pair.
- **P1 (required revision):** Loop 7 has a unit test for context precedence, but
  its executable Loop-7 aggregate has no conflicting supplied-context case or
  context-branch read-count metric.

## Applied causal-weight invariant

With the runtime convention `a[t+1] ~ W @ a[t]`, row `i` is receiver/post and
column `j` is sender/pre.  For distinct coordinates `j` (cue/pre) and `i`
(target/post), causal eligibility after the two isolated spikes is

$$e_{ij}=A_+ r_+>0,\qquad e_{ji}\ne e_{ij}.$$

The focused **runtime** test must use the causal opt-in, positive external
learning signal, `stdp_apply_interval=1`, no noise/delay/Dale transform, and a
weight/setup for which projection cannot erase the tested entry.  It must retain
the pre-apply matrix `W0`, execute pre then post through `BrainRuntime.step`, and
assert, on the applied matrix,

$$\Delta W_{ij}=W^{+}_{ij}-W^{0}_{ij}>0,$$

with `i != j`, finite `W+`, and a non-positive-or-smaller reverse increment
reported as a diagnostic.  The test must choose signal scale / initial row norm
so the projected entry remains above `theta_on=0.01`; otherwise the exact
counterexample is `0 < lr*g*A_+r_+ < theta_off`, which projection maps to zero.
It must additionally instantiate the default configuration and assert its
orientation remains `legacy` (and preserve the existing legacy asymmetric
eligibility result).  This is the minimum proof of the contract's
`Delta W[post,pre] > 0`, not a claim about recall success.

Numerical spot check from the present definitions (`A+=0.01`, `r+=0.95`) gives
`e[1,0]=+0.0095` and `e[0,1]=-0.0114` after pre `0`, post `1`.  Starting from
zero weights, `lr=0.001, gate=1` projects the causal entry to `0.0`, whereas
`gate=2000` leaves it as `1.0` after row normalization.  Thus a raw eligibility
assertion, or an under-threshold applied test, would be a false positive for the
contract's applied-weight claim.

## Latest-valid temporal-to-replay invariants

Let each temporal key be `k=(subject, "target")`, and let
`v(e)=(valid_session, sequence, evidence_id)`.  For every candidate key, replay
may consume an episode iff

$$e^*(k)=\max_{v} E_k,\quad e^*(k).operation=UPSERT,\quad e^*(k).value\ne\varnothing.$$

The replay-source builder must obtain its selected value/evidence from
`TemporalAuditedMemory.recall`, resolve the fixed codebook episode from that
selected value, and return an audit list containing at least key, selected
value, evidence id, valid session, and whether it was replayed.  Its replayed
IDs/values must equal the latest-valid recall manifest exactly; a DELETE recall
must be abstained and absent from both the replay list and pair loop.

Use a fixture with (a) a newer UPSERT that arrives before a stale older UPSERT,
and (b) a DELETE that arrives before a stale UPSERT for a different key.  The
same event multiset ingested in forward and reversed arrival order must yield
identical selected audit/replay lists.  A deliberately arrival-last selector
must disagree with the valid-time manifest (it selects the stale value and/or
resurrects the deleted key).  Capacity and `max_versions_per_key` must exceed
the fixture history, or compaction becomes a confound.  Audit equality must be
checked before stores are detached; cutoff alone cannot recover provenance.

## Loop 7 supplied-context precedence

The aggregate fixture must issue an enabled `fact` query containing a supplied
context value/evidence that conflicts with the stored value.  Let `R0,R1` be
the temporal `recall_count` immediately before/after that call.  Report

$$I_{context}=1[route=context\ \land\ value=context\_value\ \land\ evidence=context\_evidence],
\qquad reads_{context}=R1-R0.$$

Require `context_precedence_accuracy=I_context=1.0` and
`context_temporal_reads=0`; include both in the Loop-7 GO predicate alongside
the preregistered route and disabled-mode measures.  A matching stored/context
value is insufficient because it cannot distinguish precedence from a memory
read.

## Leakage and boundary checks

- The replay audit may contain source provenance during encoding only.  After
  `_detach`, no temporal/hippocampal row or selected target may enter cue-only
  rollout or decoding.
- Arrival-order ablation is a negative control, not a selection rule eligible
  for native results.
- These corrections establish wiring and direction only; they do not alter the
  predecessor's Loop 8/9 STOP verdicts or authorize confirmation seeds.

## Reproduction targets

- Focused: `tests/test_stdp.py`, `tests/test_runtime_native_loops.py`, and
  `tests/test_runtime_temporal_memory.py` with cache disabled and a temporary
  base directory outside the repository.
- Source inspected: `stdp.py`, `runtime.py`, `temporal_memory.py`,
  `runtime_temporal_memory.py`, and `runtime_native_loops.py`.
