# Pre-implementation audit

Status: COMPLETE

Gate: PASS

## 1. Stable snapshot audited

The independent read-only audit examined `00-contract.md`, `10-sources.md`,
`11-math.md`, and `12-routes.md` against the frozen `runtime.py`, `kernel.rs`,
and `tests/test_runtime_contracts.py` sources. No runtime response or empirical
asset was opened and no implementation witness had been executed.

## 2. P0 correction retained

The actual runtime is a ticked discrete hybrid map. It has no continuously
resolved event time, vector fields on both sides of a guard, or transversality
denominator. Continuous-time saltation is therefore not an admissible object.
The frozen replacement is:

1. ordinary Jacobian on a strict fixed discrete branch;
2. one-sided/Bouligand directional derivative at a continuous clip face;
3. transition receipt, with no finite derivative, when bit, TopK, or lifecycle
   state changes.

## 3. Revision-2 findings closed before execution

The first audit returned `REVISE` for reproducibility rather than for a new
mathematical contradiction. Revision 2 closed every item before implementation:

- `max_axon_delay=2` is explicit and the zero-based ring table distinguishes
  the unbounded runtime counter from its modulo slot, read-before-write, old
  activation, and the third-call arrival;
- the ring equation is assignment, not additive update;
- actual selection uses global `active_threshold=.22`, WAKE budget 2, and
  `force_all_active_selection=False`;
- lifecycle `(ACTIVE,DORMANT,ACTIVE)` is frozen and is the source of
  `q_prev=(1,0,1)`;
- float64 immutable-state derivative tests are separated from float32
  Torch/Rust comparisons;
- runtime-reachable clip faces are separated from scalar primitive
  domain-extension diagnostics;
- mirror and runtime permutation tolerances are distinct and
  `dale_law=False` is frozen;
- Rust import availability and extension/backend selection must be receipted.

The second independent audit returned `PASS`: no P0 or P1 remained in the
stable pre-implementation snapshot.

## 4. Backend boundary confirmed from source

The Torch delay branch accepts and mutates a ring buffer/index. The Rust kernel
signature accepts neither. Rust therefore applies current activation while
Torch applies the selected delayed slot when delay is enabled. The existing
parity test explicitly disables axon delay. Accordingly:

- no-delay/no-tie cell-state parity is a positive control;
- delay-on mismatch is a preregistered expected failure and yields
  `RUNTIME_DELAY_PARITY_BLOCKED`;
- Rust's private active-count return is excluded from the direct-cell parity
  gate because outer `step()` recomputes Python TopK selection, and the Rust
  wrapper's internal selection threshold is hard-coded at `.22`.

## 5. Authorization

Implementation is authorized only for the frozen synthetic witness. It may not
change source runtime code, fixture values, steps, tolerances, guard directions,
or the claim ceiling. A formula mismatch permits the single revision path in
the contract with the failed artifact retained. No brain, AGI, anatomy, or
learning conclusion is authorized.
