# Implementation

Status: COMPLETE

## 1. Artifact

The frozen witness is `artifacts/a7_discrete_hybrid_witness.py`. It is a
self-contained synthetic apparatus with two deliberately separated layers:

1. a NumPy float64 mirror of the exact fixed Torch branch, including
   activation, refractory state, memory trace, adaptation, STP facilitation
   and resource, two delay slots, old-activation ring overwrite, bit update,
   global eligibility threshold, and budgeted TopK selection;
2. isolated float32 `BrainRuntime._step_torch` and `_step_rust` calls that bind
   the mirror to the frozen implementation without opening learning, recall,
   auto-mode, F1, or empirical paths.

The implementation propagates the full 24-dimensional branch directional
derivative analytically. The ordinary interior Jacobian is assembled from
basis directions and compared with immutable-state central differences. At
clip faces the same propagation uses the one-sided rule; bit, selection, and
lifecycle crossings return only discrete receipts.

## 2. Implemented gates

- H-A: independent Torch/mirror one-step state, recurrent, salience, bit,
  selection mask, buffer and counter;
- H-B: all-block 24 by 24 branch Jacobian;
- H-C: runtime-reachable lower/upper clip faces and separately labeled scalar
  diagnostics for unreachable upper faces;
- H-D: bit, eligibility, kth-boundary, exact-tie, and lifecycle guard receipts;
- H-E: exact three-call ring timing and the one-tick previous-lifecycle lag;
- H-F: neuron-permutation equivariance in float64 and float32, outside ties;
- H-G: no-delay Torch/Rust positive control plus the preregistered delay-on
  semantic mismatch.

## 3. Preserved apparatus revisions

No equation, fixture, direction, step, tolerance, or decision gate changed
after witness execution.

- `revisions/00-apparatus-path-failure/` preserves the first script and
  traceback receipt. A wrong provenance-only kernel path stopped execution
  before H-A; only two path literals were corrected.
- `revisions/01-version-receipt/` preserves the first passing script/result.
  Distribution metadata was unavailable, so the final apparatus adds the
  imported `reality_stone.__version__` fallback and records its source. All
  numerical quantities reproduced exactly.

## 4. Final hashes

- witness SHA-256:
  `9021d90352933d903b7af6c716d47d27773792978060805d931994de6ad8fad8`
- result SHA-256:
  `cc6954b7f8120fb231494a4cc5cc0498895270ff1a93dfa834027f3fb22c9e6a`
- contract SHA-256:
  `86f72a7f188e63f03edb19efbe9eb67613b254f3aa0db4214953e684658ac1fe`
- pre-implementation audit SHA-256:
  `e74f8875b10da49a2b8eaba48b72ce56fb76d365be1530e5a970566d28748861`

No runtime source file was modified by this run.
