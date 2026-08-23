# Implementation

Status: IN_PROGRESS

## Outcome

Loops 6--10 now have an opt-in executable harness over the real PyTorch `BrainRuntime`. The
default runtime remains on the legacy STDP orientation and continues automatic hippocampal
encoding unless an experiment explicitly selects the new controls.

## Runtime changes

- `STDPConfig.orientation` and `BrainRuntimeConfig.stdp_orientation` add a default-preserving
  `legacy` / opt-in `causal` choice.
- Causal orientation uses row-as-post, column-as-pre eligibility consistent with `W @ pre`.
- `BrainRuntimeConfig.hippocampal_encoding_enabled` defaults to `True`; native cutoff explicitly
  sets it to `False`.
- `BrainRuntime.reset_evaluation_state()` clears transient dynamics without changing weights or
  stores.
- `BrainRuntime.install_bounded_recurrent_delta()` is an opt-in finite, shape-checked,
  Frobenius-bounded recurrent write used only by separately labeled Route B.

## Native-loop harness

`runtime_native_loops.py` implements:

- Loop 6 valid-time selection and temporal-order ablation;
- Loop 7 selective `RuntimeTemporalAgent` routing and disabled-read audit;
- Loop 8 Route A causal STDP replay, physical double-store cutoff, matched controls, fixed
  codebook, snapshot parity, and cue plus six zero-input runtime steps;
- Loop 9 factorized held-out `do(B=1)` intervention under Route A;
- Loop 10 deterministic frozen ridge self-prediction from pre-transition native observables and
  committed action, with persistence and OOD error-monitoring controls;
- separately labeled Route B bounded supervised low-rank recurrent writes for Loops 8 and 9.

## Files

- `reality_stone/python/reality_stone/clarus/stdp.py`
- `reality_stone/python/reality_stone/clarus/runtime.py`
- `reality_stone/python/reality_stone/clarus/runtime_native_loops.py`
- `reality_stone/python/reality_stone/clarus/runtime_native_loops_benchmark.py`
- `reality_stone/python/reality_stone/clarus/__init__.py`
- `tests/test_stdp.py`
- `tests/test_runtime_native_loops.py`

Route A STOP results were frozen before Route B was added. Route B does not alter or relabel
Route A outcomes.

## Stable-snapshot audit gaps

The current implementation is executable but not contract-complete:

- the asymmetric test proves eligibility orientation but not the applied
  $\Delta W_{\mathrm{post},\mathrm{pre}}>0$ invariant;
- Loop 6 latest-valid selection is evaluated separately but is not yet the actual episode source
  consumed by Loop 8 replay;
- Loop 7's harness does not yet include the supplied-context-precedence case already covered by
  the older focused temporal wrapper test;
- confirmation remains unopened.
