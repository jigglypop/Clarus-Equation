# M0/M1 implementation record

Status: COMPLETE

## Isolation boundary

The predecessor runtime and its frozen native-loop benchmark were not changed. The alternative
mechanisms live in:

- `reality_stone/python/reality_stone/clarus/runtime_alternative_memory.py`;
- `reality_stone/python/reality_stone/clarus/runtime_alternative_memory_benchmark.py`;
- `tests/test_runtime_alternative_memory.py`.

Both routes use `BrainRuntime.step`, mutate the actual dense recurrent matrix through the bounded
native install boundary, rebuild sparse CSR state, physically clear temporal and hippocampal stores,
and decode only a sealed cue-plus-zero-input rollout.

## M0

M0 independently resets native dynamics to obtain cue and target states, constructs the declared
supervised target matrix, performs rank-1/2/4/full SVD truncation, and compares no-write,
target-shuffled, identical-spectrum random low-rank, and norm-matched cue-only controls. Dale and
structural projection are disabled. This is labelled only as a capacity ceiling.

## M1

M1 implements a signed row-post/column-pre eligibility object external to the runtime. The object
observes local native activations but never owns weight. Each block:

1. presents the cue through a native step;
2. resets all native transient state before the value phase;
3. preserves only the external eligibility trace and, for forward arms, the staged replay row;
4. presents three NREM replay ticks;
5. applies one target-blind `+1.0` clock pulse at block end;
6. structurally projects once and installs one bounded delta in the native recurrent matrix.

Zero, sign, time reversal, eligibility reset, no replay, and target shuffle execute equal numbers of
events, ticks, and pulses. Time reversal removes its staged row before cue delivery so target recall
cannot leak into the cue phase. The applied association contrast is computed from actual final minus
initial runtime weight.

## Development revisions retained

- **v1:** fixed-clock and eligibility-reset both recalled perfectly. The route correctly returned
  STOP because native activation residue was an alternate phase bridge.
- **Revision 1/2:** added an equal native transient reset between phases. M1 then passed 15/16; one
  deleted cue scored `0.170074` above the predecessor threshold `0.15`.
- **Revision 2/2:** froze an M1-only threshold `0.20`. The minimum known score was `0.529764`, leaving
  a strict development gap. No confirmation seed had been opened.

No further M1 revision is authorized.

## Focused validation

The final source passed:

```text
python -B -m pytest tests/test_runtime_alternative_memory.py \
  tests/test_runtime_native_loops.py tests/test_stdp.py \
  -q -p no:cacheprovider --basetemp C:\tmp\clarus-alt-memory-adjacent
```

Result: `28 passed` in `5.16s`. The only output was PyTorch's existing sparse-CSR beta/invariant
warning. `git diff --check` also passed.

