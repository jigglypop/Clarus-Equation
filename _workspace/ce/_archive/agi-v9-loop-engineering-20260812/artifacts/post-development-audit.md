# V9 memory post-development audit

Status: COMPLETE

Gate: STOP

Confirmation: BLOCKED

## Provenance

- preregistration SHA-256:
  `B34C22FB2ABD40D2DA5F7DE2E58C22EE1325758B033B5BD163C9C6881C655E03`
- development result SHA-256:
  `698CDB09294A28F54070EEC34C80DE07FCF9C6AC6EF3BFD5D51799EBCAC015C8`
- result-bound preregistration hash: exact match
- current source reconstruction: `PREREG_MATCH`
- seed count: `256`
- result file created once after the original long-running child completed
- confirmation result existed at audit: `False`

The wrapper yielded after its 10-minute wait limit, but the already-started original child
continued. It was monitored without a second invocation and atomically wrote the sole result
at approximately 16 minutes elapsed time.

## Registered results

| Arm | Mean accuracy |
|---|---:|
| V9 | `0.3457031250` |
| stateless | `0.2463378906` |
| level0 | `0.2822265625` |
| upper reset | `0.2822265625` |
| cross cut | `0.2822265625` |
| monolithic | `0.6115722656` |

The strongest registered comparator was `monolithic`.

| Gate | Value | Threshold | Result |
|---|---:|---:|---|
| paired mean V9 improvement | `-0.2658691406` | at least `0.02` | FAIL |
| paired 95% bootstrap interval | `[-0.2788146973,-0.2524414063]` | lower bound greater than `0` | FAIL |
| upper-reset loss | `0.0634765625` | at least `0.05` | PASS |
| cross-cut loss | `0.0634765625` | at least `0.05` | PASS |
| causal integrity | all counters zero | all zero | PASS |

Overall registered verdict: `STOP`.

## Interpretation

The nested path has a real causal contribution relative to its upper reset and cross-scale cut,
but it is not competitive with a simple same-state-count bank of explicit timescales. The
current tower's upper levels are delayed weak copies with the same within-level recurrence gain;
they are not independently slow recurrent modes. That diagnosis is post-development and may
motivate a new target-aware hypothesis, but it cannot change this result or open confirmation.

## Binding closure

- V9 task-utility superiority is rejected for this registered mechanism/task.
- The unit state-mediation result survives.
- L5 untouched confirmation is not run.
- Any level-dependent recurrence or new readout is a new model and requires a fresh
  preregistration and new seed roles; it is not a repair of this score.
