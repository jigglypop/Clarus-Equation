# BA-TR20 validation

Focused and adjacent validation after the alias repair:

```text
.codex\hooks\python.cmd pytest tests\test_runtime_delay_events.py \
  tests\test_runtime_endogenous_competition_homeostasis.py \
  tests\test_runtime_adaptive_competition_composition.py \
  tests\test_runtime_factorized_competition_composition.py \
  tests\test_runtime_factorized_composition_consistent.py \
  tests\test_runtime_factorized_one_shot_event_composition.py \
  -q -p no:cacheprovider
17 passed in 6.67s
```

R2 calibration seed `107003` passed: atomic 4/4, factorized pairs 4/4,
independent one-shot union 4/4, stream 4/4, legacy WTA 0/4,
misaligned provenance 0/4, and suppressed source event 0/4.

Fresh development seeds `107101..107116` produced:

- row gate: 16/16;
- atomic one-shot recall: 64/64;
- source-factorized one-shot pair recall: 64/64;
- independent one-shot union: 64/64;
- persistent-stream positive control: 64/64;
- legacy global WTA: 0/64;
- one-tick-misaligned source receipt: 0/64;
- fully suppressed source event: 0/64;
- first-arrival hidden positive count: exactly two for every pair;
- delivered source packet receipt: `[0,0,0,2,0,0,0]` for every pair;
- source ring-write receipt: `[0,2,0,0,0,0,0]` for every pair;
- correct target activation: `6.57624623272568e-5` to
  `1.70997343957424e-4`; wrong target activation: exactly zero.

The experiment uses a common `1e-5` component threshold for both singleton
and pair target sets. Temporal and hippocampal stores were cut before every
probe and the gates remained zero. Confirmation stays sealed.

