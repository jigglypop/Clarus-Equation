# C. elegans trial-behavior boundary audit

This audit separates the supported connectome proxy from an empirical trial-behavior gate.

## verdict

- empirical trial gate ready: `False`
- verdict: `data_boundary`
- proxy supported: `True`

## proxy evidence

| item | value |
|---|---:|
| adult matched/wrong | 3.431872 |
| adult permutation p | 0.034393 |
| developmental weighted stages | 8/8 |
| developmental mean matched/wrong | 3.213504 |

## local trial data scan

| required field | found |
|---|---|
| `trial_id` | `False` |
| `stimulus_label_or_time` | `False` |
| `behavior_label_or_trace` | `False` |
| `worm_id` | `False` |
| `timebase` | `False` |

## candidate local files

- none

## interpretation

- The current C. elegans behavior result remains a weighted-connectome proxy.
- It should not be promoted to an empirical trial-behavior equation without stimulus labels, behavior traces/labels, worm or trial ids, and a timebase.
- The next real closure requires a trial-level C. elegans stimulus-behavior dataset or a time-aligned neural/behavior recording.
