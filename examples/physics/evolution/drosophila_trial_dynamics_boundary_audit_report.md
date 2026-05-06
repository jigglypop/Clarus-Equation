# Drosophila trial-dynamics boundary audit

This audit separates adult FlyWire structural closure from empirical trial dynamics.

## verdict

- empirical trial dynamics ready: `False`
- verdict: `data_boundary`
- structural closed: `True`

## structural evidence

| item | value |
|---|---:|
| larva memory-loop touched fraction | 0.325455 |
| larva memory internal / boundary | 1.016627 |
| adult observed/random memory-action loop | 3.738545 |
| adult loop p | 0.012987 |

## required trial-dynamics fields

| required field | found |
|---|---|
| `trial_id` | `False` |
| `timebase` | `False` |
| `neural_activity_or_spikes` | `False` |
| `behavior_label_or_trace` | `False` |
| `stimulus_or_task_epoch` | `False` |
| `celltype_or_region_mapping` | `False` |

## candidate local files

- none

## interpretation

- Drosophila remains closed at the structural connectome level: celltype/action/memory co-differentiation plus a memory/action loop.
- Trial dynamics are not falsified; they are not testable from the local files.
- Promotion to a temporal behavior equation requires time-aligned neural activity or spikes, behavior traces, task/stimulus epochs, and celltype/region mapping.
