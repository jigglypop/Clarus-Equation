# Mouse IBL/OpenAlyx action subspace region/probe ablation

Ablation is applied to the unit matrices before the low-rank transition and nested action-subspace readout.

## setup

- candidates: 12
- top CCF ids: `[215, 1020, 946, 128, 313]`
- top acronyms: `['APN', 'PO', 'PH', 'MRN', 'MB']`
- probe label: `probe00`

## summary

| target | condition | evaluated | supported | mean dBA | median dBA | mean delta vs full | passed |
|---|---|---:|---:|---:|---:|---:|---|
| `first_movement_speed` | `full` | 12/12 | 9/12 | 0.013697 | 0.011018 | 0.000000 | `True` |
| `first_movement_speed` | `drop_probe` | 6/12 | 3/6 | 0.009880 | 0.001420 | -0.003817 | `False` |
| `first_movement_speed` | `only_probe` | 9/12 | 7/9 | 0.008768 | 0.002841 | -0.004929 | `True` |
| `wheel_action_direction` | `full` | 12/12 | 8/12 | 0.020350 | 0.017363 | 0.000000 | `True` |
| `wheel_action_direction` | `drop_probe` | 6/12 | 4/6 | 0.007663 | 0.004311 | -0.012687 | `False` |
| `wheel_action_direction` | `only_probe` | 9/12 | 6/9 | 0.020289 | 0.008952 | -0.000061 | `False` |

## verdict

- `drop_probe` asks whether the action increment survives without the dominant probe00 block.
- `only_probe` asks whether the dominant probe00 block is sufficient by itself.
- Speed weakens without probe00: 9/12 to 3/6, mean dBA 0.013697 to 0.009880.
- Speed with only probe00 still passes the current rule: 7/9, mean dBA 0.008768.
- Wheel strongly depends on probe00 by mean effect: drop_probe lowers mean dBA from 0.020350 to 0.007663.
- Wheel with only probe00 nearly matches the full mean, 0.020289 vs 0.020350, but is 6/9 and misses the current 7-replication rule.
