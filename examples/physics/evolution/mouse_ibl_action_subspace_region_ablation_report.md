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
| `first_movement_speed` | `drop_top_ccf` | 11/12 | 4/11 | 0.008413 | 0.000000 | -0.005284 | `False` |
| `first_movement_speed` | `only_top_ccf` | 9/12 | 5/9 | 0.009357 | 0.002854 | -0.004340 | `False` |
| `wheel_action_direction` | `full` | 12/12 | 8/12 | 0.020350 | 0.017363 | 0.000000 | `True` |
| `wheel_action_direction` | `drop_top_ccf` | 11/12 | 10/11 | 0.018655 | 0.010498 | -0.001695 | `True` |
| `wheel_action_direction` | `only_top_ccf` | 9/12 | 4/9 | 0.014889 | 0.000601 | -0.005461 | `False` |

## verdict

- `drop_top_ccf` asks whether the action increment survives without the top anatomical block.
- `only_top_ccf` asks whether the top anatomical block is sufficient by itself.
- Speed weakens when the top anatomical block is removed: 9/12 to 4/11, mean dBA 0.013697 to 0.008413.
- Wheel remains replicated after removing the top anatomical block: 8/12 to 10/11, mean dBA 0.020350 to 0.018655.
- The top anatomical block alone is not sufficient by the current replication rule for either action target.
