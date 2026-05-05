# Mouse IBL/OpenAlyx action top-unit sufficiency

Top units are selected inside each outer fold from train-fitted loading mass.

## setup

- candidates: 12
- probe label: `probe00`
- top units per fold: 16

## summary

| target | condition | evaluated | supported | mean dBA | median dBA | mean delta vs full | passed |
|---|---|---:|---:|---:|---:|---:|---|
| `first_movement_speed` | `full` | 12/12 | 9/12 | 0.013697 | 0.011018 | 0.000000 | `True` |
| `first_movement_speed` | `drop_top_units` | 12/12 | 6/12 | 0.005858 | 0.002820 | -0.007839 | `False` |
| `first_movement_speed` | `only_top_units` | 9/12 | 6/9 | 0.008726 | 0.004545 | -0.004971 | `False` |
| `wheel_action_direction` | `full` | 12/12 | 8/12 | 0.020350 | 0.017363 | 0.000000 | `True` |
| `wheel_action_direction` | `drop_top_units` | 12/12 | 6/12 | 0.007747 | 0.001398 | -0.012603 | `False` |
| `wheel_action_direction` | `only_top_units` | 9/12 | 7/9 | 0.023757 | 0.008952 | 0.003407 | `True` |

## verdict

- `only_top_units` asks whether fold-local top probe units are sufficient.
- `drop_top_units` asks whether the remaining unit ensemble can compensate.
- Speed weakens when fold-local top probe00 units are removed: 9/12 to 6/12, mean dBA 0.013697 to 0.005858.
- Speed with only top probe00 units is 6/9, mean dBA 0.008726, so it misses the current 7-replication rule.
- Wheel weakens strongly when fold-local top probe00 units are removed: mean dBA 0.020350 to 0.007747.
- Wheel with only top probe00 units passes: 7/9, mean dBA 0.023757.
