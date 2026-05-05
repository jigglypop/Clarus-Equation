# Mouse IBL/OpenAlyx action innovation-subspace mechanism map

$$
\epsilon_{S_{train}}\rightarrow\mathrm{unit/PCA\ loading\ mass}
$$

## setup

- candidates: 3
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- action mechanism map passed: `False`

## target summary

| target | candidates | supported | mean dBA | median dBA | passed |
|---|---:|---:|---:|---:|---|
| `first_movement_speed` | 3 | 2/3 | 0.003416 | 0.008523 | `False` |
| `wheel_action_direction` | 3 | 3/3 | 0.033414 | 0.033859 | `False` |

## top region loading mass

| target | region | mean mass |
|---|---|---:|
| `first_movement_speed` | `ccf_id:215` | 0.198510 |
| `first_movement_speed` | `ccf_id:382` | 0.105206 |
| `first_movement_speed` | `ccf_id:632` | 0.076437 |
| `first_movement_speed` | `ccf_id:502` | 0.071454 |
| `first_movement_speed` | `ccf_id:10703` | 0.068451 |
| `first_movement_speed` | `ccf_id:463` | 0.067300 |
| `first_movement_speed` | `ccf_id:843` | 0.057168 |
| `first_movement_speed` | `ccf_id:621` | 0.047713 |
| `first_movement_speed` | `ccf_id:128` | 0.043963 |
| `first_movement_speed` | `ccf_id:658` | 0.025354 |
| `first_movement_speed` | `ccf_id:1089` | 0.019503 |
| `first_movement_speed` | `ccf_id:771` | 0.018470 |
| `wheel_action_direction` | `ccf_id:215` | 0.191914 |
| `wheel_action_direction` | `ccf_id:382` | 0.104281 |
| `wheel_action_direction` | `ccf_id:632` | 0.077893 |
| `wheel_action_direction` | `ccf_id:463` | 0.068390 |
| `wheel_action_direction` | `ccf_id:10703` | 0.067587 |
| `wheel_action_direction` | `ccf_id:502` | 0.065598 |
| `wheel_action_direction` | `ccf_id:621` | 0.060893 |
| `wheel_action_direction` | `ccf_id:843` | 0.053611 |
| `wheel_action_direction` | `ccf_id:128` | 0.053485 |
| `wheel_action_direction` | `ccf_id:771` | 0.020916 |
| `wheel_action_direction` | `ccf_id:114` | 0.018709 |
| `wheel_action_direction` | `ccf_id:1089` | 0.018309 |

## probe loading mass

| target | probe | mean mass |
|---|---|---:|
| `first_movement_speed` | `probe00` | 0.538344 |
| `first_movement_speed` | `probe01` | 0.461656 |
| `wheel_action_direction` | `probe00` | 0.553717 |
| `wheel_action_direction` | `probe01` | 0.446283 |

## top feature loading mass

| target | feature | probe | region | mean top-feature mass |
|---|---|---|---|---:|
| `first_movement_speed` | `probe00:cluster:471` | `probe00` | `ccf_id:658` | 0.004325 |
| `first_movement_speed` | `probe01:other_units` | `probe01` | `OTHER_UNITS` | 0.004094 |
| `first_movement_speed` | `probe00:other_units` | `probe00` | `OTHER_UNITS` | 0.003780 |
| `first_movement_speed` | `probe00:cluster:273` | `probe00` | `ccf_id:621` | 0.003629 |
| `first_movement_speed` | `probe00:cluster:454` | `probe00` | `ccf_id:632` | 0.003435 |
| `first_movement_speed` | `probe00:cluster:101` | `probe00` | `ccf_id:10703` | 0.002910 |
| `first_movement_speed` | `probe00:cluster:36` | `probe00` | `ccf_id:215` | 0.002834 |
| `first_movement_speed` | `probe00:cluster:284` | `probe00` | `ccf_id:463` | 0.002813 |
| `first_movement_speed` | `probe00:cluster:322` | `probe00` | `ccf_id:534` | 0.002784 |
| `first_movement_speed` | `probe00:cluster:145` | `probe00` | `ccf_id:621` | 0.002722 |
| `first_movement_speed` | `probe00:cluster:162` | `probe00` | `ccf_id:632` | 0.002637 |
| `first_movement_speed` | `probe01:cluster:228` | `probe01` | `ccf_id:382` | 0.002633 |
| `first_movement_speed` | `probe01:cluster:44` | `probe01` | `ccf_id:10703` | 0.002549 |
| `first_movement_speed` | `probe01:cluster:233` | `probe01` | `ccf_id:382` | 0.002517 |
| `first_movement_speed` | `probe00:cluster:102` | `probe00` | `ccf_id:215` | 0.002425 |
| `first_movement_speed` | `probe00:cluster:661` | `probe00` | `ccf_id:843` | 0.002424 |
| `wheel_action_direction` | `probe00:cluster:180` | `probe00` | `ccf_id:621` | 0.007241 |
| `wheel_action_direction` | `probe00:cluster:273` | `probe00` | `ccf_id:621` | 0.005976 |
| `wheel_action_direction` | `probe00:cluster:391` | `probe00` | `ccf_id:632` | 0.005671 |
| `wheel_action_direction` | `probe00:cluster:10` | `probe00` | `ccf_id:114` | 0.005613 |
| `wheel_action_direction` | `probe00:cluster:5` | `probe00` | `ccf_id:114` | 0.005212 |
| `wheel_action_direction` | `probe00:cluster:143` | `probe00` | `ccf_id:621` | 0.005038 |
| `wheel_action_direction` | `probe00:cluster:217` | `probe00` | `ccf_id:621` | 0.005020 |
| `wheel_action_direction` | `probe00:cluster:168` | `probe00` | `ccf_id:621` | 0.005013 |
| `wheel_action_direction` | `probe00:cluster:169` | `probe00` | `ccf_id:621` | 0.004952 |
| `wheel_action_direction` | `probe00:cluster:275` | `probe00` | `ccf_id:463` | 0.004929 |
| `wheel_action_direction` | `probe00:cluster:170` | `probe00` | `ccf_id:621` | 0.004663 |
| `wheel_action_direction` | `probe00:other_units` | `probe00` | `OTHER_UNITS` | 0.004572 |
| `wheel_action_direction` | `probe00:cluster:284` | `probe00` | `ccf_id:463` | 0.004531 |
| `wheel_action_direction` | `probe00:cluster:722` | `probe00` | `ccf_id:843` | 0.004212 |
| `wheel_action_direction` | `probe01:cluster:119` | `probe01` | `ccf_id:128` | 0.004036 |
| `wheel_action_direction` | `probe00:cluster:154` | `probe00` | `ccf_id:10703` | 0.004025 |

## verdict

- Loading mass maps selected innovation axes back to target-window unit PCA loadings.
- This is a descriptive mechanism map. It does not prove causal localization.
