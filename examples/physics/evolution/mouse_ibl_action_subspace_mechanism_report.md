# Mouse IBL/OpenAlyx action innovation-subspace mechanism map

$$
\epsilon_{S_{train}}\rightarrow\mathrm{unit/PCA\ loading\ mass}
$$

## setup

- candidates: 12
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- action mechanism map passed: `True`

## target summary

| target | candidates | supported | mean dBA | median dBA | passed |
|---|---:|---:|---:|---:|---|
| `first_movement_speed` | 12 | 9/12 | 0.013697 | 0.011018 | `True` |
| `wheel_action_direction` | 12 | 8/12 | 0.020350 | 0.017363 | `True` |

## top region loading mass

| target | region | mean mass |
|---|---|---:|
| `first_movement_speed` | `ccf_id:215` | 0.112423 |
| `first_movement_speed` | `ccf_id:1020` | 0.082453 |
| `first_movement_speed` | `ccf_id:946` | 0.075238 |
| `first_movement_speed` | `ccf_id:128` | 0.066257 |
| `first_movement_speed` | `ccf_id:313` | 0.064130 |
| `first_movement_speed` | `ccf_id:146` | 0.050320 |
| `first_movement_speed` | `ccf_id:741` | 0.046145 |
| `first_movement_speed` | `ccf_id:381` | 0.045674 |
| `first_movement_speed` | `ccf_id:1093` | 0.041932 |
| `first_movement_speed` | `ccf_id:976` | 0.028525 |
| `first_movement_speed` | `ccf_id:382` | 0.027433 |
| `first_movement_speed` | `ccf_id:502` | 0.025124 |
| `wheel_action_direction` | `ccf_id:215` | 0.108823 |
| `wheel_action_direction` | `ccf_id:1020` | 0.082451 |
| `wheel_action_direction` | `ccf_id:946` | 0.077436 |
| `wheel_action_direction` | `ccf_id:128` | 0.072745 |
| `wheel_action_direction` | `ccf_id:313` | 0.066272 |
| `wheel_action_direction` | `ccf_id:146` | 0.051341 |
| `wheel_action_direction` | `ccf_id:381` | 0.046345 |
| `wheel_action_direction` | `ccf_id:741` | 0.045814 |
| `wheel_action_direction` | `ccf_id:1093` | 0.038671 |
| `wheel_action_direction` | `ccf_id:382` | 0.027023 |
| `wheel_action_direction` | `ccf_id:976` | 0.024724 |
| `wheel_action_direction` | `ccf_id:502` | 0.024562 |

## annotated top region loading mass

Annotation source: `iblatlas.atlas.BrainRegions`.

| target | rank | CCF id | acronym | anatomical name | mean mass |
|---|---:|---:|---|---|---:|
| `first_movement_speed` | 1 | 215 | `APN` | Anterior pretectal nucleus | 0.112423 |
| `first_movement_speed` | 2 | 1020 | `PO` | Posterior complex of the thalamus | 0.082453 |
| `first_movement_speed` | 3 | 946 | `PH` | Posterior hypothalamic nucleus | 0.075238 |
| `first_movement_speed` | 4 | 128 | `MRN` | Midbrain reticular nucleus | 0.066257 |
| `first_movement_speed` | 5 | 313 | `MB` | Midbrain | 0.064130 |
| `wheel_action_direction` | 1 | 215 | `APN` | Anterior pretectal nucleus | 0.108823 |
| `wheel_action_direction` | 2 | 1020 | `PO` | Posterior complex of the thalamus | 0.082451 |
| `wheel_action_direction` | 3 | 946 | `PH` | Posterior hypothalamic nucleus | 0.077436 |
| `wheel_action_direction` | 4 | 128 | `MRN` | Midbrain reticular nucleus | 0.072745 |
| `wheel_action_direction` | 5 | 313 | `MB` | Midbrain | 0.066272 |

## probe loading mass

| target | probe | mean mass |
|---|---|---:|
| `first_movement_speed` | `probe00` | 0.634586 |
| `first_movement_speed` | `probe00a` | 0.166667 |
| `first_movement_speed` | `probe01` | 0.115414 |
| `first_movement_speed` | `probe00b` | 0.083333 |
| `wheel_action_direction` | `probe00` | 0.638429 |
| `wheel_action_direction` | `probe00a` | 0.166667 |
| `wheel_action_direction` | `probe01` | 0.111571 |
| `wheel_action_direction` | `probe00b` | 0.083333 |

## top feature loading mass

| target | feature | probe | region | mean top-feature mass |
|---|---|---|---|---:|
| `first_movement_speed` | `probe00:cluster:227` | `probe00` | `ccf_id:128` | 0.006008 |
| `first_movement_speed` | `probe00:cluster:248` | `probe00` | `ccf_id:215` | 0.003658 |
| `first_movement_speed` | `probe00:cluster:116` | `probe00` | `ccf_id:1093` | 0.003577 |
| `first_movement_speed` | `probe00:cluster:220` | `probe00` | `ccf_id:128` | 0.003471 |
| `first_movement_speed` | `probe00:cluster:21` | `probe00` | `ccf_id:313` | 0.003431 |
| `first_movement_speed` | `probe00:cluster:100` | `probe00` | `ccf_id:313` | 0.003133 |
| `first_movement_speed` | `probe00a:cluster:73` | `probe00a` | `ccf_id:741` | 0.003100 |
| `first_movement_speed` | `probe00:cluster:161` | `probe00` | `ccf_id:128` | 0.002962 |
| `first_movement_speed` | `probe00:cluster:278` | `probe00` | `ccf_id:10` | 0.002663 |
| `first_movement_speed` | `probe00:cluster:282` | `probe00` | `ccf_id:1020` | 0.002654 |
| `first_movement_speed` | `probe00:cluster:290` | `probe00` | `ccf_id:10` | 0.002644 |
| `first_movement_speed` | `probe00b:cluster:114` | `probe00b` | `ccf_id:381` | 0.002437 |
| `first_movement_speed` | `probe00a:cluster:95` | `probe00a` | `ccf_id:741` | 0.002342 |
| `first_movement_speed` | `probe00a:cluster:78` | `probe00a` | `ccf_id:741` | 0.002318 |
| `first_movement_speed` | `probe00:cluster:211` | `probe00` | `ccf_id:128` | 0.002285 |
| `first_movement_speed` | `probe00:cluster:862` | `probe00` | `ccf_id:502` | 0.002262 |
| `wheel_action_direction` | `probe00:cluster:161` | `probe00` | `ccf_id:128` | 0.003844 |
| `wheel_action_direction` | `probe00:cluster:282` | `probe00` | `ccf_id:1020` | 0.003842 |
| `wheel_action_direction` | `probe00:cluster:391` | `probe00` | `ccf_id:1020` | 0.003535 |
| `wheel_action_direction` | `probe00:cluster:274` | `probe00` | `ccf_id:146` | 0.003261 |
| `wheel_action_direction` | `probe00:other_units` | `probe00` | `OTHER_UNITS` | 0.003195 |
| `wheel_action_direction` | `probe00:cluster:217` | `probe00` | `ccf_id:128` | 0.003171 |
| `wheel_action_direction` | `probe00a:cluster:95` | `probe00a` | `ccf_id:741` | 0.003092 |
| `wheel_action_direction` | `probe00:cluster:209` | `probe00` | `ccf_id:128` | 0.002984 |
| `wheel_action_direction` | `probe00a:cluster:46` | `probe00a` | `ccf_id:414` | 0.002959 |
| `wheel_action_direction` | `probe00b:cluster:96` | `probe00b` | `ccf_id:381` | 0.002782 |
| `wheel_action_direction` | `probe00a:cluster:9` | `probe00a` | `ccf_id:946` | 0.002718 |
| `wheel_action_direction` | `probe00a:cluster:74` | `probe00a` | `ccf_id:741` | 0.002704 |
| `wheel_action_direction` | `probe00:cluster:141` | `probe00` | `ccf_id:128` | 0.002684 |
| `wheel_action_direction` | `probe00:cluster:290` | `probe00` | `ccf_id:10` | 0.002595 |
| `wheel_action_direction` | `probe00:cluster:366` | `probe00` | `ccf_id:146` | 0.002592 |
| `wheel_action_direction` | `probe00:cluster:273` | `probe00` | `ccf_id:1093` | 0.002493 |

## verdict

- Loading mass maps selected innovation axes back to target-window unit PCA loadings.
- This is a descriptive mechanism map. It does not prove causal localization.
