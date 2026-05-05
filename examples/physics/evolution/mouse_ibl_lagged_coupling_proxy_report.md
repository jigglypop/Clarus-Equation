# Mouse IBL/OpenAlyx lagged region-coupling proxy gate

Same-window interaction이 실패했기 때문에 source window와 target window를 분리한 lagged coupling proxy를 검사했다.
이 feature는 causal connectivity가 아니라 temporal ordering을 보존한 region-rate product다.

## setup

- candidates: 5
- folds: 5
- ridge: 1.0
- permutations: 200
- max units per probe: 96
- max lagged pairs: 750
- lagged supported over region: `False`
- lagged beats top unit: `False`
- lagged coupling proxy passed: `False`

## model equation

$$
z_{iab}^{\mathrm{lag}}=r_{ia}^{\mathrm{source}}r_{ib}^{\mathrm{target}}.
$$

The tested model is

$$
R_i^{\mathrm{lag}}=[R_i^{\mathrm{target}},Z_i^{\mathrm{lag}}].
$$

Source windows are pre-stimulus for choice/speed and pre-movement for wheel direction.

## target replication

| target | candidates | lagged beats region | lagged beats unit | task+lagged beats task+region | task+lagged beats task+unit | mean source BA | mean region BA | mean lagged BA | mean unit BA | mean lagged-region delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 5 | 0 | 0 | 0 | 0 | 0.531141 | 0.675560 | 0.637484 | 0.744883 | -0.038076 |
| `first_movement_speed` | 5 | 1 | 1 | 0 | 1 | 0.619563 | 0.742104 | 0.706063 | 0.756299 | -0.036041 |
| `wheel_action_direction` | 5 | 0 | 0 | 0 | 0 | 0.534773 | 0.716240 | 0.674517 | 0.777766 | -0.041723 |

## candidate summaries

| candidate | trials | lagged>region | lagged>unit | task+lagged>task+region | task+lagged>task+unit | choice lag-region | speed lag-region | wheel lag-region |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 0 | 0 | 0 | 0 | -0.061322 | -0.079969 | -0.077407 |
| `nyu30_motor_striatal_multi_probe` | 933 | 0 | 0 | 0 | 0 | -0.046398 | -0.004770 | -0.041413 |
| `dy014_striatal_septal_probe` | 608 | 0 | 0 | 0 | 0 | -0.018035 | -0.023810 | -0.023171 |
| `dy011_motor_cortex_probe` | 402 | 1 | 1 | 0 | 1 | -0.028084 | 0.008091 | -0.049638 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 0 | 0 | 0 | 0 | -0.036539 | -0.079746 | -0.016987 |

## verdict

- lagged supported over region: `False`
- lagged beats top unit: `False`
- lagged coupling proxy passed: `False`

해석:

- Lagged coupling이 additive region보다 반복적으로 높으면 same-window product 실패가 시간 방향성 문제였다는 뜻이다.
- Lagged coupling이 top-unit을 이기지 못하면 unit-detail residual은 계속 남는다.
- 이 gate도 causal connectivity가 아니다. 더 강한 버전은 trial-split lag selection, all-unit nested regularization, 혹은 GLM coupling이다.
