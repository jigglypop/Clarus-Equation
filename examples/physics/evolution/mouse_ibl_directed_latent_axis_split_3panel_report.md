# Mouse IBL/OpenAlyx directed latent-axis split gate

$$
y_{action}=g_a(X_t,R_t,\hat H_t,\epsilon_{t,k})
$$

## setup

- candidates: 3
- components: 12
- folds: 5
- directed axis gate passed: `False`

## target summary

| target | candidates | best axis supported | mean best dBA | median best dBA | mean positive axes | supported |
|---|---:|---:|---:|---:|---:|---|
| `choice_sign` | 3 | 3 | 0.013942 | 0.012511 | 5.000 | `False` |
| `first_movement_speed` | 3 | 3 | 0.015886 | 0.013772 | 4.333 | `False` |
| `wheel_action_direction` | 3 | 3 | 0.029787 | 0.032426 | 5.667 | `False` |

## choice/action split on best axes

| metric | value |
|---|---:|
| mean action - choice best-axis dBA | 0.008895 |
| median action - choice best-axis dBA | 0.009384 |
| split supported | 2/3 |

## per-session best axes

| candidate | choice axis | speed axis | wheel axis | choice dBA | speed dBA | wheel dBA | action-choice dBA | split |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 6 | 7 | 4 | 0.012511 | 0.011364 | 0.032426 | 0.009384 | `True` |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 4 | 1 | 4 | 0.003597 | 0.022523 | 0.023084 | 0.019206 | `True` |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 3 | 4 | 3 | 0.025716 | 0.013772 | 0.033851 | -0.001905 | `False` |

## verdict

- This gate asks whether a single innovation component can carry target-specific readout.
- If action best-axis increments exceed choice best-axis increments, the next model should split directed action axes from policy/choice readout.
