# Mouse IBL/OpenAlyx directed latent-axis split gate

$$
y_{action}=g_a(X_t,R_t,\hat H_t,\epsilon_{t,k})
$$

## setup

- candidates: 12
- components: 12
- folds: 5
- directed axis gate passed: `True`

## target summary

| target | candidates | best axis supported | mean best dBA | median best dBA | mean positive axes | supported |
|---|---:|---:|---:|---:|---:|---|
| `choice_sign` | 12 | 11 | 0.009646 | 0.005360 | 3.917 | `True` |
| `first_movement_speed` | 12 | 11 | 0.020485 | 0.018147 | 4.750 | `True` |
| `wheel_action_direction` | 12 | 12 | 0.021819 | 0.018257 | 4.417 | `True` |

## choice/action split on best axes

| metric | value |
|---|---:|
| mean action - choice best-axis dBA | 0.011506 |
| median action - choice best-axis dBA | 0.011937 |
| split supported | 8/12 |

## per-session best axes

| candidate | choice axis | speed axis | wheel axis | choice dBA | speed dBA | wheel dBA | action-choice dBA | split |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 6 | 7 | 4 | 0.012511 | 0.011364 | 0.032426 | 0.009384 | `True` |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 4 | 1 | 4 | 0.003597 | 0.022523 | 0.023084 | 0.019206 | `True` |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 3 | 4 | 3 | 0.025716 | 0.013772 | 0.033851 | -0.001905 | `False` |
| `steinmetzlab_Subjects_NR_0029_2023-09-07_001` | 3 | 2 | 3 | 0.010746 | 0.040658 | 0.009813 | 0.014490 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-09-05_001` | 1 | 5 | 2 | 0.004977 | 0.007692 | 0.004405 | 0.001072 | `False` |
| `hausserlab_Subjects_PL050_2023-06-15_001` | 4 | 5 | 4 | 0.034695 | 0.006579 | 0.045049 | -0.008881 | `False` |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | 5 | 3 | 12 | 0.004292 | 0.011278 | 0.013430 | 0.008062 | `True` |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | 6 | 8 | 11 | 0.002632 | 0.043182 | 0.009524 | 0.023721 | `True` |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | 7 | 11 | 4 | 0.003035 | 0.033043 | 0.003815 | 0.015394 | `True` |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | 6 | 6 | 10 | 0.005744 | -0.001344 | 0.003767 | -0.004533 | `False` |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | 2 | 2 | 5 | 0.000000 | 0.029101 | 0.035955 | 0.032528 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | 2 | 4 | 3 | 0.007812 | 0.027971 | 0.046712 | 0.029529 | `True` |

## verdict

- This gate asks whether a single innovation component can carry target-specific readout.
- If action best-axis increments exceed choice best-axis increments, the next model should split directed action axes from policy/choice readout.
