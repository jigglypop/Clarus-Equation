# Mouse IBL/OpenAlyx nested innovation-subspace gate

$$
y_t=g(X_t,R_t,\hat H_t,\epsilon_{t,S_{train}})
$$

## setup

- candidates: 12
- components: 12
- outer folds: 5
- inner folds: 3
- subspace size: 3
- nested subspace gate passed: `True`

## target summary

| target | candidates | subspace supported | mean dBA | median dBA | supported |
|---|---:|---:|---:|---:|---|
| `choice_sign` | 12 | 5 | 0.006472 | -0.000203 | `False` |
| `first_movement_speed` | 12 | 9 | 0.013697 | 0.011018 | `True` |
| `wheel_action_direction` | 12 | 8 | 0.020350 | 0.017363 | `True` |

## choice/action split

| metric | value |
|---|---:|
| mean action - choice subspace dBA | 0.010552 |
| median action - choice subspace dBA | 0.007308 |
| split supported | 8/12 |

## per-session split

| candidate | choice dBA | speed dBA | wheel dBA | action-choice dBA | split |
|---|---:|---:|---:|---:|---|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.023697 | 0.008523 | 0.033859 | -0.002506 | `False` |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | -0.002963 | 0.013514 | 0.029199 | 0.024320 | `True` |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.027535 | -0.011788 | 0.037184 | -0.014837 | `False` |
| `steinmetzlab_Subjects_NR_0029_2023-09-07_001` | -0.007456 | 0.040673 | -0.002005 | 0.026790 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-09-05_001` | 0.004977 | 0.000000 | 0.004405 | -0.002774 | `False` |
| `hausserlab_Subjects_PL050_2023-06-15_001` | 0.048234 | -0.011513 | 0.060967 | -0.023507 | `False` |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | -0.002005 | 0.003759 | -0.002856 | 0.002456 | `True` |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | -0.011376 | 0.034091 | 0.005526 | 0.031184 | `True` |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | 0.001598 | 0.022069 | 0.000189 | 0.009531 | `True` |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | -0.004579 | 0.002728 | -0.001718 | 0.005084 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | -0.005208 | 0.031746 | 0.038360 | 0.040261 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | 0.005208 | 0.030560 | 0.041092 | 0.030618 | `True` |

## verdict

- Axes are selected inside the outer train fold, so outer test trials do not choose the subspace.
- A positive result supports a reproducible train-selected innovation subspace rather than pure post-hoc best-axis selection.
