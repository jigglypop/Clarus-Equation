# Mouse IBL/OpenAlyx nested innovation-subspace gate

$$
y_t=g(X_t,R_t,\hat H_t,\epsilon_{t,S_{train}})
$$

## setup

- candidates: 3
- components: 12
- outer folds: 5
- inner folds: 3
- subspace size: 3
- nested subspace gate passed: `False`

## target summary

| target | candidates | subspace supported | mean dBA | median dBA | supported |
|---|---:|---:|---:|---:|---|
| `choice_sign` | 3 | 2 | 0.016090 | 0.023697 | `False` |
| `first_movement_speed` | 3 | 2 | 0.003416 | 0.008523 | `False` |
| `wheel_action_direction` | 3 | 3 | 0.033414 | 0.033859 | `False` |

## choice/action split

| metric | value |
|---|---:|
| mean action - choice subspace dBA | 0.002325 |
| median action - choice subspace dBA | -0.002506 |
| split supported | 1/3 |

## per-session split

| candidate | choice dBA | speed dBA | wheel dBA | action-choice dBA | split |
|---|---:|---:|---:|---:|---|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.023697 | 0.008523 | 0.033859 | -0.002506 | `False` |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | -0.002963 | 0.013514 | 0.029199 | 0.024320 | `True` |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.027535 | -0.011788 | 0.037184 | -0.014837 | `False` |

## verdict

- Axes are selected inside the outer train fold, so outer test trials do not choose the subspace.
- A positive result supports a reproducible train-selected innovation subspace rather than pure post-hoc best-axis selection.
