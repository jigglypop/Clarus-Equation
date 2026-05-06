# Mouse IBL/OpenAlyx nested innovation-subspace gate

$$
y_t=g(X_t,R_t,\hat H_t,\epsilon_{t,S_{train}})
$$

## setup

- candidates: 1
- components: 12
- outer folds: 5
- inner folds: 3
- subspace size: 3
- nested subspace gate passed: `False`

## target summary

| target | candidates | subspace supported | mean dBA | median dBA | supported |
|---|---:|---:|---:|---:|---|
| `choice_sign` | 1 | 0 | -0.002591 | -0.002591 | `False` |
| `first_movement_speed` | 1 | 0 | -0.014205 | -0.014205 | `False` |
| `wheel_action_direction` | 1 | 0 | -0.005172 | -0.005172 | `False` |

## choice/action split

| metric | value |
|---|---:|
| mean action - choice subspace dBA | -0.007098 |
| median action - choice subspace dBA | -0.007098 |
| split supported | 0/1 |

## per-session split

| candidate | choice dBA | speed dBA | wheel dBA | action-choice dBA | split |
|---|---:|---:|---:|---:|---|
| `C:\Users\22310326\Downloads\ONE\openalyx.internationalbrainlab.org\steinmetzlab\Subjects\NR_0031\2023-07-14\001` | -0.002591 | -0.014205 | -0.005172 | -0.007098 | `False` |

## verdict

- Axes are selected inside the outer train fold, so outer test trials do not choose the subspace.
- A positive result supports a reproducible train-selected innovation subspace rather than pure post-hoc best-axis selection.
