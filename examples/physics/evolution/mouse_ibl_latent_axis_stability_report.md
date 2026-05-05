# Mouse IBL/OpenAlyx latent-axis stability

$$
\epsilon_{t,k_j}\quad\mathrm{stable?}
$$

## setup

- source: `/Users/yeomdonghwan/Desktop/clarus/examples/physics/evolution/mouse_ibl_directed_latent_axis_split_results.json`
- candidates: 12
- components: 12
- axis identity gate passed: `False`
- subspace gate passed: `False`

## target axis distribution

| target | top axis | top1 | top3 | entropy | top1 null p | mean best dBA | stable identity | concentrated subspace |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `choice_sign` | 6 | 3/12 | 7/12 | 0.750000 | 0.687350 | 0.009646 | `False` | `False` |
| `first_movement_speed` | 4 | 2/12 | 6/12 | 0.860529 | 0.999950 | 0.020485 | `False` | `False` |
| `wheel_action_direction` | 4 | 4/12 | 8/12 | 0.703510 | 0.167700 | 0.021819 | `False` | `True` |

## within-session sharing

| metric | count |
|---|---:|
| choice-speed same axis | 2/12 |
| choice-wheel same axis | 4/12 |
| speed-wheel same axis | 0/12 |
| all three same axis | 0/12 |
| all three axes <= 6 | 7/12 |

## per-session axes

| candidate | choice | speed | wheel | unique axes | all low |
|---|---:|---:|---:|---:|---|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 6 | 7 | 4 | 3 | `False` |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 4 | 1 | 4 | 2 | `True` |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 3 | 4 | 3 | 2 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-09-07_001` | 3 | 2 | 3 | 2 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-09-05_001` | 1 | 5 | 2 | 3 | `True` |
| `hausserlab_Subjects_PL050_2023-06-15_001` | 4 | 5 | 4 | 2 | `True` |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | 5 | 3 | 12 | 3 | `False` |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | 6 | 8 | 11 | 3 | `False` |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | 7 | 11 | 4 | 3 | `False` |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | 6 | 6 | 10 | 2 | `False` |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | 2 | 2 | 5 | 2 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | 2 | 4 | 3 | 3 | `True` |

## verdict

- Single best axes are behavior-informative, but their identity is not yet stable enough to name one shared axis.
- The next test should use a pre-registered low-dimensional subspace or nested axis selection inside outer folds.
