# Mouse IBL/OpenAlyx choice innovation-axis reproducibility gate

$$
y_{choice}=g(P_{rich},\epsilon_{S_{choice,train}})
$$

## setup

- candidates: 24
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- choice innovation reproducibility gate passed: `False`

## increment summary

| metric | value |
|---|---:|
| supported | 8/24 |
| mean dBA | 0.001288 |
| median dBA | 0.000298 |
| replicated | `False` |

## global axis summary

| metric | value |
|---|---:|
| top1 axis | 3 |
| top1 count | 36/360 |
| top1 share | 0.100000 |
| top3 axes | `[3, 6, 1]` |
| top3 count | 107/360 |
| top3 share | 0.297222 |
| entropy | 0.995436 |
| top1 null p | 0.930300 |
| stable identity | `False` |
| concentrated subspace | `True` |

## per-session

| candidate | policy BA | subspace BA | dBA | supported | top1 axis | top1 | top3 axes | top3 | entropy |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.854443 | 0.859183 | 0.004739 | `True` | 6 | 3 | `[6, 2, 4]` | 7 | 0.926992 |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 0.901970 | 0.903768 | 0.001799 | `False` | 5 | 3 | `[5, 8, 4]` | 8 | 0.838569 |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.869599 | 0.872695 | 0.003096 | `True` | 3 | 5 | `[3, 7, 12]` | 12 | 0.620817 |
| `steinmetzlab_Subjects_NR_0029_2023-09-07_001` | 0.786952 | 0.788158 | 0.001206 | `False` | 2 | 3 | `[2, 4, 11]` | 7 | 0.852607 |
| `steinmetzlab_Subjects_NR_0029_2023-09-05_001` | 0.858823 | 0.858213 | -0.000610 | `False` | 3 | 2 | `[3, 12, 5]` | 6 | 0.941030 |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | 0.850446 | 0.845520 | -0.004926 | `False` | 1 | 4 | `[1, 5, 4]` | 9 | 0.741030 |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | 0.783125 | 0.770625 | -0.012500 | `False` | 5 | 3 | `[5, 11, 2]` | 7 | 0.852607 |
| `steinmetzlab_Subjects_NR_0029_2023-08-29_001` | 0.869156 | 0.860269 | -0.008887 | `False` | 3 | 3 | `[3, 2, 6]` | 7 | 0.889800 |
| `hausserlab_Subjects_PL050_2023-06-15_001` | 0.872290 | 0.919134 | 0.046845 | `True` | 4 | 5 | `[4, 8, 9]` | 11 | 0.687942 |
| `hausserlab_Subjects_PL050_2023-06-13_001` | 0.865215 | 0.861016 | -0.004198 | `False` | 11 | 3 | `[11, 3, 5]` | 8 | 0.801377 |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | 0.762756 | 0.758824 | -0.003932 | `False` | 12 | 2 | `[12, 10, 6]` | 6 | 0.866645 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | 0.851454 | 0.848729 | -0.002725 | `False` | 9 | 3 | `[9, 5, 11]` | 7 | 0.852607 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | 0.833759 | 0.829287 | -0.004472 | `False` | 6 | 3 | `[6, 7, 1]` | 8 | 0.838569 |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | 0.819607 | 0.815284 | -0.004323 | `False` | 2 | 3 | `[2, 10, 9]` | 7 | 0.852607 |
| `churchlandlab_ucla_Subjects_MFD_07_2023-08-31_001` | 0.812457 | 0.811432 | -0.001025 | `False` | 3 | 3 | `[3, 2, 4]` | 9 | 0.787339 |
| `churchlandlab_ucla_Subjects_MFD_07_2023-08-29_001` | 0.855401 | 0.856682 | 0.001281 | `False` | 9 | 3 | `[9, 7, 12]` | 7 | 0.815415 |
| `churchlandlab_ucla_Subjects_MFD_06_2023-08-29_001` | 0.705116 | 0.722043 | 0.016927 | `True` | 4 | 3 | `[4, 1, 8]` | 7 | 0.889800 |
| `churchlandlab_ucla_Subjects_MFD_06_2023-08-25_001` | 0.837991 | 0.840800 | 0.002809 | `True` | 10 | 2 | `[10, 1, 3]` | 6 | 0.866645 |
| `churchlandlab_ucla_Subjects_MFD_06_2023-08-24_001` | 0.840498 | 0.844020 | 0.003522 | `True` | 3 | 4 | `[3, 1, 2]` | 9 | 0.778222 |
| `churchlandlab_ucla_Subjects_MFD_06_2023-08-23_001` | 0.835078 | 0.828735 | -0.006342 | `False` | 12 | 2 | `[12, 7, 1]` | 6 | 0.866645 |
| `churchlandlab_ucla_Subjects_MFD_06_2023-08-22_001` | 0.831478 | 0.833797 | 0.002319 | `True` | 6 | 3 | `[6, 1, 9]` | 8 | 0.875762 |
| `churchlandlab_ucla_Subjects_MFD_05_2023-08-17_001` | 0.802102 | 0.806318 | 0.004216 | `True` | 2 | 3 | `[2, 11, 10]` | 8 | 0.801377 |
| `churchlandlab_ucla_Subjects_MFD_05_2023-08-16_001` | 0.811553 | 0.813214 | 0.001661 | `False` | 1 | 3 | `[1, 12, 10]` | 7 | 0.815415 |
| `churchlandlab_ucla_Subjects_MFD_05_2023-08-15_001` | 0.821532 | 0.815954 | -0.005578 | `False` | 7 | 4 | `[7, 9, 10]` | 9 | 0.778222 |

## verdict

- Positive dBA means a choice-selected innovation subspace survives after richer policy.
- Axis concentration asks whether selected axes repeat across sessions and outer folds.
- If dBA replicates but axes do not concentrate, the term remains a session-adaptive innovation readout rather than a named stable subspace.
