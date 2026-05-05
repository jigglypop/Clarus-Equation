# Mouse IBL/OpenAlyx choice innovation-axis reproducibility gate

$$
y_{choice}=g(P_{rich},\epsilon_{S_{choice,train}})
$$

## setup

- candidates: 12
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- choice innovation reproducibility gate passed: `False`

## increment summary

| metric | value |
|---|---:|
| supported | 3/12 |
| mean dBA | 0.002016 |
| median dBA | -0.001667 |
| replicated | `False` |

## global axis summary

| metric | value |
|---|---:|
| top1 axis | 5 |
| top1 count | 20/180 |
| top1 share | 0.111111 |
| top3 axes | `[5, 6, 4]` |
| top3 count | 58/180 |
| top3 share | 0.322222 |
| entropy | 0.991867 |
| top1 null p | 0.857200 |
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
| `hausserlab_Subjects_PL050_2023-06-15_001` | 0.872290 | 0.919134 | 0.046845 | `True` | 4 | 5 | `[4, 8, 9]` | 11 | 0.687942 |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | 0.762756 | 0.758824 | -0.003932 | `False` | 12 | 2 | `[12, 10, 6]` | 6 | 0.866645 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | 0.851454 | 0.848729 | -0.002725 | `False` | 9 | 3 | `[9, 5, 11]` | 7 | 0.852607 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | 0.833759 | 0.829287 | -0.004472 | `False` | 6 | 3 | `[6, 7, 1]` | 8 | 0.838569 |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | 0.819607 | 0.815284 | -0.004323 | `False` | 2 | 3 | `[2, 10, 9]` | 7 | 0.852607 |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | 0.850446 | 0.845520 | -0.004926 | `False` | 1 | 4 | `[1, 5, 4]` | 9 | 0.741030 |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | 0.783125 | 0.770625 | -0.012500 | `False` | 5 | 3 | `[5, 11, 2]` | 7 | 0.852607 |

## verdict

- Positive dBA means a choice-selected innovation subspace survives after richer policy.
- Axis concentration asks whether selected axes repeat across sessions and outer folds.
- If dBA replicates but axes do not concentrate, the term remains a session-adaptive innovation readout rather than a named stable subspace.
