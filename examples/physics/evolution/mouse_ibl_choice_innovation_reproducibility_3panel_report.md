# Mouse IBL/OpenAlyx choice innovation-axis reproducibility gate

$$
y_{choice}=g(P_{rich},\epsilon_{S_{choice,train}})
$$

## setup

- candidates: 3
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- choice innovation reproducibility gate passed: `False`

## increment summary

| metric | value |
|---|---:|
| supported | 2/3 |
| mean dBA | 0.003211 |
| median dBA | 0.003096 |
| replicated | `False` |

## global axis summary

| metric | value |
|---|---:|
| top1 axis | 7 |
| top1 count | 7/45 |
| top1 share | 0.155556 |
| top3 axes | `[7, 3, 5]` |
| top3 count | 18/45 |
| top3 share | 0.400000 |
| entropy | 0.954592 |
| top1 null p | 0.696400 |
| stable identity | `False` |
| concentrated subspace | `False` |

## per-session

| candidate | policy BA | subspace BA | dBA | supported | top1 axis | top1 | top3 axes | top3 | entropy |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.854443 | 0.859183 | 0.004739 | `True` | 6 | 3 | `[6, 2, 4]` | 7 | 0.926992 |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 0.901970 | 0.903768 | 0.001799 | `False` | 5 | 3 | `[5, 8, 4]` | 8 | 0.838569 |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.869599 | 0.872695 | 0.003096 | `True` | 3 | 5 | `[3, 7, 12]` | 12 | 0.620817 |

## verdict

- Positive dBA means a choice-selected innovation subspace survives after richer policy.
- Axis concentration asks whether selected axes repeat across sessions and outer folds.
- If dBA replicates but axes do not concentrate, the term remains a session-adaptive innovation readout rather than a named stable subspace.
