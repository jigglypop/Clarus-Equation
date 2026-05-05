# Mouse IBL/OpenAlyx choice/action innovation split

$$
y_{choice}=g_c(X_t,R_t,\hat H_t)
\qquad
y_{action}=g_a(X_t,R_t,\hat H_t,\epsilon_t)
$$

## setup

- source panel: `/Users/yeomdonghwan/Desktop/clarus/examples/physics/evolution/mouse_ibl_innovation_behavior_12panel_results.json`
- candidates: 12
- min split delta: 0.002
- split gate passed: `True`

## summary

| metric | value |
|---|---:|
| mean choice eps after Hhat | 0.004624 |
| mean action eps after Hhat | 0.023320 |
| mean action - choice | 0.018696 |
| median action - choice | 0.015874 |
| action - choice 95% bootstrap low | 0.006277 |
| action - choice 95% bootstrap high | 0.031995 |
| sign-flip p | 0.017090 |
| choice supported | 3/12 |
| any action supported | 10/12 |
| both actions supported | 6/12 |
| split supported | 9/12 |

## per-session split

| candidate | choice | speed | wheel | action mean | action - choice | split |
|---|---:|---:|---:|---:|---:|---|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.036429 | 0.039773 | 0.064415 | 0.052094 | 0.015665 | `True` |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | -0.008890 | 0.011261 | 0.040551 | 0.025906 | 0.034796 | `True` |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.027535 | 0.007859 | 0.040460 | 0.024159 | -0.003376 | `False` |
| `steinmetzlab_Subjects_NR_0029_2023-09-07_001` | -0.011952 | 0.051528 | -0.002005 | 0.024761 | 0.036713 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-09-05_001` | 0.000000 | 0.002564 | 0.007153 | 0.004858 | 0.004858 | `True` |
| `hausserlab_Subjects_PL050_2023-06-15_001` | 0.047195 | -0.004934 | 0.076257 | 0.035662 | -0.011534 | `False` |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | -0.009868 | -0.005639 | -0.003953 | -0.004796 | 0.005072 | `True` |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | -0.002818 | 0.025000 | 0.001528 | 0.013264 | 0.016082 | `True` |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | -0.002082 | 0.033073 | -0.002491 | 0.015291 | 0.017373 | `True` |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | -0.004579 | -0.016220 | -0.007204 | -0.011712 | -0.007133 | `False` |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | -0.010276 | 0.015873 | 0.066017 | 0.040945 | 0.051221 | `True` |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | -0.005208 | 0.040752 | 0.078069 | 0.059410 | 0.064619 | `True` |

## verdict

- The larger panel supports an action-linked innovation term more strongly than a choice-linked innovation term.
- The next model should split choice and action readouts instead of forcing one behavioral equation for all targets.
