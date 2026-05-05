# Mouse IBL/OpenAlyx choice policy/history gate

$$
y_{choice}=g(X_{policy/history},R,\hat H,\epsilon_{S_{train}})
$$

## setup

- candidates: 12
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- choice innovation residual supported: `False`
- strict policy dominance supported: `False`
- choice policy/history gate passed: `True`

## model summary

| model | mean BA | median BA |
|---|---:|---:|
| `policy_history` | 0.836522 | 0.843374 |
| `policy_history_region` | 0.839474 | 0.844699 |
| `policy_history_region_predicted_latent` | 0.841794 | 0.849048 |
| `policy_history_region_predicted_latent_nested_eps` | 0.848266 | 0.843158 |

## increment summary

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| `region_after_policy_history` | 4/12 | 0.002952 | -0.000510 | `False` |
| `predicted_latent_after_policy_history_region` | 7/12 | 0.002320 | 0.002501 | `True` |
| `nested_eps_after_policy_history_region_predicted_latent` | 5/12 | 0.006472 | -0.000203 | `False` |
| `all_neural_after_policy_history` | 5/12 | 0.011744 | -0.000156 | `False` |

## per-session

| candidate | policy BA | full BA | R after policy | Hhat after X,R | nested eps after X,R,Hhat | all neural after policy |
|---|---:|---:|---:|---:|---:|---:|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.858802 | 0.878140 | -0.006728 | 0.002370 | 0.023697 | 0.019338 |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 0.904933 | 0.903768 | 0.001799 | 0.000000 | -0.002963 | -0.001165 |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.861916 | 0.943381 | 0.058709 | -0.004779 | 0.027535 | 0.081465 |
| `steinmetzlab_Subjects_NR_0029_2023-09-07_001` | 0.794189 | 0.788158 | -0.001206 | 0.002632 | -0.007456 | -0.006031 |
| `steinmetzlab_Subjects_NR_0029_2023-09-05_001` | 0.865275 | 0.865275 | -0.004977 | 0.000000 | 0.004977 | 0.000000 |
| `hausserlab_Subjects_PL050_2023-06-15_001` | 0.856688 | 0.927468 | 0.010572 | 0.011974 | 0.048234 | 0.070779 |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | 0.778921 | 0.754172 | -0.027396 | 0.004652 | -0.002005 | -0.024749 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | 0.837354 | 0.832277 | 0.000187 | 0.006112 | -0.011376 | -0.005077 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | 0.827542 | 0.824346 | -0.001921 | -0.002874 | 0.001598 | -0.003196 |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | 0.791794 | 0.797025 | -0.003158 | 0.012968 | -0.004579 | 0.005232 |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | 0.849394 | 0.854038 | 0.002463 | 0.007389 | -0.005208 | 0.004644 |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | 0.811458 | 0.811146 | 0.007083 | -0.012604 | 0.005208 | -0.000313 |

## verdict

- If `nested eps after X,R,Hhat` stays weak, the choice failure is not fixed by reusing the action innovation subspace.
- If `all neural after policy` is also weak, the next choice term should be a richer policy/history latent rather than a neural innovation term.
