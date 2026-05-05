# Mouse IBL/OpenAlyx richer choice policy/history gate

$$
y_{choice}=g(P_{rich},R,\hat H,\epsilon_{S_{train}})
$$

## setup

- candidates: 12
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- max policy lag: 5
- richer policy supported: `False`
- choice neural residual after richer policy supported: `True`
- choice nested eps after richer policy supported: `False`
- richer choice policy gate passed: `False`

## model summary

| model | mean BA | median BA |
|---|---:|---:|
| `linear_policy_history` | 0.836522 | 0.843374 |
| `richer_policy_history` | 0.837102 | 0.850950 |
| `richer_policy_history_region` | 0.839692 | 0.846948 |
| `richer_policy_history_region_predicted_latent` | 0.840482 | 0.851776 |
| `richer_policy_history_region_predicted_latent_nested_eps` | 0.846516 | 0.853543 |

## increment summary

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| `richer_policy_after_linear_policy` | 5/12 | 0.000580 | -0.000956 | `False` |
| `region_after_richer_policy` | 5/12 | 0.002590 | 0.001219 | `False` |
| `predicted_latent_after_richer_policy_region` | 5/12 | 0.000791 | 0.000021 | `False` |
| `nested_eps_after_richer_policy_region_predicted_latent` | 5/12 | 0.006033 | 0.001085 | `False` |
| `all_neural_after_richer_policy` | 7/12 | 0.009414 | 0.002078 | `True` |

## per-session

| candidate | linear BA | richer BA | full BA | rich-linear | R after rich | Hhat after rich,R | nested eps after rich,R,Hhat | all neural after rich |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.858802 | 0.854443 | 0.853559 | -0.004359 | -0.013174 | 0.009700 | 0.002591 | -0.000884 |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 0.904933 | 0.901970 | 0.900223 | -0.002963 | 0.001216 | -0.009524 | 0.006560 | -0.001747 |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.861916 | 0.869599 | 0.935506 | 0.007683 | 0.046382 | 0.000136 | 0.019389 | 0.065907 |
| `steinmetzlab_Subjects_NR_0029_2023-09-07_001` | 0.794189 | 0.786952 | 0.789035 | -0.007237 | 0.001206 | -0.000329 | 0.001206 | 0.002083 |
| `steinmetzlab_Subjects_NR_0029_2023-09-05_001` | 0.865275 | 0.858823 | 0.861616 | -0.006453 | 0.002183 | -0.000354 | 0.000964 | 0.002793 |
| `hausserlab_Subjects_PL050_2023-06-15_001` | 0.856688 | 0.872290 | 0.923652 | 0.015601 | 0.003641 | 0.008333 | 0.039388 | 0.051362 |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | 0.778921 | 0.762756 | 0.759826 | -0.016165 | -0.009367 | 0.010228 | -0.003791 | -0.002929 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | 0.837354 | 0.851454 | 0.853526 | 0.014101 | 0.001222 | -0.000093 | 0.000943 | 0.002072 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | 0.827542 | 0.833759 | 0.817807 | 0.006216 | 0.000000 | -0.014998 | -0.000953 | -0.015951 |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | 0.791794 | 0.819607 | 0.809737 | 0.027813 | -0.009415 | 0.006257 | -0.006711 | -0.009869 |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | 0.849394 | 0.850446 | 0.862762 | 0.001052 | 0.002181 | 0.010134 | 0.000000 | 0.012315 |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | 0.811458 | 0.783125 | 0.790938 | -0.028333 | 0.005000 | -0.010000 | 0.012813 | 0.007812 |

## verdict

- `richer policy after linear policy` tests whether multi-lag and interaction policy features improve choice beyond the previous linear task/history block.
- `nested eps after rich,R,Hhat` tests whether the action-style innovation subspace still replicates as a choice residual after the richer policy block.
