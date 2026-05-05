# Mouse IBL/OpenAlyx richer choice policy/history gate

$$
y_{choice}=g(P_{rich},R,\hat H,\epsilon_{S_{train}})
$$

## setup

- candidates: 3
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- max policy lag: 5
- richer policy supported: `False`
- choice neural residual after richer policy supported: `False`
- choice nested eps after richer policy supported: `False`
- richer choice policy gate passed: `False`

## model summary

| model | mean BA | median BA |
|---|---:|---:|
| `linear_policy_history` | 0.875217 | 0.861916 |
| `richer_policy_history` | 0.875337 | 0.869599 |
| `richer_policy_history_region` | 0.886812 | 0.903186 |
| `richer_policy_history_region_predicted_latent` | 0.886916 | 0.893662 |
| `richer_policy_history_region_predicted_latent_nested_eps` | 0.896429 | 0.900223 |

## increment summary

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| `richer_policy_after_linear_policy` | 1/3 | 0.000120 | -0.002963 | `False` |
| `region_after_richer_policy` | 1/3 | 0.011475 | 0.001216 | `False` |
| `predicted_latent_after_richer_policy_region` | 1/3 | 0.000104 | 0.000136 | `False` |
| `nested_eps_after_richer_policy_region_predicted_latent` | 3/3 | 0.009513 | 0.006560 | `False` |
| `all_neural_after_richer_policy` | 1/3 | 0.021092 | -0.000884 | `False` |

## per-session

| candidate | linear BA | richer BA | full BA | rich-linear | R after rich | Hhat after rich,R | nested eps after rich,R,Hhat | all neural after rich |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.858802 | 0.854443 | 0.853559 | -0.004359 | -0.013174 | 0.009700 | 0.002591 | -0.000884 |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 0.904933 | 0.901970 | 0.900223 | -0.002963 | 0.001216 | -0.009524 | 0.006560 | -0.001747 |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.861916 | 0.869599 | 0.935506 | 0.007683 | 0.046382 | 0.000136 | 0.019389 | 0.065907 |

## verdict

- `richer policy after linear policy` tests whether multi-lag and interaction policy features improve choice beyond the previous linear task/history block.
- `nested eps after rich,R,Hhat` tests whether the action-style innovation subspace still replicates as a choice residual after the richer policy block.
