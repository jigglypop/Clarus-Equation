# Mouse IBL/OpenAlyx choice policy/history gate

$$
y_{choice}=g(X_{policy/history},R,\hat H,\epsilon_{S_{train}})
$$

## setup

- candidates: 3
- outer folds: 5
- inner folds: 3
- components: 12
- subspace size: 3
- choice innovation residual supported: `False`
- strict policy dominance supported: `False`
- choice policy/history gate passed: `False`

## model summary

| model | mean BA | median BA |
|---|---:|---:|
| `policy_history` | 0.875217 | 0.861916 |
| `policy_history_region` | 0.893144 | 0.906732 |
| `policy_history_region_predicted_latent` | 0.892340 | 0.906732 |
| `policy_history_region_predicted_latent_nested_eps` | 0.908430 | 0.903768 |

## increment summary

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| `region_after_policy_history` | 1/3 | 0.017926 | 0.001799 | `False` |
| `predicted_latent_after_policy_history_region` | 1/3 | -0.000803 | 0.000000 | `False` |
| `nested_eps_after_policy_history_region_predicted_latent` | 2/3 | 0.016090 | 0.023697 | `False` |
| `all_neural_after_policy_history` | 2/3 | 0.033213 | 0.019338 | `False` |

## per-session

| candidate | policy BA | full BA | R after policy | Hhat after X,R | nested eps after X,R,Hhat | all neural after policy |
|---|---:|---:|---:|---:|---:|---:|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.858802 | 0.878140 | -0.006728 | 0.002370 | 0.023697 | 0.019338 |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 0.904933 | 0.903768 | 0.001799 | 0.000000 | -0.002963 | -0.001165 |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.861916 | 0.943381 | 0.058709 | -0.004779 | 0.027535 | 0.081465 |

## verdict

- If `nested eps after X,R,Hhat` stays weak, the choice failure is not fixed by reusing the action innovation subspace.
- If `all neural after policy` is also weak, the next choice term should be a richer policy/history latent rather than a neural innovation term.
