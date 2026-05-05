# Mouse IBL/OpenAlyx choice neural-block synergy gate

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
- full after policy supported: `False`
- synergy over best single supported: `False`
- synergy over best pair supported: `False`
- choice neural synergy gate passed: `False`

## model summary

| model | blocks | mean BA | median BA |
|---|---|---:|---:|
| `P` | `none` | 0.875337 | 0.869599 |
| `P_EPS_SUB` | `EPS_SUB` | 0.879224 | 0.875926 |
| `P_HHAT` | `HHAT` | 0.890412 | 0.894827 |
| `P_R` | `R` | 0.886812 | 0.903186 |
| `P_HHAT_EPS_SUB` | `HHAT,EPS_SUB` | 0.901864 | 0.899589 |
| `P_R_EPS_SUB` | `R,EPS_SUB` | 0.895814 | 0.901387 |
| `P_R_HHAT` | `R,HHAT` | 0.886916 | 0.893662 |
| `P_R_HHAT_EPS_SUB` | `R,HHAT,EPS_SUB` | 0.896429 | 0.900223 |

## increment summary

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| `full_after_policy` | 1/3 | 0.021092 | -0.000884 | `False` |
| `full_minus_best_single` | 1/3 | 0.001564 | -0.003254 | `False` |
| `full_minus_best_pair` | 0/3 | -0.006035 | -0.007109 | `False` |
| `best_single_after_policy` | 3/3 | 0.019528 | 0.002963 | `False` |
| `best_pair_after_policy` | 2/3 | 0.027126 | 0.006225 | `False` |

## best model counts

- best single counts: `{'P_EPS_SUB': 2, 'P_HHAT': 1}`
- best pair counts: `{'P_HHAT_EPS_SUB': 2, 'P_R_EPS_SUB': 1}`

## per-session

| candidate | policy BA | full BA | best single | best pair | full-policy | full-best-single | full-best-pair |
|---|---:|---:|---|---|---:|---:|---:|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.854443 | 0.853559 | `P_EPS_SUB` | `P_HHAT_EPS_SUB` | -0.000884 | -0.003254 | -0.007109 |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 0.901970 | 0.900223 | `P_EPS_SUB` | `P_R_EPS_SUB` | -0.001747 | -0.004711 | -0.001165 |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.869599 | 0.935506 | `P_HHAT` | `P_HHAT_EPS_SUB` | 0.065907 | 0.012655 | -0.009830 |

## verdict

- `full after policy` tests whether the combined neural block remains after richer policy.
- `full minus best single` asks whether the full neural block beats any one of R, Hhat, or EPS alone.
- `full minus best pair` is the stricter three-way synergy test.
