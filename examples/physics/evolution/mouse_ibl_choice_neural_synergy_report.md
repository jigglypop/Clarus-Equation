# Mouse IBL/OpenAlyx choice neural-block synergy gate

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
- full after policy supported: `True`
- synergy over best single supported: `False`
- synergy over best pair supported: `False`
- choice neural synergy gate passed: `False`

## model summary

| model | blocks | mean BA | median BA |
|---|---|---:|---:|
| `P` | `none` | 0.837102 | 0.850950 |
| `P_EPS_SUB` | `EPS_SUB` | 0.842266 | 0.852734 |
| `P_HHAT` | `HHAT` | 0.840806 | 0.854294 |
| `P_R` | `R` | 0.839692 | 0.846948 |
| `P_HHAT_EPS_SUB` | `HHAT,EPS_SUB` | 0.848622 | 0.859746 |
| `P_R_EPS_SUB` | `R,EPS_SUB` | 0.844350 | 0.850032 |
| `P_R_HHAT` | `R,HHAT` | 0.840482 | 0.851776 |
| `P_R_HHAT_EPS_SUB` | `R,HHAT,EPS_SUB` | 0.846516 | 0.853543 |

## increment summary

| increment | positive | mean dBA | median dBA | supported |
|---|---:|---:|---:|---|
| `full_after_policy` | 7/12 | 0.009414 | 0.002078 | `True` |
| `full_minus_best_single` | 3/12 | -0.000933 | -0.000822 | `False` |
| `full_minus_best_pair` | 1/12 | -0.003688 | -0.004323 | `False` |
| `best_single_after_policy` | 8/12 | 0.010347 | 0.002667 | `True` |
| `best_pair_after_policy` | 5/12 | 0.013102 | 0.001518 | `False` |

## best model counts

- best single counts: `{'P_EPS_SUB': 6, 'P_HHAT': 3, 'P_R': 3}`
- best pair counts: `{'P_HHAT_EPS_SUB': 6, 'P_R_EPS_SUB': 4, 'P_R_HHAT': 2}`

## per-session

| candidate | policy BA | full BA | best single | best pair | full-policy | full-best-single | full-best-pair |
|---|---:|---:|---|---|---:|---:|---:|
| `steinmetzlab_Subjects_NR_0031_2023-07-14_001` | 0.854443 | 0.853559 | `P_EPS_SUB` | `P_HHAT_EPS_SUB` | -0.000884 | -0.003254 | -0.007109 |
| `steinmetzlab_Subjects_NR_0031_2023-07-12_001` | 0.901970 | 0.900223 | `P_EPS_SUB` | `P_R_EPS_SUB` | -0.001747 | -0.004711 | -0.001165 |
| `hausserlab_Subjects_PL050_2023-06-12_001` | 0.869599 | 0.935506 | `P_HHAT` | `P_HHAT_EPS_SUB` | 0.065907 | 0.012655 | -0.009830 |
| `steinmetzlab_Subjects_NR_0029_2023-09-07_001` | 0.786952 | 0.789035 | `P_R` | `P_R_EPS_SUB` | 0.002083 | 0.000877 | 0.000877 |
| `steinmetzlab_Subjects_NR_0029_2023-09-05_001` | 0.858823 | 0.861616 | `P_R` | `P_R_HHAT` | 0.002793 | 0.000610 | 0.000964 |
| `hausserlab_Subjects_PL050_2023-06-15_001` | 0.872290 | 0.923652 | `P_EPS_SUB` | `P_HHAT_EPS_SUB` | 0.051362 | 0.013714 | -0.004854 |
| `churchlandlab_ucla_Subjects_MFD_09_2023-10-19_001` | 0.762756 | 0.759826 | `P_EPS_SUB` | `P_R_HHAT` | -0.002929 | -0.000141 | -0.003791 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-08_001` | 0.851454 | 0.853526 | `P_HHAT` | `P_HHAT_EPS_SUB` | 0.002072 | -0.001502 | -0.007615 |
| `churchlandlab_ucla_Subjects_MFD_08_2023-09-07_001` | 0.833759 | 0.817807 | `P_R` | `P_R_EPS_SUB` | -0.015951 | -0.015951 | -0.009574 |
| `churchlandlab_ucla_Subjects_MFD_07_2023-09-01_001` | 0.819607 | 0.809737 | `P_HHAT` | `P_HHAT_EPS_SUB` | -0.009869 | -0.010323 | -0.007679 |
| `steinmetzlab_Subjects_NR_0029_2023-08-31_001` | 0.850446 | 0.862762 | `P_EPS_SUB` | `P_HHAT_EPS_SUB` | 0.012315 | -0.005773 | -0.002604 |
| `steinmetzlab_Subjects_NR_0029_2023-08-30_001` | 0.783125 | 0.790938 | `P_EPS_SUB` | `P_R_EPS_SUB` | 0.007812 | 0.002604 | 0.008125 |

## verdict

- `full after policy` tests whether the combined neural block remains after richer policy.
- `full minus best single` asks whether the full neural block beats any one of R, Hhat, or EPS alone.
- `full minus best pair` is the stricter three-way synergy test.
