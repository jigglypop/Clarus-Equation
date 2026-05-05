# Mouse IBL/OpenAlyx block-regularized unit residual gate

Common-ridge nested gate 다음 반례는 unit block penalty mismatch다.
이 gate는 outer fold의 train split 내부에서만 \(\lambda_U\)를 고르고, held-out outer fold에서 block-regularized residual을 평가한다.

## setup

- candidates: 5
- folds: 5
- inner folds: 3
- task penalty: 1.0
- region penalty: 1.0
- unit penalties: `[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]`
- min unit spikes: 1000
- max units per probe: 192
- block unit residual after task+region supported: `True`
- region residual after block task+unit supported: `True`
- block-regularized unit gate passed: `True`

## block equation

$$
\hat\beta_{\lambda_U}^{(-k)}
=
\arg\min_\beta
\left\|y-Z\beta\right\|_2^2
+
\lambda_X\|\beta_X\|_2^2
+
\lambda_R\|\beta_R\|_2^2
+
\lambda_U\|\beta_U\|_2^2.
$$

The unit penalty is chosen only on the outer-train split:

$$
\lambda_U^{*(-k)}
=
\arg\max_{\lambda_U\in\Lambda_U}
\mathrm{BA}_{\mathrm{inner}}(\lambda_U).
$$

The main residual is

$$
\Delta_{U\mid X,R}^{\mathrm{block}}
=
\mathrm{BA}(M_{XRU}^{\mathrm{block}})
-
\mathrm{BA}(M_{XR}).
$$

## target replication

| target | candidates | block unit residual count | region residual count | unit>region after task | mean task BA | mean task+region BA | mean block task+unit BA | mean block task+region+unit BA | mean block unit residual | mean region residual | median lambda_U XRU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 5 | 4 | 2 | 4 | 0.844886 | 0.845811 | 0.866224 | 0.862736 | 0.016925 | -0.003489 | 300 |
| `first_movement_speed` | 5 | 5 | 0 | 5 | 0.689940 | 0.750006 | 0.798966 | 0.784605 | 0.034599 | -0.014360 | 1000 |
| `wheel_action_direction` | 5 | 5 | 4 | 3 | 0.822181 | 0.837202 | 0.855839 | 0.865689 | 0.028487 | 0.009850 | 300 |

## candidate summaries

| candidate | trials | unit residuals | region residuals | choice U_block_given_XR | speed U_block_given_XR | wheel U_block_given_XR | choice R_given_XU_block | speed R_given_XU_block | wheel R_given_XU_block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 2 | 1 | -0.001667 | 0.012790 | 0.007205 | -0.001667 | -0.009590 | 0.010550 |
| `nyu30_motor_striatal_multi_probe` | 933 | 3 | 2 | 0.026014 | 0.049238 | 0.042709 | 0.002924 | -0.009610 | 0.002874 |
| `dy014_striatal_septal_probe` | 608 | 3 | 1 | 0.027454 | 0.100340 | 0.015919 | -0.016949 | -0.006803 | 0.020789 |
| `dy011_motor_cortex_probe` | 402 | 3 | 1 | 0.010307 | 0.010596 | 0.028128 | 0.000729 | -0.026554 | -0.007590 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 3 | 1 | 0.022516 | 0.000030 | 0.048476 | -0.002481 | -0.019246 | 0.022628 |

## verdict

- block unit residual after task+region supported: `True`
- region residual after block task+unit supported: `True`
- block-regularized unit gate passed: `True`

해석:

- 이 gate가 양성이면 common-ridge all-unit 음성 결과는 penalty mismatch였다고 볼 수 있다.
- 이 gate도 음성이면 unit identity는 flat comparison에서는 강하지만, \([X,R]\) 뒤의 독립 additive 항으로는 아직 승격되지 않는다.
- 다음 단계는 single-trial behavior target에 대한 temporal GLM coupling이다.
