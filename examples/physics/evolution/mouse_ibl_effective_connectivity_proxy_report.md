# Mouse IBL/OpenAlyx region-interaction effective-connectivity proxy gate

Flat-unit gate 다음 단계로, channel-rescued region rates의 pairwise interaction이 additive region bins보다 나은지 확인한다.
이 interaction은 causal connectivity가 아니라 trial-window effective coupling proxy다.

## setup

- candidates: 5
- folds: 5
- ridge: 1.0
- permutations: 200
- max units per probe: 96
- max interaction pairs: 750
- interaction supported over region: `False`
- interaction beats top unit: `False`
- effective-connectivity proxy passed: `False`

## model equation

$$
z_{iab}=r_{ia}r_{ib},\qquad a<b.
$$

The tested interaction model is

$$
R_i^{\mathrm{int}}=[R_i^{\mathrm{hybrid}},Z_i],
$$

and the main increments are

$$
\Delta_{\mathrm{int-region}}=\mathrm{BA}(R^{\mathrm{int}})-\mathrm{BA}(R^{\mathrm{hybrid}}),
\qquad
\Delta_{\mathrm{task+int}}=\mathrm{BA}([X^{\mathrm{task}},R^{\mathrm{int}}])-\mathrm{BA}([X^{\mathrm{task}},R^{\mathrm{hybrid}}]).
$$

## target replication

| target | candidates | int beats region | int beats unit | task+int beats task+region | task+int beats task+unit | mean region BA | mean int BA | mean unit BA | mean task+int BA | mean int-region delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `choice_sign` | 5 | 3 | 0 | 1 | 1 | 0.675560 | 0.668513 | 0.744883 | 0.822268 | -0.007047 |
| `first_movement_speed` | 5 | 3 | 1 | 3 | 2 | 0.742104 | 0.735488 | 0.756299 | 0.745631 | -0.006616 |
| `wheel_action_direction` | 5 | 2 | 0 | 0 | 3 | 0.716240 | 0.700149 | 0.777766 | 0.822171 | -0.016091 |

## candidate summaries

| candidate | trials | int>region | int>unit | task+int>task+region | task+int>task+unit | choice int-region | speed int-region | wheel int-region |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `witten29_thalamic_visual_reference` | 663 | 0 | 0 | 0 | 2 | -0.019573 | -0.019190 | -0.019955 |
| `nyu30_motor_striatal_multi_probe` | 933 | 2 | 0 | 2 | 0 | 0.004200 | 0.021640 | -0.021340 |
| `dy014_striatal_septal_probe` | 608 | 3 | 0 | 1 | 1 | 0.008361 | 0.022109 | 0.005625 |
| `dy011_motor_cortex_probe` | 402 | 1 | 1 | 1 | 1 | -0.032431 | 0.000113 | -0.067641 |
| `dy008_cp_somatosensory_thalamic_probe` | 409 | 2 | 0 | 0 | 2 | 0.004207 | -0.057753 | 0.022855 |

## verdict

- interaction supported over region: `False`
- interaction beats top unit: `False`
- effective-connectivity proxy passed: `False`

해석:

- Region interaction이 additive region보다 반복적으로 높으면, mouse 항에는 additive region identity보다 weighted interaction 항이 필요하다.
- Region interaction이 top-unit을 이기지 못하면, unit-detail residual은 여전히 남는다.
- 이 결과는 causal effective connectivity가 아니라 windowed interaction proxy다. 다음 강한 버전은 lagged coupling 또는 trial-split nested regularization이다.
